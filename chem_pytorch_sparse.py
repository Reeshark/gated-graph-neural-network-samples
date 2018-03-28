#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_sparse.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format).
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format).
    --log_dir DIR            Log dir name.
    --data_dir DIR           Data dir name.
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
"""
from collections import defaultdict
import numpy as np
import torch
import sys, traceback
#import tensorflow as tf
from chem_pytorch import ChemModel
from utils_pytorch import glorot_init, SMALL_NUMBER

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a


class SparseGGNNChemModel(ChemModel):
    def __init__(self, args):
        super(SparseGGNNChemModel, self).__init__(args)
        self.params.update({
            'batch_size': 20,
            'use_edge_bias': False,
            'use_edge_msg_avg_aggregation': True,

            'layer_timesteps': [1, 1, 1, 1],  # number of layers & propagation steps per layer

            'graph_rnn_cell': 'GRU',  # GRU or RNN
            'graph_rnn_activation': 'tanh',  # tanh, ReLU
            'graph_state_dropout_keep_prob': 1.,
            'task_sample_ratios': {},
        })

    def compute_final_node_representations(self):
        pass
        # cur_node_states = self.placeholders['initial_node_representation']  # number of nodes in batch v x D
        # num_nodes = tf.shape(self.placeholders['initial_node_representation'], out_type=tf.int64)[0]

        # for (layer_idx, num_timesteps) in enumerate(self.params['layer_timesteps']):
        #     with tf.variable_scope('gnn_layer_%i' % layer_idx):
        #         adjacency_matrices = []  # type: List[tf.SparseTensor]
        #         for adjacency_list_for_edge_type in self.placeholders['adjacency_lists']:
        #             # adjacency_list_for_edge_type (shape [-1, 2]) includes all edges of type e_type of a sparse graph with v nodes (ids from 0 to v).
        #             adjacency_matrix_for_edge_type = tf.SparseTensor(indices=adjacency_list_for_edge_type,
        #                                                              values=tf.ones_like(
        #                                                                  adjacency_list_for_edge_type[:, 1],
        #                                                                  dtype=tf.float32),
        #                                                              dense_shape=[num_nodes, num_nodes])
        #             adjacency_matrices.append(adjacency_matrix_for_edge_type)

        #         for step in range(num_timesteps):
        #             with tf.variable_scope('timestep_%i' % step):
        #                 incoming_messages = []  # list of v x D

        #                 # Collect incoming messages per edge type
        #                 for adjacency_matrix in adjacency_matrices:
        #                     incoming_messages_per_type = tf.sparse_tensor_dense_matmul(adjacency_matrix,
        #                                                                                cur_node_states)  # v x D
        #                     incoming_messages.extend([incoming_messages_per_type])

        #                 # Pass incoming messages through linear layer:
        #                 incoming_messages = tf.concat(incoming_messages, axis=1)  # v x [2 *] edge_types
        #                 messages_passed = tf.matmul(incoming_messages,
        #                                             self.weights['edge_weights'][layer_idx])  # v x D

        #                 if self.params['use_edge_bias']:
        #                     messages_passed += tf.matmul(self.placeholders['num_incoming_edges_per_type'],
        #                                                  self.weights['edge_biases'][layer_idx])  # v x D

        #                 if self.params['use_edge_msg_avg_aggregation']:
        #                     num_incoming_edges = tf.reduce_sum(self.placeholders['num_incoming_edges_per_type'],
        #                                                        keep_dims=True, axis=-1)  # v x 1
        #                     messages_passed /= num_incoming_edges + SMALL_NUMBER

        #                 # pass updated vertex features into RNN cell
        #                 cur_node_states = self.weights['rnn_cells'][layer_idx](messages_passed, cur_node_states)[1]  # v x D

        # return cur_node_states

    def gated_regression(self, last_h, regression_gate, regression_transform):
        pass
        # # last_h: [v x h]
        # gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=-1)  # [v x 2h]
        # gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)  # [v x 1]

        # # Sum up all nodes per-graph
        # num_nodes = tf.shape(gate_input, out_type=tf.int64)[0]
        # graph_nodes = tf.SparseTensor(indices=self.placeholders['graph_nodes_list'],
        #                               values=tf.ones_like(self.placeholders['graph_nodes_list'][:, 0],
        #                                                   dtype=tf.float32),
        #                               dense_shape=[self.placeholders['num_graphs'], num_nodes])  # [g x v]
        # return tf.squeeze(tf.sparse_tensor_dense_matmul(graph_nodes, gated_outputs), axis=[-1])  # [g]

    # ----- Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data, num_edge_types, is_training_data):
        processed_graphs = []
        for d in raw_data:
            processed_graphs.append({"edge_list": d['graph'],
                                     "annotation": d["node_features"],
                                     "label": [d["targets"][task_id][0] for task_id in self.params['task_ids']]})

        # if is_training_data:
        #     np.random.shuffle(processed_graphs)
        #     for task_id in self.params['task_ids']:
        #         task_sample_ratio = self.params['task_sample_ratios'].get(str(task_id))
        #         if task_sample_ratio is not None:
        #             ex_to_sample = int(len(processed_graphs) * task_sample_ratio)
        #             for ex_id in range(ex_to_sample, len(processed_graphs)):
        #                 processed_graphs[ex_id]['labels'][task_id] = None

        return processed_graphs

    def __graph_to_adjacency_lists(self, graph, num_edge_types): 
        '''
        return Tuple of Dict[int, np.ndarray] and Dict[int, Dict[int, int]]]
        adj_list: {edge type: list of (src, dest)}
        num_incoming_edges_dicts_per_type: {edge type: {node_id: num_incoming_edges}}
        '''
        adj_lists = defaultdict(list)
        num_incoming_edges_dicts_per_type = defaultdict(lambda: defaultdict(lambda: 0))
        for src, e, dest in graph:
            fwd_edge_type = e - 1  # Make edges start from 0
            adj_lists[fwd_edge_type].append((src, dest))
            num_incoming_edges_dicts_per_type[fwd_edge_type][dest] += 1
            if self.params['tie_fwd_bkwd']:
                adj_lists[fwd_edge_type].append((dest, src))
                num_incoming_edges_dicts_per_type[fwd_edge_type][src] += 1

        final_adj_lists = {e: np.array(sorted(lm), dtype=np.int32)
                           for e, lm in adj_lists.items()}

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd']):
            for (edge_type, edges) in adj_lists.items():
                bwd_edge_type = num_edge_types + edge_type
                final_adj_lists[bwd_edge_type] = np.array(sorted((y, x) for (x, y) in edges), dtype=np.int32)
                for (x, y) in edges:
                    num_incoming_edges_dicts_per_type[bwd_edge_type][y] += 1

        return final_adj_lists, num_incoming_edges_dicts_per_type


    def make_minibatch_iterator(self, data, is_training):
        """Create minibatches by flattening adjacency matrices into a single adjacency matrix with
        multiple disconnected components.
        Batch size = total number of nodes from each sample
        RETURN: tuple of
        np.array(batch_node_features),
        np.concatenate(batch_num_incoming_edges_per_type, axis=0),
        np.array(batch_graph_nodes_list, dtype=np.int32),
        np.transpose(batch_target_task_values, axes=[1,0]),
        #np.transpose(batch_target_task_mask, axes=[1, 0]),
        #num_graphs_in_batch,
        #dropout_keep_prob,
        adj_list

        initial_node_representation
        num_incoming_edges_per_type
        graph_nodes_list
        target_values
        target_mask 
        num_graphs 
        graph_state_keep_prob

        """

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features_padded = []
            batch_node_features = []
            batch_target_task_values = []
            batch_adj_matrix = []
            while num_graphs < len(data) and num_graphs_in_batch < self.params['batch_size']:
                cur_graph = data[num_graphs]
                num_graphs_in_batch += 1
                num_nodes_in_graph = len(cur_graph['annotation'])
                padded_features = np.pad(cur_graph['annotation'],
                                         ((0, 0), (0, self.params['hidden_size'] - self.annotation_size)),
                                         'constant') #(len of node_features (i.e. annotation_size), hidden_size)
                node_feature = np.zeros((self.max_num_vertices, self.params['hidden_size']))
                for j in range(num_nodes_in_graph):
                    for src, _, dest in cur_graph['edge_list']:
                        node_feature[src, :] += padded_features[j, :]
                        node_feature[dest, :] += padded_features[j, :]
                batch_node_features_padded.append(node_feature)#padded_features)
                adj_matrix = create_adjacency_matrix(cur_graph['edge_list'], self.max_num_vertices, self.num_edge_types)
                batch_adj_matrix.append(adj_matrix)

                target_task_values = []
                for target_val in cur_graph['label']:
                    target_task_values.append(target_val)
                batch_target_task_values.append(target_task_values)

            num_graphs += num_graphs_in_batch
            batch_data = (  
                            torch.FloatTensor(np.array(batch_adj_matrix).astype(float)),
                            torch.FloatTensor(np.copy(np.array(batch_node_features_padded))[:, :, :self.annotation_size]),
                            torch.FloatTensor(np.array(batch_node_features_padded)),
                            torch.FloatTensor(np.array(batch_target_task_values)) #axes=[1,0]
                        )
            yield batch_data


def main():
    try:
        model = SparseGGNNChemModel('./')
        model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()


if __name__ == "__main__":
    main()
