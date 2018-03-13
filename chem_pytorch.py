#!/usr/bin/env/python

import torch
import time
import os
import json
import numpy as np
import pickle
import random
from utils_pytorch import MLP, ThreadedIterator, AverageMeter, SMALL_NUMBER
#from callbacks import EarlyStopping
from torch.autograd import Variable
from model_pytorch import repackage_hidden, GGNN

def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
    bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    for src, e, dest in graph:
        amat[e-1, dest, src] = 1
        amat[e-1 + bwd_edge_offset, src, dest] = 1
    return amat


class ChemModel(object):

    def __init__(self, data_dir, config_file=None):
        self.data_dir = data_dir

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        # Collect parameters:
        self.params = {
            'num_epochs': 3000,
            'patience': 25,
            'learning_rate': 0.0001,
            'clamp_gradient_norm': 1.0,
            'out_layer_dropout_keep_prob': 1.0,

            'hidden_size': 100,
            'num_timesteps': 4,
            'use_graph': True,

            'tie_fwd_bkwd': True,
            'task_ids': [0],

            'random_seed': 0,
            'use_cuda': True
        }
        if config_file is not None:
            with open(config_file, 'r') as f:
                self.params.update(json.load(f))
        with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
            json.dump(self.params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        #self.train_data = self.load_data("molecules_train.json", is_training_data=True)
        #self.valid_data = self.load_data("molecules_valid.json", is_training_data=False)

        # Build the actual model
        torch.manual_seed(self.params['random_seed'])
        if self.params['use_cuda']:
            torch.cuda.manual_seed_all(self.params['random_seed'])


    def load_data(self, file_name, is_training_data):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        #restrict = self.args.get("--restrict_data")
        #if restrict is not None and restrict > 0:
        #    data = data[:restrict]

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v+1 for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2)) ###directed graph
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0])) ###5
        return self.process_raw_graphs(data, self.num_edge_types, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string): #return List[List[int]]
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data, is_training_data):
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        return GGNN(self.params['hidden_size'], self.annotation_size, self.num_edge_types, self.max_num_vertices, self.params['num_timesteps'])
        # self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
        #                                                     name='target_values')
        # self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
        #                                                   name='target_mask')
        # self.placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
        # self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

        # with tf.variable_scope("graph_model"):
        #     self.prepare_specific_graph_model()
        #     # This does the actual graph work:
        #     if self.params['use_graph']:
        #         self.ops['final_node_representations'] = self.compute_final_node_representations()
        #     else:
        #         self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['initial_node_representation'])

        # self.ops['losses'] = []
        # for (internal_id, task_id) in enumerate(self.params['task_ids']):
        #     with tf.variable_scope("out_layer_task%i" % task_id):
        #         with tf.variable_scope("regression_gate"):
        #             self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
        #                                                                    self.placeholders['out_layer_dropout_keep_prob'])
        #         with tf.variable_scope("regression"):
        #             self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
        #                                                                         self.placeholders['out_layer_dropout_keep_prob'])
        #         computed_values = self.gated_regression(self.ops['final_node_representations'],
        #                                                 self.weights['regression_gate_task%i' % task_id],
        #                                                 self.weights['regression_transform_task%i' % task_id])
        #         diff = computed_values - self.placeholders['target_values'][internal_id,:]
        #         task_target_mask = self.placeholders['target_mask'][internal_id,:]
        #         task_target_num = tf.reduce_sum(task_target_mask) + SMALL_NUMBER
        #         diff = diff * task_target_mask  # Mask out unused values
        #         self.ops['accuracy_task%i' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
        #         task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
        #         # Normalise loss to account for fewer task-specific examples in batch:
        #         task_loss = task_loss * (1.0 / (self.params['task_sample_ratios'].get(task_id) or 1.0))
        #         self.ops['losses'].append(task_loss)
        # self.ops['loss'] = tf.reduce_sum(self.ops['losses'])


    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self):
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self):
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data, is_training):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, loader, criterion, optimizer, is_training):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0
        #accuracies = []
        n_processed_data= 0
        batch_iterator = ThreadedIterator(loader, max_queue_size=2*self.params['batch_size'])
        for step, batch_data in enumerate(batch_iterator):
            print("step {}".format(step))
            adj_matrix, annotation, padded_annotation, target = batch_data
            #padding = np.zeros((annotation.shape[0], self.max_num_vertices, self.params['hidden_size'] - self.annotation_size))#.double()
            #init_input = np.concatenate((annotation, padding), axis=2) # batch_size x n_nodes x hidden_size
            batch_size = target.size()[0]
            n_processed_data += batch_size
            init_input = padded_annotation
            init_input = Variable(init_input, requires_grad=False)
            adj_matrix = Variable(adj_matrix, requires_grad=False)
            annotation = Variable(annotation, requires_grad=False)
            target = Variable(target, requires_grad=False)

            if self.params['use_cuda']:
                init_input = init_input.cuda()
                adj_matrix = adj_matrix.cuda()
                annotation = annotation.cuda()
                target = target.cuda()
                criterion = criterion.cuda()

            optimizer.zero_grad()
            self.model.zero_grad()
            output = self.model(init_input, annotation, adj_matrix)
            loss = criterion(output, target) #* num_graphs
            total_loss += loss.cpu().data.numpy()
            #acc = (output.cpu().data.numpy() == target.cpu().data.numpy()).sum() ################ CHECK
            #accuracies.append(acc) 

            # for m in self.model.parameters():
            #     print (m)
            # print(self.model)
            if is_training:
                loss.backward(retain_variables=True)
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.params['clamp_gradient_norm'])
                optimizer.step()

        #accuracies = np.sum(accuracies, axis=0)/ n_processed_data
        #loss = loss / processed_graphs
        return np.asscalar(total_loss[0])#, accuracies


    def train(self):
        val_data = self.load_data('molecules_valid.json', False)
        train_data = self.load_data('molecules_train.json', True)
        self.model = self.make_model()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        
        train_loader = self.make_minibatch_iterator(train_data, True)
        val_loader = self.make_minibatch_iterator(val_data, False)
        if self.params['use_cuda']:
            self.model.cuda()
        #es = EarlyStopping('loss', min_delta=0, patience=self.params['patience'])
        log_to_save = []
        best_val_loss, best_val_loss_epoch = float("inf"), 0
        for epoch in range(self.params['num_epochs']):
            print("epoch {}".format(epoch))         
            train_loss = self.run_epoch(train_loader, criterion, optimizer, True)
            print("Epoch {} Train loss {}".format(epoch, train_loss))
            val_loss = self.run_epoch(val_loader, criterion, optimizer, False)
            print("Epoch {} Val loss {}".format(epoch, val_loss))

            log_entry = {
                        'epoch': epoch,
                        'train_results': (train_loss),
                        'valid_results': (val_loss),
                        }
            log_to_save.append(log_entry)
            with open(self.log_file, 'w') as f:
                for i in range(len(log_to_save)):
                    json.dump(log_to_save[i], f, indent=4)

            if val_loss < best_val_loss:
                self.save_model(self.best_model_file)
                print(" (Best epoch so far, cum. val. loss decreased to %.5f from %.5f. Saving to '%s')" % (val_loss, best_val_loss, self.best_model_file))
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
            elif epoch - best_val_loss_epoch >= self.params['patience']:
                print("Stopping training after %i epochs without improvement on validation loss." % self.params['patience'])
                break
 

    def save_model(self, path):
        data_to_save = {
                         "params": self.params,
                         "model_weights": self.model.state_dict()
                       }
        torch.save(data_to_save, path)

    def restore_model(self, path):
        pass
