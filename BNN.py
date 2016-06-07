__author__ = 'thushv89'

import theano
import theano.Tensor as T
import numpy as np


class BNN(object):

    def __init__(self,in_size,out_size,layer_sizes,learning_rate):
        self.in_size = in_size
        self.out_size = out_size
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.layers = []
        self.sym_x = T.dmatrix('x')
        self.sym_y = T.dmatrix('y')

        # hold the values of each layer for pretraining
        self.pretrain_layers = []

    def make_layers(self):

        # hidden layers
        for i,l_size in self.layer_sizes:
            if i==0:
                w = theano.shared(np.random.random_sample((self.in_size,l_size)),name='w'+str(self.in_size)+'->'+str(l_size))
            else:
                w = theano.shared(np.random.random_sample((self.layer_sizes[i-1],l_size)),name='w'+str(self.layer_sizes[i-1])+'->'+str(l_size))
        self.layers.append(w)

        # last layer logistic regression
        w = theano.shared(np.random.random_sample((self.layer_sizes[-1],self.out_size)),name='w'+str(self.layer_sizes[-1])+'->'+str(self.out_size))
        self.layers.append(w)

    def process(self):

        hid_out
        for layer in self.layers[:-1]:

    def pre_train_func(self):
if __name__=='__main__':
