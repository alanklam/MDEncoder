import numpy as np
import pandas as pd
import os.path
from .RBM import *
from .RBM_with_linear_hidden_units import *
from .RBM_with_linear_visible_units import *
from .RBM_with_linear_units import *

from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras import regularizers
from numba import njit

learning_rate = 0.01

class Autoencoder:

    kind = 'Autoencoder'

    def __init__(self,layer_dims, verbose=1):
        '''
            Inputs:

            - layer_dims = A list of the layer sizes, input layer first, latent last
            layer_dims includes only layers in a encoder network
            - verbose = 1 prints model information
            
        '''
        self.verbose = verbose
        self.latent_dim = layer_dims[-1]
        self.v_dim      = layer_dims[0]
        self.num_hidden_layers = len(layer_dims)-1
        self.layer_dims = layer_dims
        if self.verbose>0:
            print("Layer dimensions:")
            for i in range(self.num_hidden_layers+1):
                print("Layer %i: %i"%(i,self.layer_dims[i]))
        self.W = []
        self.b = []
        self.a = []
        self.pretrained = False
        
        return

    @classmethod
    def pretrained_from_file(cls,filename):
        '''
            Initialize with pretrained weights from a file.

        '''
        i = 0
        weights = []
        layer_dims = []

        while os.path.isfile(filename+"_"+str(i)+"_a.csv"): # load the next layer's weights
            weights.append(RBM.load_weights(filename+"_"+str(i))) # load the next dict of weights
            layer_dims.append(np.shape(weights[i]['W'])[0])
            i = i+1
        layer_dims.append(np.shape(weights[i-1]['W'])[1])

        rbm = cls(layer_dims)

        for i in range(rbm.num_hidden_layers): 
            rbm.W.append(weights[i]['W'])
            rbm.a.append(weights[i]['a'])
            rbm.b.append(weights[i]['b'])
        
        rbm.pretrained = True

        return rbm

    def pretrain(self,x,epochs,num_samples = 50000, batch_size = 100):
        '''
            Greedy layer-wise training
            
            By default, assume the first and last layer are with real values units

            shape(x) = (v_dim, number_of_examples)
        '''
        RBM_layers = []
        if self.num_hidden_layers == 1: # initialize RBM's
            RBM_layers.append(RBM_with_linear_units(self.layer_dims[0],self.layer_dims[1]))
        else:
            for i in range(self.num_hidden_layers):
                #print("Using linear units RBM")
                RBM_layers.append(RBM_with_linear_units(self.layer_dims[i],self.layer_dims[i+1])) 
                #if (i < self.num_hidden_layers - 1):
                #    if i==0:
                #        RBM_layers.append(RBM_with_linear_visible_units(self.layer_dims[i],self.layer_dims[i+1]))
                #    else:    
                #        RBM_layers.append(RBM(self.layer_dims[i],self.layer_dims[i+1]))
                #else:
                #    RBM_layers.append(RBM_with_linear_hidden_units(self.layer_dims[i],self.layer_dims[i+1]))

        for i in range(self.num_hidden_layers):  # train RBM's 
            if self.verbose>0:
                print("Training RBM layer %i"%(i+1))

            RBM_layers[i].train(x, epochs=epochs, batch_size=batch_size, verbose=self.verbose) # train the ith RBM
            
            if not(i == self.num_hidden_layers - 1): # generate samples to train next layer
                _,x = RBM_layers[i].gibbs_sampling(n=2,m=num_samples,v=x) 
                x = x.astype(float)

            self.W.append(RBM_layers[i].W) # save trained weights
            self.b.append(RBM_layers[i].b)
            self.a.append(RBM_layers[i].a)

        self.pretrained = True

        return

    
    def unroll(self,sparse):
        '''
            Unrolls the pretrained RBM network into a keras model 
            and sets weight matrices to pretrained values.

            Returns the keras models (full autoencoder, encoder, decoder networks)
        '''
        if self.pretrained == False:
            print("Model not pretrained.")
            return

        # define keras model structure
        inputs = Input(shape=(self.v_dim,))
        x = inputs

        # build encoder
        for i in range(self.num_hidden_layers):
            weights = [self.W[i],self.b[i].flatten()]
            #last layer tanh
            if (i == self.num_hidden_layers - 1):
                embed_dim = self.layer_dims[i+1]
                #if self.num_hidden_layers>2:
                if False:
                    x = Dense(embed_dim,
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights)(x)
                    x = BatchNormalization()(x)
                    embedded = Activation('tanh',name='embedded')(x)
                else:
                    #print("linear")
                    embedded = Dense(embed_dim,
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights, name='embedded')(x)
            elif (i <= self.num_hidden_layers - 2):
                #layers before second last hidden layer use Batch Normalization
                if self.num_hidden_layers>2 and False:
                    #print("Adding BN")
                    x = Dense(self.layer_dims[i+1], 
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights)(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)
                else:
                    #sigmoid for binary feature selection
                    #print("No BN")
                    x = Dense(self.layer_dims[i+1], 
                              activation='relu',
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights)(x)
        encoder = Model(inputs,embedded)
        
        embedded_input = Input(shape=(embed_dim,))
        y = embedded_input
        # build decoder
        for i in range(self.num_hidden_layers):
            weights = [self.W[self.num_hidden_layers-i-1].T,self.a[self.num_hidden_layers-i-1].flatten()]
            #x = Dense(self.layer_dims[self.num_hidden_layers-i-1],
            #          activation='sigmoid', 
            #          weights = weights)(x)
            #last layer no activation
            if (i == self.num_hidden_layers - 1):
                y = Dense(self.layer_dims[self.num_hidden_layers-i-1],
                          activation=None,
                          weights = weights)(y)
            elif (i <= self.num_hidden_layers - 2):
                #layers before second last hidden layer use Batch Normalization
                if self.num_hidden_layers>2 and False:
                    #print("Adding BN")
                    y = Dense(self.layer_dims[self.num_hidden_layers-i-1], 
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights)(y)
                    y = BatchNormalization()(y)
                    y = Activation('relu')(y)
                else:
                    #print("No BN")
                    #sigmoid for binary feature selection
                    y = Dense(self.layer_dims[self.num_hidden_layers-i-1], 
                              activation='relu',
                              kernel_regularizer=regularizers.l2(sparse),
                              weights = weights)(y)
                
        decoder = Model(embedded_input, y, name='reconstruct')
    
        outputs = decoder(encoder(inputs))
        autoencoder = Model(inputs=inputs,outputs=[outputs,embedded])
        return autoencoder, encoder, decoder

    def build_model(self,sparse):
        '''
            Returns keras models for autoencoder network, without pretraining
            Initialize weight values by glorot_uniform
        '''
        if self.verbose>0:
        	print("No RBM pretraining, initialize connection weight randomly.")

        # define keras model structure
        inputs = Input(shape=(self.v_dim,))
        x = inputs

        # build encoder
        for i in range(self.num_hidden_layers):
            #last layer: linear activation
            if (i == self.num_hidden_layers - 1):
                embed_dim = self.layer_dims[i+1]
                embedded = Dense(embed_dim, activation=None,
                	kernel_regularizer=regularizers.l2(sparse), name='embedded')(x)
            elif (i <= self.num_hidden_layers - 2):
                #layers before second last hidden layer use Batch Normalization
                if self.num_hidden_layers>2:
                    x = Dense(self.layer_dims[i+1], 
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse))(x)
                    x = BatchNormalization()(x)
                    x = Activation('relu')(x)
                else:
                    x = Dense(self.layer_dims[i+1], 
                              activation='relu',
                              kernel_regularizer=regularizers.l2(sparse))(x)
        encoder = Model(inputs,embedded)
        
        embedded_input = Input(shape=(embed_dim,))
        y = embedded_input
        # build decoder
        for i in range(self.num_hidden_layers):
            #last layer no activation
            if (i == self.num_hidden_layers - 1):
                y = Dense(self.layer_dims[self.num_hidden_layers-i-1],
                          activation=None)(y)
            elif (i <= self.num_hidden_layers - 2):
                #layers before second last hidden layer use Batch Normalization
                if self.num_hidden_layers>2:
                    y = Dense(self.layer_dims[self.num_hidden_layers-i-1], 
                              activation=None,
                              kernel_regularizer=regularizers.l2(sparse))(y)
                    y = BatchNormalization()(y)
                    y = Activation('relu')(y)
                else:
                    y = Dense(self.layer_dims[self.num_hidden_layers-i-1], 
                              activation='relu',
                              kernel_regularizer=regularizers.l2(sparse))(y)
                
        decoder = Model(embedded_input, y, name='reconstruct')
    
        outputs = decoder(encoder(inputs))
        autoencoder = Model(inputs=inputs,outputs=[outputs,embedded])
        return autoencoder, encoder, decoder

