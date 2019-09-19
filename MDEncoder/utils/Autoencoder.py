import numpy as np
import pandas as pd
import os.path
from .RBM import *
from .RBM_with_linear_hidden_units import *
from .RBM_with_linear_visible_units import *

from keras.layers import Input, Dense
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

            - layer_dims = A list of the layer sizes, visible first, latent last

            Note that the number of hidden layers in the unrolled autoencoder 
            will be twice the length of layer_dims. 
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

            Still needs to be unrolled.
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

    def pretrain(self,x,epochs,num_samples = 50000, batch_size = 100 , real_input=True):
        '''
            Greedy layer-wise training
            
            The last layer is a RBM with linear hidden units

            shape(x) = (v_dim, number_of_examples)
        '''
        RBM_layers = []

        for i in range(self.num_hidden_layers): # initialize RBM's
	
            if (i < self.num_hidden_layers - 1):
                if i==0 and real_input:
                    RBM_layers.append(RBM_with_linear_visible_units(self.layer_dims[i],self.layer_dims[i+1]))
                else:    
                    RBM_layers.append(RBM(self.layer_dims[i],self.layer_dims[i+1]))
            else:
                RBM_layers.append(RBM_with_linear_hidden_units(self.layer_dims[i],self.layer_dims[i+1]))
     

        for i in range(self.num_hidden_layers):  # train RBM's 
            if self.verbose>0:
                print("Training RBM layer %i"%(i+1))

            RBM_layers[i].train(x, epochs=epochs, batch_size=batch_size, verbose=self.verbose) # train the ith RBM
            
            if not(i == self.num_hidden_layers - 1): # generate samples to train next layer
                _,x = RBM_layers[i].gibbs_sampling(2,num_samples) 
                x = x.astype(float)

            self.W.append(RBM_layers[i].W) # save trained weights
            self.b.append(RBM_layers[i].b)
            self.a.append(RBM_layers[i].a)

        self.pretrained = True

        return

    
    def unroll(self,sparse):
        '''
            Unrolls the pretrained RBM network into a DFF keras model 
            and sets hidden layer parameters to pretrained values.

            Returns the keras model
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
                embedded = Dense(embed_dim,
                          activation='tanh',  
                          weights = weights, name='embedded')(x)
            else:
                #sigmoid for binary feature selection
                x = Dense(self.layer_dims[i+1], 
                          activation='relu',
                          activity_regularizer=regularizers.l1(sparse),
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
            else:
                #sigmoid for binary feature selection
                y = Dense(self.layer_dims[self.num_hidden_layers-i-1], 
                          activation='relu',
                          activity_regularizer=regularizers.l1(sparse),
                          weights = weights)(y)
        decoder = Model(embedded_input, y, name='reconstruct')
    
        outputs = decoder(encoder(inputs))
        autoencoder = Model(inputs=inputs,outputs=[outputs,embedded])
        return autoencoder, encoder, decoder
    
    def save(self,filename):
        '''
            saves the pretrained weights. Saving and loading a keras model 
            after pretraining is better done directly to the self.autoencoder
            object using the keras fnctions save() and load_model()
        '''

        if self.pretrained == True:
            for i in range(self.num_hidden_layers):
                weights = {"W":self.W[i],'a':self.a[i],'b':self.b[i]}
                RBM.save_weights(weights,filename+"_"+str(i))
        else: 
            print("No pretrained weights to save.")

        return
