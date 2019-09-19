# RBM class
'''
    Adapted from code by Ruslan Salakhutdinov and Geoff Hinton
    Available at: http://science.sciencemag.org/content/suppl/2006/08/04/313.5786.504.DC1

    A class defining a restricted Boltzmann machine
    whose hidden units are "real-valued feature detectors 
    drawn from a unit variance Gaussian whose mean is determined by the input from 
    the logistic visible units" (Hinton, 2006)
    
    The only difference from RBM_with_probs is how h_probs are generated and h_states are 
    sampled.

'''
import numpy as np
import random
import matplotlib.pyplot as plt
from .RBM import *
from numba import jit, prange
learning_rate = 0.001

class RBM_with_linear_hidden_units(RBM):

    def h_probs(self,v):
        '''
            h_probs is defined differently than in the RBM
            with binary hidden units.
            
            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert(v.shape[0] == self.v_dim)
        return self.b + np.dot(self.W.T,v)

    def train(self, x, epochs = 10, batch_size = 100, verbose = 1, learning_rate = learning_rate, initialize_weights = True):
        ''' 
            Trains the RBM with the 1-step Contrastive Divergence algorithm (Hinton, 2002).
            
            Input:
            - x has shape (v_dim, number_of_examples)
            - plot = True plots debugging related plots after every epoch
            - initialize_weights = False to continue training a model 
              (e.g. loaded from earlier trained weights)

        '''
        assert(x.shape[0]==self.v_dim)
        # initialize weights and parameters
        if initialize_weights == True: 
            self.W =  np.ascontiguousarray(np.random.normal(0.,0.1,size = (self.v_dim,self.h_dim)))
            # visible bias a_i is initialized to ln(p_i/(1-p_i)), p_i = (proportion of examples where x_i = 1)
            #self.a = (np.log(np.mean(x,axis = 1,keepdims=True)+1e-10) - np.log(1-np.mean(x,axis = 1,keepdims=True)+1e-10))
            self.a = np.zeros((self.v_dim,1))
            self.b = np.zeros((self.h_dim,1))
                
        @jit(cache=True,parallel=True)
        def nb_mean2(x):
            d0 = np.shape(x)[0]
            d1 = np.shape(x)[1]
            tmp = np.empty((d0,d1))
            for c in prange(d0*d1):
                i = c // d1
                j = c % d1
                tmp[i,j] = np.mean(x[i,j,:])
            return tmp

        @jit(cache=True,parallel=True)
        def nb_mean1(x):
            #keeping the np array dimension
            d0 = np.shape(x)[0]
            tmp = np.empty((d0,1))
            for i in prange(d0):
                tmp[i,0] = np.mean(x[i,:])
            return tmp
        
        @jit()
        def contrastive_grad(x,W,a,b,v_dim,h_dim,epochs,x_shape,batch_size,verbose):
            np.random.seed(0)
        
            # track mse 
            error = 0.
            error_sum = 0.

            # hyperparameters used by Hinton for MNIST
            initialmomentum  = 0.5
            finalmomentum    = 0.9
            weightcost       = 0.0002
            num_minibatches  = int(x_shape/batch_size)

            DW = np.zeros((v_dim,h_dim))
            Da = np.zeros((v_dim,1))
            Db = np.zeros((h_dim,1))

            # initialize weights and parameters
            if initialize_weights == True: 
                W =  np.ascontiguousarray(np.random.normal(0.,0.1,size = (v_dim,h_dim)))
            # visible bias a_i is initialized to ln(p_i/(1-p_i)), p_i = (proportion of examples where x_i = 1)
            #self.a = (np.log(np.mean(x,axis = 1,keepdims=True)+1e-10) - np.log(1-np.mean(x,axis = 1,keepdims=True)+1e-10))
                a = np.zeros((v_dim,1))
                b = np.zeros((h_dim,1))
                
            error_log = 0.
            for i in range(epochs):
                if verbose>0:
                    print("Epoch ",(i+1))
                np.random.shuffle(x.T)

                if i>5:
                    momentum = finalmomentum
                else: 
                    momentum = initialmomentum
            
                for j in range(num_minibatches):
                
                    # get the next batch
                    v_pos_states =  np.ascontiguousarray(x[:,j*batch_size:(j+1)*batch_size])

                    # get hidden probs, positive product, and sample hidden states
                    h_pos_probs  = b + np.dot(W.T,v_pos_states)
                    pos_prods    = np.expand_dims(v_pos_states,1)*np.expand_dims(h_pos_probs,0)
                    h_pos_states = h_pos_probs + np.random.normal(0.,1.,size = h_pos_probs.shape)
                    
                    # get negative probs and product
                    v_neg_probs  = 1/(1+np.exp(-(a + np.dot(W,h_pos_states))))
                    h_neg_probs  = b + np.dot(W.T,v_neg_probs)
                    neg_prods    = np.expand_dims(v_neg_probs,1)*np.expand_dims(h_neg_probs,0)
                    
                    # compute the gradients, averaged over minibatch, with momentum and regularization
                    cd = nb_mean2(pos_prods - neg_prods)
                    DW = momentum*DW + learning_rate*(cd - weightcost*W)
                    Da = momentum*Da + learning_rate*nb_mean1(v_pos_states - v_neg_probs)                    
                    Db = momentum*Db + learning_rate*nb_mean1(h_pos_probs - h_neg_probs)
                
                    # update weights and biases
                    W = W + DW
                    a = a + Da
                    b = b + Db
                
                    # log the mse of the reconstructed images
                    error = np.mean((v_pos_states - v_neg_probs)**2)
                    error_sum = error_sum + error

                error_sum = error_sum/num_minibatches
                if verbose>0:
                    print("Reconstruction MSE = ",error_sum)
                if abs(error_sum - error_log)/error_sum < 0.005:
                    break
                error_log = error_sum
                error_sum = 0.
            return W, a, b

        self.W, self.a, self.b = contrastive_grad(x,self.W, self.a, self.b,self.v_dim,self.h_dim,epochs,x.shape[1],batch_size,verbose)

        return

    def gibbs_sampling(self, n=1, m=1,v=None):
        '''
            n - number of iterations of blocked Gibbs sampling
        '''
        if v is None:
            v_probs = np.full((self.v_dim,m),0.5)
            v = np.random.binomial(1,v_probs)

        h_probs  = self.h_probs(v)
        h_states = np.random.binomial(1,h_probs)
        for i in range(n):
            v_probs  = self.v_probs(h_states)
            v_states = np.random.binomial(1,v_probs)
            h_probs  = self.h_probs(v_states)
            h_states = h_probs + np.random.normal(0.,1.,size = h_probs.shape) # this line changes
        return v_states, h_states




