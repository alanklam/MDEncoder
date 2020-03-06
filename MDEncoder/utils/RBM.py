# RBM class
'''
    Adapted from code by Ruslan Salakhutdinov and Geoff Hinton
    Available at: http://science.sciencemag.org/content/suppl/2006/08/04/313.5786.504.DC1
    
    A class defining a restricted Boltzmann machine.  

'''
import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit, prange

learning_rate = 0.1

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class RBM:

    def __init__(self,v_dim,h_dim):
        '''
            v_dim = dimension of the visible layer
            h_dim = dimension of the hidden layer
        '''
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.W = np.zeros((self.v_dim,self.h_dim))
        self.a = np.zeros((self.v_dim,1))
        self.b = np.zeros((self.h_dim,1))
        return

    @classmethod
    def from_Values(cls,weights):
        '''
            Initialize with trained weights.
        '''
        W,a,b = weights['W'],weights['a'],weights['b']
        assert (W.shape[0] == a.shape[0]) and (W.shape[1] == b.shape[0])
        rbm = cls(W.shape[0],W.shape[1])
        rbm.W = W
        rbm.a = a
        rbm.b = b
        return rbm

    @classmethod
    def from_File(cls,filename):
        '''
            Initialize with weights loaded from a file.
        '''
        return cls.from_Values(RBM.load_weights(filename))

    def v_probs(self,h):
        '''
            Input:
            - h has shape (h_dim,m)
            - a has shape (v_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert(h.shape[0] == self.h_dim)
        v_probs = sigmoid(self.a + np.dot(self.W,h))
        assert(not np.sum(np.isnan(v_probs)))
        return v_probs

    def h_probs(self,v):
        '''
            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert(v.shape[0] == self.v_dim)
        h_probs = sigmoid(self.b + np.dot(self.W.T,v))
        assert(not np.sum(np.isnan(h_probs)))
        return h_probs

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
        @jit(cache=True,parallel=True)
        def nb_binomial(n,p):
            d0 = np.shape(p)[0]
            d1 = np.shape(p)[1]
            tmp = np.empty_like(p)
            for c in prange(d0*d1):
                i = c // d1
                j = c % d1
                tmp[i,j] = np.random.binomial(n,p[i,j])
            return tmp

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
        def contrastive_grad(x,v_dim,h_dim,epochs,x_shape,batch_size,initialize_weights,verbose):
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
                    h_pos_probs  = 1/(1+np.exp(-(b + np.dot(W.T,v_pos_states))))
                    pos_prods    = np.expand_dims(v_pos_states,1)*np.expand_dims(h_pos_probs,0)
                    h_pos_states = nb_binomial(1,h_pos_probs)
                                
                    # get negative probs and product
                    v_neg_probs  = a + np.dot(W,h_pos_states)
                    h_neg_probs  = 1/(1+np.exp(-(b + np.dot(W.T,v_neg_probs))))
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
                if (abs(error_sum - error_log)/error_sum < 0.01) or (abs(error_sum - error_log) < 0.005):
                    break
                error_log = error_sum
                error_sum = 0.
            return W, a, b

        self.W, self.a, self.b = contrastive_grad(x,self.v_dim,self.h_dim,epochs,x.shape[1],batch_size,initialize_weights,verbose)

        return

    def gibbs_sampling(self, n=1, m=1,v=None):
        '''
            n - number of iterations of blocked Gibbs sampling
            m - number of samples generated
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
            h_states = np.random.binomial(1,h_probs)
        return v_states,h_states

    def plot_weights(self):
        '''
            For debugging 
        '''
        
        return

    def plot_weight_histogram(self):
        '''
            For debugging 
        '''
        plt.figure(1)

        plt.subplot(311)
        plt.title('Weights')
        plt.hist(self.W.flatten(),bins='auto')
        
        plt.subplot(312)
        plt.title('Visible biases')
        plt.hist(self.a.flatten(),bins='auto')
        
        plt.subplot(313)
        plt.title('Hidden biases')
        plt.hist(self.b.flatten(),bins='auto')

        plt.tight_layout()

        plt.show()
        return

    def save(self, filename):
        '''
            Save trained weights of self to file
        '''
        weights = {"W":self.W,"a":self.a,"b":self.b}
        RBM.save_weights(weights,filename)
        return

    @staticmethod
    def save_weights(weights,filename):
        '''
            Save RBM weights to file
        '''
        np.savetxt(filename + '_a.csv',weights['a'],delimiter=",")
        np.savetxt(filename + '_b.csv',weights['b'],delimiter=",")
        np.savetxt(filename + '_W.csv',weights['W'],delimiter=",")
        return

    @staticmethod
    def load_weights(filename):
        '''
            Save RBM weights to file
        '''
        W = np.loadtxt(filename + '_W.csv',delimiter=",")
        a = np.loadtxt(filename + '_a.csv',delimiter=",").reshape((W.shape[0],1))
        b = np.loadtxt(filename + '_b.csv',delimiter=",").reshape((W.shape[1],1))
        return {"W":W,"a":a,"b":b}


