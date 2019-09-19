from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
from .utils.Autoencoder import *
from .utils.losses import *
import os

class MDEncoder:
    
    kind = 'MDEncoder'
     
    def __init__(self,layer_dims=[],f_dim=None,latent_dim=4,n_layers=1,n_neurons=16,sparse=1e-6,batch_size=64,lr=0.001,alpha=0.5):
        self.sparse = sparse
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        if len(layer_dims)==0:
            self.layer_dims = [f_dim]
            for i in range(n_layers):
                self.layer_dims.append(n_neurons)
                n_neurons = n_neurons//4
            self.layer_dims.append(latent_dim) 
        else:
            self.layer_dims = layer_dims
            
        self.model = None
        self.autoencoder =None 
        self.encoder = None 
        self.decoder = None
    
    @classmethod
    def load(cls,path):
        MDmodel = cls()
        MDmodel.sparse , MDmodel.batch_size , MDmodel.lr , MDmodel.alpha , MDmodel.layer_dims = pickle.load(open(f'{path}/parameters.pkl',"rb"))
        K.clear_session()
        MDmodel.model = Autoencoder(layer_dims = MDmodel.layer_dims)
        MDmodel.model.pretrain(np.zeros((MDmodel.layer_dims[0],1)), epochs = 0, num_samples = 1, verbose=0)
        MDmodel.autoencoder, MDmodel.encoder, MDmodel.decoder = MDmodel.model.unroll(MDmodel.sparse)
        MDmodel.autoencoder.load_weights(f'{path}/autoencoder.h5')
        MDmodel.encoder.load_weights(f'{path}/encoder.h5')
        MDmodel.decoder.load_weights(f'{path}/decoder.h5')
        print("Model loaded!")
        return MDmodel
            
    def save(self,path):
        pickle.dump([self.sparse , self.batch_size ,self.lr , self.alpha , self.layer_dims], open(f'{path}/parameters.pkl',"wb"))
        self.autoencoder.save_weights(f'{path}/autoencoder.h5')
        self.encoder.save_weights(f'{path}/encoder.h5')
        self.decoder.save_weights(f'{path}/decoder.h5')
        print("Model saved!")
        
    def create_model(f_dim=None,latent_dim=4,n_layers=1,n_neurons=16,sparse=1e-6,batch_size=64,lr=0.001,alpha=0.5):
        layer_dims = [f_dim]
        for i in range(n_layers):
            layer_dims.append(n_neurons)
            n_neurons = n_neurons//4
        layer_dims.append(latent_dim)    
        model = Autoencoder(layer_dims = layer_dims)
        model.pretrain(np.transpose(x_train), epochs = 100, num_samples = x_train.shape[0], progress=False)
        autoencoder , encoder , decoder = model.unroll(sparse)
        loss_dict = {'reconstruct':'mse','embedded': embedded_loss(f_dim=f_dim,l_dim=latent_dim) }
        loss_weights_dict = {'reconstruct': 1.0-alpha, 'embedded': alpha}
        metric_dict = {'embedded': RV_metric}
        autoencoder.compile(optimizer=Adam(lr=lr),loss=loss_dict, 
                            loss_weights=loss_weights_dict, metrics=metric_dict)
        #output compiled model and batch_size parameter for later use in gridsearch
        return autoencoder, batch_size

    def fit(self,x_train,x_test,output_path=".",epochs=(100,200),verbose=2,save_files=True,show_losses=True,loss_history=False):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        K.clear_session()
        self.model = Autoencoder(layer_dims = self.layer_dims, verbose=verbose)
        self.model.pretrain(np.transpose(x_train), epochs = epochs[0], num_samples = x_train.shape[0], batch_size=self.batch_size)
        self.autoencoder, self.encoder, self.decoder = self.model.unroll(self.sparse)
        
        loss_dict = {'reconstruct':'mse','embedded': embedded_loss(f_dim=self.layer_dims[0],l_dim=self.layer_dims[-1]) }
        metric_dict = {'embedded': RV_metric}
        self.autoencoder.compile(optimizer=Adam(lr=self.lr),
                            loss=loss_dict, loss_weights={'reconstruct': 1.0-self.alpha, 'embedded': self.alpha},
                            metrics=metric_dict)
        if verbose > 0:
            print(self.autoencoder.summary())
        
        callbacks = []
        callbacks.append( EarlyStopping(monitor='val_embedded_RV_metric',min_delta=1e-4,patience=10,verbose=verbose) )
        if save_files:
            callbacks.append( ModelCheckpoint(f'{output_path}/checkpt.h5',monitor='val_embedded_RV_metric',save_best_only=True,save_weights_only=True,verbose=0) )
        history = self.autoencoder.fit(x_train, [x_train,x_train], epochs=epochs[1], batch_size=self.batch_size, shuffle=True, 
                                  validation_data=(x_test, [x_test,x_test]), verbose=verbose,callbacks=callbacks)
        if save_files:
            self.autoencoder.load_weights(f'{output_path}/checkpt.h5')        
            pickle.dump(history.history, open(f'{output_path}/losses.pkl',"wb"))
            self.save(output_path)
        
        if show_losses:
            plt.figure()
            plt.plot(history.history['loss'],linewidth=3,linestyle='--',color='b',label='Train loss')
            plt.plot(history.history['val_loss'],linewidth=3,linestyle='--',color='r',label='Val loss')
            plt.plot(history.history['embedded_RV_metric'],linewidth=3,color='b',label='Train RV')
            plt.plot(history.history['val_embedded_RV_metric'],linewidth=3,color='r',label='Val RV')
            plt.xlabel('Epochs',fontsize=18)
            plt.ylabel('Loss',fontsize=18)
            plt.legend(fontsize=15)

        if loss_history:
            return history
        
