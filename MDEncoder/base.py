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
                n_neurons = n_neurons//2
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
        print( MDmodel.alpha , MDmodel.layer_dims)
        K.clear_session()
        MDmodel.model = Autoencoder(layer_dims = MDmodel.layer_dims, verbose=0)
        MDmodel.model.pretrain(np.zeros((MDmodel.layer_dims[0],1)), epochs = 0, num_samples = 1)
        MDmodel.autoencoder, MDmodel.encoder, MDmodel.decoder = MDmodel.model.unroll(MDmodel.sparse)
        MDmodel.autoencoder.load_weights(f'{path}/autoencoder.h5')
        MDmodel.encoder.load_weights(f'{path}/encoder.h5')
        MDmodel.decoder.load_weights(f'{path}/decoder.h5')
        print("Model loaded!")
        print(MDmodel.autoencoder.summary())
        return MDmodel
            
    def save(self,path):
        pickle.dump([self.sparse , self.batch_size ,self.lr , self.alpha , self.layer_dims], open(f'{path}/parameters.pkl',"wb"))
        self.autoencoder.save_weights(f'{path}/autoencoder.h5')
        self.encoder.save_weights(f'{path}/encoder.h5')
        self.decoder.save_weights(f'{path}/decoder.h5')
        print("Model saved!")

    def fit(self,x_train,x_test,output_path=".",epochs=(100,200),verbose=2,xyz_dim=None,sample_weight=None,save_files=True,show_losses=True,loss_history=False,restart=False):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        #skip initial pretraining if restart from current model
        if not xyz_dim:
            xyz_dim = 0
        if not restart:
            K.clear_session()
            self.model = Autoencoder(layer_dims = self.layer_dims, verbose=verbose)
            if epochs[0]>0:
                self.model.pretrain(np.transpose(x_train), epochs = epochs[0], num_samples = x_train.shape[0], batch_size=self.batch_size)
                self.autoencoder, self.encoder, self.decoder = self.model.unroll(self.sparse)
            else:
                self.autoencoder, self.encoder, self.decoder = self.model.build_model(self.sparse)
        
        loss_dict = {'reconstruct':'mse','embedded': embedded_loss(f_dim=self.layer_dims[0],l_dim=self.layer_dims[-1],xyz_dim=xyz_dim) }
        metric_dict = {'reconstruct':RFVE_metric,'embedded': RV_metric(xyz_dim=xyz_dim)}
        #metric_dict = {'reconstruct':'mse', 'embedded': embedded_loss(f_dim=self.layer_dims[0],l_dim=self.layer_dims[-1])}
        if sample_weight is not None:
            weight_list = [sample_weight,sample_weight]
        else:
            weight_list = None

        self.autoencoder.compile(optimizer=Adam(lr=self.lr),
                            loss=loss_dict, loss_weights={'reconstruct': 1.0-self.alpha, 'embedded': self.alpha},
                            metrics=metric_dict)
        #self.autoencoder.compile(optimizer=Adam(lr=self.lr),
        #                    loss=loss_dict, loss_weights={'reconstruct': 1.0-self.alpha, 'embedded': self.alpha})
        if verbose > 0:
            print(self.autoencoder.summary())
        
        callbacks = []
        #callbacks.append( EarlyStopping(monitor='val_embedded_RV_metric',min_delta=1e-4,patience=10,verbose=verbose) )
        callbacks.append( EarlyStopping(monitor='val_loss',min_delta=2e-6,patience=10,verbose=verbose) )
        if save_files:
            callbacks.append( ModelCheckpoint(f'{output_path}/checkpt.h5',monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=0) )        
        history = self.autoencoder.fit(x_train, [x_train,x_train], epochs=epochs[1], batch_size=self.batch_size, shuffle=True, sample_weight=weight_list,
                                  validation_data=(x_test, [x_test,x_test]), verbose=verbose,callbacks=callbacks)
        if save_files:
            self.autoencoder.load_weights(f'{output_path}/checkpt.h5')        
            pickle.dump(history.history, open(f'{output_path}/losses.pkl',"wb"))
            self.save(output_path)
        
        if show_losses:
            plt.figure()
            plt.plot(history.history['loss'],linewidth=3,linestyle='--',color='b',label='Train loss')
            plt.plot(history.history['val_loss'],linewidth=3,linestyle='--',color='r',label='Val loss')
            plt.xlabel('Epochs',fontsize=18)
            plt.ylabel('Loss',fontsize=18)
            plt.legend(fontsize=15)
            plt.figure()
            plt.plot(history.history['val_embedded_RV_metric'],linewidth=3,color='r',label='Val RV')
            plt.plot(history.history['val_reconstruct_RFVE_metric'],linewidth=3,color='b',label='Val RFVE')
            plt.xlabel('Epochs',fontsize=18)
            plt.ylabel('Loss',fontsize=18)
            plt.legend(fontsize=15)

        if loss_history:
            return history
        
