from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
import pickle
from hyperopt import hp, tpe, fmin , space_eval, Trials
from ..base import *

class hyperoptsearchCV():

    def __init__(self, space, n_splits=3,random=0,verbose=0,epochs=(20,150),max_evals=100):
        self.space = space
        self.best = None
        self.best_param = None
        self.search_param = dict( n_splits=n_splits,random=random,verbose=verbose,epochs=epochs , max_evals=max_evals)
        self.trials = Trials()

    @classmethod
    def load(cls,file):
        search_model = cls(space=None)
        space , dict_obj = pickle.load(open(file,"rb"))
        search_model.space = space
        search_model.__dict__.update(dict_obj)
        print("Hyperopt search object loaded from %s!" % file)
        return search_model
        
    def save(self,file):
        pickle.dump((self.space,self.__dict__), open(file,"wb"))
        print("Hyperopt search object saved to %s!" % file)

    def fit(self, data , xyz_dim=0, sample_weight=None):

        def CV_function(param_set):
            model = MDEncoder(**param_set)
            kf = KFold(n_splits=self.search_param['n_splits'],random_state=self.search_param['random'])
            score = []
            for train_index, test_index in kf.split(data):
                if sample_weight is not None:
                    weight = sample_weight[train_index]
                else:
                    weight = None
                history = model.fit(data[train_index], data[test_index], save_files=False, xyz_dim=xyz_dim, sample_weight=weight, epochs= self.search_param['epochs'], verbose=self.search_param['verbose'], show_losses=False,loss_history=True)
                total_score = np.array(history.history['val_embedded_RV_metric'])+np.array(history.history['val_reconstruct_RFVE_metric'])
                #total_score = np.array(history.history['val_embedded_RV_metric'])
                #total_score = np.array(history.history['val_reconstruct_RFVE_metric'])
                total_score[np.isnan(total_score)] = 2.0
                score.append( np.min(total_score) )
                #score.append( min(history.history['val_embedded_embedded_loss']) + min(history.history['val_reconstruct_mean_squared_error']) )
            return np.mean(score)

        self.best = fmin(CV_function,self.space,algo=tpe.suggest,max_evals=self.search_param['max_evals'],trials=self.trials)
        self.best_param = space_eval(self.space, self.best)
        print("Best results: ",self.best_param)

class gridsearchCV():
    
    def __init__(self, f_dim = None, n_layers=[1], n_neurons=[16], latent_dim=[2], sparse=[0], alpha=[0.5], lr=[1e-3], batch_size=[64],n_splits=3,random=0,verbose=0,epochs=(50,50)):
        assert isinstance(f_dim, int) , "Provide integer value for feature dimension!"
        self.param_grid = list(ParameterGrid(dict( f_dim=[f_dim], n_layers=n_layers, n_neurons=n_neurons, latent_dim=latent_dim, sparse=sparse, alpha=alpha, lr=lr, batch_size=batch_size )))
        self.gridsearch_param = dict( n_splits=n_splits,random=random,verbose=verbose,epochs=epochs )
        self.cv_hist = []
        self.best_score = None
        self.best_param = None
        self.param_rank = None
        
    @classmethod
    def load(cls,file):
        grid_model = cls(f_dim=1,verbose=0)
        grid_model.__dict__.update(pickle.load(open(file,"rb")))
        print("Grid search object loaded from %s!" % file)
        return grid_model
        
    def save(self,file):
        pickle.dump(self.__dict__, open(file,"wb"))
        print("Grid search object saved to %s!" % file)
        
    def fit(self,data):
        self.best_score = 1.0
        for param_set in self.param_grid:
            kf = KFold(n_splits=self.gridsearch_param['n_splits'],random_state=self.gridsearch_param['random'])
            score = []
            for train_index, test_index in kf.split(data):
                model = MDEncoder(**param_set)
                history = model.fit(data[train_index],data[test_index],save_files=False,epochs=self.gridsearch_param['epochs'], verbose=self.gridsearch_param['verbose'],show_losses=False,loss_history=True)
                score.append(min(history.history['val_embedded_RV_metric']))
            print("Finished with parameter set: {0}\nMean val_embedded_RV_metric = {1}".format(param_set, np.mean(score)))
            self.cv_hist.append((param_set,score))
            if np.mean(score) < self.best_score:
                self.best_score = np.mean(score)
                self.best_param = param_set 

        print("Best score: %f using %s" % (self.best_score, self.best_param))
        self.param_rank = sorted(self.cv_hist, key=lambda x:np.mean(x[1]), reverse=False)
        self.param_rank = list(map(lambda x:(x[0],np.mean(x[1])), self.param_rank))

