import tensorflow as tf
from keras import backend as K

def embedded_loss(f_dim,l_dim,xyz_dim): 
    def embedded_loss(y_true, y_pred):
        # pairwise squared distance loss, for multidimensional scaling
        na = tf.reduce_sum(tf.square(y_true[:,xyz_dim:]), 1)
        nb = tf.reduce_sum(tf.square(y_true[:,xyz_dim:]), 1)
        
        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        #D1 = (tf.sqrt(tf.maximum(na - 2*tf.matmul(y_true, y_true, False, True) + nb, 0.0)))/np.sqrt(input_dim)
        #factor of 2 from the symmetric distance matrix
        D1 = (na - 2*tf.matmul(y_true[:,xyz_dim:], y_true[:,xyz_dim:], False, True) + nb)/(f_dim-xyz_dim)/2
        d_min = tf.convert_to_tensor(1e-10,dtype=D1.dtype.base_dtype)
        d_max = tf.convert_to_tensor(40.0,dtype=D1.dtype.base_dtype)
        D1 = tf.sqrt(tf.clip_by_value(D1,d_min,d_max))
                     
        na2 = tf.reduce_sum(tf.square(y_pred), 1)
        nb2 = tf.reduce_sum(tf.square(y_pred), 1)
    
        # na as a row and nb as a column vectors
        na2 = tf.reshape(na2, [-1, 1])
        nb2 = tf.reshape(nb2, [1, -1])

        # return pairwise euclidead difference matrix
        #D2 = (tf.sqrt(tf.maximum(na2 - 2*tf.matmul(y_pred, y_pred, False, True) + nb2, 0.0)))/np.sqrt(latent_dim)
        D2 = (na2 - 2*tf.matmul(y_pred, y_pred, False, True) + nb2)/l_dim/2
        D2 = tf.sqrt(tf.clip_by_value(D2,d_min,d_max))
        mbe = K.mean(tf.square(D1-D2))
        
        return mbe  
    return embedded_loss

def RFVE_metric(y_true, y_pred):
    sse = K.sum(K.square(y_true-y_pred))
    _epsilon = tf.convert_to_tensor(1e-7,dtype=sse.dtype.base_dtype)
    sst = K.sum(K.square(y_true-K.mean(y_true)))
    return sse/(sst+_epsilon)

def RV_metric(xyz_dim):    
    def RV_metric(y_true, y_pred):

        # squared norms of each row in A and B
        norm = tf.reduce_sum(tf.square(y_true[:,xyz_dim:]), 1)    
        # na as a row and nb as a column vectors
        na = tf.reshape(norm, [-1, 1])
        nb = tf.reshape(norm, [1, -1])
        # return pairwise euclidead difference matrix
        dist_raw = tf.sqrt(tf.maximum(na - 2*tf.matmul(y_true[:,xyz_dim:], y_true[:,xyz_dim:], False, True) + nb, 0.0))
        _epsilon = tf.convert_to_tensor(1e-7,dtype=dist_raw.dtype.base_dtype)
        #dist_raw = dist_raw/(tf.reduce_max(dist_raw)+_epsilon)
        
        norm = tf.reduce_sum(tf.square(y_pred), 1)    
        na = tf.reshape(norm, [-1, 1])
        nb = tf.reshape(norm, [1, -1])
        dist_emb = tf.sqrt(tf.maximum(na - 2*tf.matmul(y_pred, y_pred, False, True) + nb, 0.0))
        #dist_emb = dist_emb/(tf.reduce_max(dist_emb)+_epsilon)
        
        dist_raw = dist_raw - K.mean(dist_raw)
        dist_emb = dist_emb - K.mean(dist_emb)
        r = K.sum(dist_raw*dist_emb)/(tf.sqrt(tf.maximum( K.sum(K.square(dist_raw))*K.sum(K.square(dist_emb)) , 0.0 ))+_epsilon)
    
        return 1-r**2
    return RV_metric
