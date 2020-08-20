# coding: utf-8

from math import sin, cos, sqrt
from six.moves import xrange

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance


class Id:

    def __init__(self, Ia, Ib):
        self.Ia = Ia
        self.Ib = Ib

    def show(self):
        print('A: %s\nB: %s' % (self.Ia, self.Ib))


def mesh_max_pooling(input, mapping):
    # minnum = -float('inf')
    minnum = 0.0
    input_dim = 9
    padding_feature = minnum * tf.ones([tf.shape(input)[0], 1, tf.shape(input)[2]], tf.float64)

    padded_input = tf.concat([padding_feature, input], 1)

    def compute_nb_feature(input_f):
        return tf.gather(input_f, mapping)

    total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
    # max_nb_index = tf.argmax(tf.abs(total_nb_feature)-, 2)
    # max_nb_feature = total_nb_feature[:,:,max_nb_index,:]

    max_nb_feature_plus = tf.reduce_max(total_nb_feature, reduction_indices=[2])

    max_nb_feature_minus = tf.reduce_max(-total_nb_feature, reduction_indices=[2])
    max_nb_feature = tf.where(max_nb_feature_plus > max_nb_feature_minus, max_nb_feature_plus, -max_nb_feature_minus)

    return max_nb_feature


def mesh_max_depooling(input, mapping):
    # minnum = -float('inf')
    input_dim = 9
    padding_feature = tf.zeros([tf.shape(input)[0], 1, tf.shape(input)[2]], tf.float64)

    padded_input = tf.concat([padding_feature, input], 1)

    def compute_nb_feature(input_f):
        return tf.gather(input_f, mapping)

    total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
    max_nb_feature = tf.reduce_max(total_nb_feature, axis=2)
    # max_nb_feature = tf.reshape(total_nb_feature, [tf.shape(input)[0], tf.shape(mapping)[0], tf.shape(input)[2]])
    # max_nb_feature = tf.reduce_max(total_nb_feature, reduction_indices=[2])

    return max_nb_feature


def mesh_mean_pooling(input, mapping, degree):
    # minnum = -float('inf')

    input_dim = 9
    padding_feature = tf.zeros([tf.shape(input)[0], 1, tf.shape(input)[2]], tf.float64)

    padded_input = tf.concat([padding_feature, input], 1)

    def compute_nb_feature(input_f):
        return tf.gather(input_f, mapping)

    total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
    mean_nb_feature = tf.reduce_sum(total_nb_feature, axis=2) / degree

    return mean_nb_feature


def mesh_mean_depooling(input, mapping, degree):
    input_dim = 9
    # padding_feature = tf.zeros([tf.shape(input)[0], 1, tf.shape(input)[2]], tf.float64)

    padded_input = input

    def compute_nb_feature(input_f):
        return tf.gather(input_f, mapping)

    total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
    mean_nb_feature = tf.reduce_sum(total_nb_feature, axis=2) / degree

    return mean_nb_feature


def load_data(path, resultmin, resultmax, useS=True, graphconv=False):
    data = h5py.File(path)
    datalist = data.keys()
    logr = np.transpose(data['FLOGRNEW'], (2, 1, 0))
    s = np.transpose(data['FS'], (2, 1, 0))
    # neighbour = data['neighbour']
    neighbour1 = np.transpose(data['neighbour1'])
    neighbour2 = np.transpose(data['neighbour2'])
    mapping = np.transpose(data['mapping'])
    cotweight1 = np.transpose(data['cotweight1'])
    cotweight2 = np.transpose(data['cotweight2'])

    pointnum1 = neighbour1.shape[0]
    pointnum2 = neighbour2.shape[0]
    maxdegree1 = neighbour1.shape[1]
    maxdegree2 = neighbour2.shape[1]
    modelnum = len(logr)

    logrmin = logr.min()
    logrmin = logrmin - 1e-6
    logrmax = logr.max()
    logrmax = logrmax + 1e-6
    smin = s.min()
    smin = smin - 1e-6
    smax = s.max()
    smax = smax + 1e-6

    rnew = (resultmax - resultmin) * (logr - logrmin) / (logrmax - logrmin) + resultmin
    snew = (resultmax - resultmin) * (s - smin) / (smax - smin) + resultmin
    if useS:
        feature = np.concatenate((rnew, snew), axis=2)
    else:
        feature = rnew

    f = np.zeros_like(feature).astype('float64')
    f = feature

    nb1 = np.zeros((pointnum1, maxdegree1)).astype('float64')
    nb1 = neighbour1

    nb2 = np.zeros((pointnum2, maxdegree2)).astype('float64')
    nb2 = neighbour2

    L1 = np.zeros((pointnum1, pointnum1)).astype('float64')
    L2 = np.zeros((pointnum2, pointnum2)).astype('float64')

    if graphconv and 'W1' in datalist:
        L = np.transpose(data['W1'])
        L1 = L
        L1 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L1)))

    if graphconv and 'W2' in datalist:
        L = np.transpose(data['W2'])
        L2 = L
        L2 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L2)))

    cotw1 = np.zeros((cotweight1.shape[0], cotweight1.shape[1], 1)).astype('float64')
    cotw2 = np.zeros((cotweight2.shape[0], cotweight2.shape[1], 1)).astype('float64')
    for i in range(1):
        cotw1[:, :, i] = cotweight1
        cotw2[:, :, i] = cotweight2

    degree1 = np.zeros((neighbour1.shape[0], 1)).astype('float64')
    for i in range(neighbour1.shape[0]):
        degree1[i] = np.count_nonzero(nb1[i])

    degree2 = np.zeros((neighbour2.shape[0], 1)).astype('float64')
    for i in range(neighbour2.shape[0]):
        degree2[i] = np.count_nonzero(nb2[i])

    mapping1 = np.zeros((pointnum2, mapping.shape[1])).astype('float64')
    maxdemapping = np.zeros((pointnum1, 1)).astype('float64')

    mapping1_col = mapping.shape[1]

    mapping1 = mapping
    # mapping2 = demapping
    for i in range(pointnum1):
        # print i
        idx = np.where(mapping1 == i + 1)
        if idx[1][0] > 0:
            maxdemapping[i] = 1
        else:
            maxdemapping[i] = idx[0][0] + 1

    meanpooling_degree = np.zeros((mapping.shape[0], 1)).astype('float64')
    for i in range(mapping.shape[0]):
        meanpooling_degree[i] = np.count_nonzero(mapping1[i])

    meandepooling_mapping = np.zeros((pointnum1, 1)).astype('float64')
    meandepooling_degree = np.zeros((pointnum1, 1)).astype('float64')

    for i in range(pointnum1):
        idx = np.where(mapping1 == i + 1)[0]
        meandepooling_mapping[i] = idx[0]
        meandepooling_degree[i] = meanpooling_degree[idx[0]]

    return f, nb1, degree1, mapping1, nb2, degree2, maxdemapping, meanpooling_degree, meandepooling_mapping, meandepooling_degree, \
        logrmin, logrmax, smin, smax, modelnum, pointnum1, pointnum2, maxdegree1, maxdegree2, mapping1_col, L1, L2, cotw1, cotw2


def load_data1(path, resultmin, resultmax, useS=True, graphconv=False):
    data = h5py.File(path)
    datalist = data.keys()
    logr = np.transpose(data['FLOGRNEW'], (2, 1, 0))
    s = np.transpose(data['FS'], (2, 1, 0))
    # neighbour = data['neighbour']
    neighbour1 = np.transpose(data['neighbour1'])
    neighbour2 = np.transpose(data['neighbour2'])
    neighbour3 = np.transpose(data['neighbour3'])
    mapping1 = np.transpose(data['mapping1'])
    mapping2 = np.transpose(data['mapping2'])
    cotweight1 = np.transpose(data['cotweight1'])
    cotweight2 = np.transpose(data['cotweight2'])
    cotweight3 = np.transpose(data['cotweight3'])

    pointnum1 = neighbour1.shape[0]
    pointnum2 = neighbour2.shape[0]
    pointnum3 = neighbour3.shape[0]
    maxdegree1 = neighbour1.shape[1]
    maxdegree2 = neighbour2.shape[1]
    maxdegree3 = neighbour3.shape[1]
    modelnum = len(logr)

    logrmin = logr.min()
    logrmin = logrmin - 1e-6
    logrmax = logr.max()
    logrmax = logrmax + 1e-6
    smin = s.min()
    smin = smin - 1e-6
    smax = s.max()
    smax = smax + 1e-6

    rnew = (resultmax - resultmin) * (logr - logrmin) / (logrmax - logrmin) + resultmin
    snew = (resultmax - resultmin) * (s - smin) / (smax - smin) + resultmin
    if useS:
        feature = np.concatenate((rnew, snew), axis=2)
    else:
        feature = rnew

    f = np.zeros_like(feature).astype('float64')
    f = feature

    nb1 = np.zeros((pointnum1, maxdegree1)).astype('float64')
    nb1 = neighbour1

    nb2 = np.zeros((pointnum2, maxdegree2)).astype('float64')
    nb2 = neighbour2

    nb3 = np.zeros((pointnum3, maxdegree3)).astype('float64')
    nb3 = neighbour3

    L1 = np.zeros((pointnum1, pointnum1)).astype('float64')
    L2 = np.zeros((pointnum2, pointnum2)).astype('float64')
    L3 = np.zeros((pointnum3, pointnum3)).astype('float64')

    if graphconv and 'W1' in datalist:
        L = np.transpose(data['W1'])
        L1 = L
        L1 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L1)))

    if graphconv and 'W2' in datalist:
        L = np.transpose(data['W2'])
        L2 = L
        L2 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L2)))

    if graphconv and 'W3' in datalist:
        L = np.transpose(data['W3'])
        L3 = L
        L3 = rescale_L(Laplacian(scipy.sparse.csr_matrix(L3)))

    cotw1 = np.zeros((cotweight1.shape[0], cotweight1.shape[1], 1)).astype('float64')
    cotw2 = np.zeros((cotweight2.shape[0], cotweight2.shape[1], 1)).astype('float64')
    cotw3 = np.zeros((cotweight3.shape[0], cotweight3.shape[1], 1)).astype('float64')
    for i in range(1):
        cotw1[:, :, i] = cotweight1
        cotw2[:, :, i] = cotweight2
        cotw3[:, :, i] = cotweight3

    degree1 = np.zeros((neighbour1.shape[0], 1)).astype('float64')
    for i in range(neighbour1.shape[0]):
        degree1[i] = np.count_nonzero(nb1[i])

    degree2 = np.zeros((neighbour2.shape[0], 1)).astype('float64')
    for i in range(neighbour2.shape[0]):
        degree2[i] = np.count_nonzero(nb2[i])

    degree3 = np.zeros((neighbour3.shape[0], 1)).astype('float64')
    for i in range(neighbour3.shape[0]):
        degree3[i] = np.count_nonzero(nb3[i])

    mapping11 = np.zeros((pointnum2, mapping1.shape[1])).astype('float64')
    maxdemapping1 = np.zeros((pointnum1, 1)).astype('float64')

    mapping12 = np.zeros((pointnum3, mapping2.shape[1])).astype('float64')
    maxdemapping2 = np.zeros((pointnum2, 1)).astype('float64')

    mapping11_col = mapping1.shape[1]
    mapping12_col = mapping2.shape[1]

    mapping11 = mapping1
    mapping12 = mapping2
    # mapping2 = demapping
    for i in range(pointnum1):
        # print i
        idx = np.where(mapping11 == i + 1)
        if idx[1][0] > 0:
            maxdemapping1[i] = 0
        else:
            maxdemapping1[i] = idx[0][0]
    for i in range(pointnum2):
        # print i
        idx = np.where(mapping12 == i + 1)
        if idx[1][0] > 0:
            maxdemapping2[i] = 0
        else:
            maxdemapping2[i] = idx[0][0]

    meanpooling_degree1 = np.zeros((mapping1.shape[0], 1)).astype('float64')
    for i in range(mapping1.shape[0]):
        meanpooling_degree1[i] = np.count_nonzero(mapping11[i])

    meanpooling_degree2 = np.zeros((mapping2.shape[0], 1)).astype('float64')
    for i in range(mapping2.shape[0]):
        meanpooling_degree2[i] = np.count_nonzero(mapping12[i])

    meandepooling_mapping1 = np.zeros((pointnum1, 1)).astype('float64')
    meandepooling_degree1 = np.zeros((pointnum1, 1)).astype('float64')

    meandepooling_mapping2 = np.zeros((pointnum2, 1)).astype('float64')
    meandepooling_degree2 = np.zeros((pointnum2, 1)).astype('float64')

    for i in range(pointnum1):
        idx = np.where(mapping11 == i + 1)[0]
        meandepooling_mapping1[i] = idx[0]
        meandepooling_degree1[i] = meanpooling_degree1[idx[0]]

    for i in range(pointnum2):
        idx = np.where(mapping12 == i + 1)[0]
        meandepooling_mapping2[i] = idx[0]
        meandepooling_degree2[i] = meanpooling_degree2[idx[0]]

    return f, nb1, degree1, mapping11, nb2, degree2, mapping12, nb3, degree3, \
        maxdemapping1, meanpooling_degree1, meandepooling_mapping1, meandepooling_degree1, \
        maxdemapping2, meanpooling_degree2, meandepooling_mapping2, meandepooling_degree2, \
        logrmin, logrmax, smin, smax, modelnum, pointnum1, pointnum2, pointnum3, maxdegree1, maxdegree2, maxdegree3, \
        mapping11_col, mapping12_col, L1, L2, L3, cotw1, cotw2, cotw3


def recover_data(recover_feature, logrmin, logrmax, smin, smax, resultmin, resultmax, useS=True):
    logr = recover_feature[:, :, 0:3]

    logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin
    # feature=[]
    if useS:
        s = recover_feature[:, :, 3:9]
        s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
        logr = np.concatenate((logr, s), axis=2)

    return logr


def linear1(input_, matrix, output_size, name='Linear', stddev=0.02, bias_start=0.0):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        # matrix = tf.get_variable("weights", [input_size, output_size], tf.float64,
        #                          tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float64,
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias


def leaky_relu(input_, alpha=0.02):
    return tf.maximum(input_, alpha * input_)


def batch_norm_wrapper(inputs, name='batch_norm', is_training=False, decay=0.9, epsilon=1e-5):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float64, trainable=True,
                                    initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float64))
            beta = tf.get_variable('beta', dtype=tf.float64, trainable=True,
                                   initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float64))
            pop_mean = tf.get_variable('overallmean', dtype=tf.float64, trainable=False,
                                       initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float64))
            pop_var = tf.get_variable('overallvar', dtype=tf.float64, trainable=False,
                                      initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float64))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float64, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float64, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float64, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float64, trainable=False)

        if is_training == True:
            axis = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

# -----------------------------------------------------graph conv--------------------------------


def Laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    # d=d.astype(W.dtype)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0.0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr_matrix
    return L


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L


def chebyshev(L, X, K):
    """Return T_k X where T_k are the Chebyshev polynomials of order up to K.
    Complexity is O(KMN)."""
    M, N = X.shape
    # assert L.dtype == X.dtype

    # L = rescale_L(L, lmax)
    # Xt = T @ X: MxM @ MxN.
    Xt = np.empty((K, M, N), L.dtype)
    # Xt_0 = T_0 X = I X = X.
    Xt[0, ...] = X
    # Xt_1 = T_1 X = L X.
    if K > 1:
        Xt[1, ...] = L.dot(X)
    # Xt_k = 2 L Xt_k-1 - Xt_k-2.
    for k in range(2, K):
        Xt[k, ...] = 2 * L.dot(Xt[k - 1, ...]) - Xt[k - 2, ...]
    return Xt


def graph_conv2(x, L, Fout, W, K, name='graph_conv', training=True, special_activation=True, no_activation=False, bn=True):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        N, M, Fin = x.get_shape()
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        x = tf.matmul(x, W)  # N*M x Fout

        x = tf.reshape(x, [-1, M, Fout])  # N x M x Fout

        if not bn:
            fb = x
        else:
            fb = batch_norm_wrapper(x, is_training=training)

        if no_activation:
            fa = fb
        elif not special_activation:
            fa = leaky_relu(fb)
        else:
            fa = tf.nn.tanh(fb)

        return fa
