"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy
import random
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import cv2
from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, train_nn_data_aug, GlobalAvgPoolLayer
import timeit
import inspect
import sys
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
import matplotlib

import matplotlib.pyplot as plt

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(learning_rate=0.1, n_epochs=200,
                    nkerns=[32, 64], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(12345)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 6 * 6),
    # or (500, 64 * 6 * 6) = (500, 2304) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=T.tanh
    )
    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1
    train_nn(train_model, validate_model, test_model, n_train_batches,
        n_valid_batches, n_test_batches, n_epochs, verbose=True)

# Problem 2.1
# Write a function to add translations
def translate_image(imgs, labels, x, y, max_num_sample, fill_constant):
    rng = numpy.random.RandomState(12345)
    imgs_tmp = numpy.array(imgs, dtype = theano.config.floatX)
    num_imgs = imgs_tmp.shape[0]
    shuffled_indices = numpy.random.permutation(numpy.arange(num_imgs))
    sampled_indices = shuffled_indices[:min(max_num_sample, num_imgs)]
    num_sampled = sampled_indices.size
    aug_data = numpy.ndarray(shape=(num_sampled, 32*32*3), dtype = theano.config.floatX)
    index = 0
    for i in sampled_indices:
        img = imgs_tmp[i,:].reshape(3,32,32).transpose(1,2,0)
        img_x = numpy.roll(img, shift = x, axis = 1)
        if x >= 0:
            img_x[:,:x,:] = fill_constant
        else:
            img_x[:,x:,:] = fill_constant
        img_xy = numpy.roll(img_x, shift = y, axis = 0)
        if  y >= 0:
            img_xy[:y,:,:] = fill_constant
        else:
            img_xy[y:,:,:] = fill_constant
        aug_data[index] = img_xy.transpose(2,0,1).flatten()
        index = index + 1
    aug_labels = numpy.array(labels[sampled_indices], dtype = labels.dtype)
    return (aug_data, aug_labels)
    
#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(learning_rate=0.1, n_epochs=200, num_samples = 5000, nkerns=[32, 64], batch_size=500):
    x = random.randint(-3,2) 
    y = random.randint(-2,3)
    num_sampled = num_samples
    fill_constant = 0
    train_nn_data_aug(translate_image, learning_rate, batch_size, nkerns, n_epochs, True, x, y, num_sampled, fill_constant)
#Problem 2.2
#Write a function to add roatations
def rotate_image(imgs, labels, max_num_sample, height, width, angle, scaling):
    rng = numpy.random.RandomState(12345)
    imgs_tmp = numpy.array(imgs, dtype = theano.config.floatX)
    num_imgs = imgs_tmp.shape[0]
    shuffled_indices = numpy.random.permutation(numpy.arange(num_imgs))
    sampled_indices = shuffled_indices[:min(max_num_sample, num_imgs)]
    num_sampled = sampled_indices.size
    aug_data = numpy.ndarray(shape=(num_sampled, 32*32*3), dtype = theano.config.floatX)
    index = 0
    M = cv2.getRotationMatrix2D((height/2,width/2),angle, scaling)
    for i in sampled_indices:
        img = imgs_tmp[i,:].reshape(3,32,32).transpose(1,2,0)
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]   
        dst_r = cv2.warpAffine(img_r,M,(height,width))
        dst_g = cv2.warpAffine(img_g,M,(height,width))
        dst_b = cv2.warpAffine(img_b,M,(height,width))        
        img[:,:,0] = dst_r
        img[:,:,1] = dst_g
        img[:,:,2] = dst_b        
        aug_data[index] = img.transpose(2,0,1).flatten()
        index = index + 1
    aug_labels = numpy.array(labels[sampled_indices], dtype = labels.dtype)
    return (aug_data, aug_labels)
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(learning_rate=0.1, n_epochs=200, num_samples = 5000, nkerns=[32, 64], batch_size=500):
    angle = random.randint(-360, 360)
    scaling = 1
    train_nn_data_aug(rotate_image, learning_rate, batch_size, nkerns, n_epochs, True, num_samples, 32, 32, angle, scaling)
#Problem 2.3
#Write a function to flip images
def flip_image(imgs, labels, max_num_sample):
    rng = numpy.random.RandomState(12345)
    imgs_tmp = numpy.array(imgs, dtype = theano.config.floatX)
    num_imgs = imgs_tmp.shape[0]
    shuffled_indices = numpy.random.permutation(numpy.arange(num_imgs))
    sampled_indices = shuffled_indices[:min(max_num_sample, num_imgs)]
    num_sampled = sampled_indices.size
    aug_data = numpy.ndarray(shape=(num_sampled, 32*32*3), dtype = theano.config.floatX)
    index = 0
    for i in sampled_indices:
        img = cv2.flip(imgs_tmp[i,:].reshape(3,32,32).transpose(1,2,0),1)
        aug_data[index] = img.transpose(2,0,1).flatten()
        index = index + 1
    aug_labels = numpy.array(labels[sampled_indices], dtype = labels.dtype)
    return (aug_data, aug_labels)
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(learning_rate=0.1, n_epochs=200, num_samples = 5000, nkerns=[32, 64], batch_size=500):
    train_nn_data_aug(flip_image, learning_rate, batch_size, nkerns, n_epochs, True, num_samples)
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(imgs, labels, noise_dist, max_num_sample):
    rng = numpy.random.RandomState(12345)
    imgs_tmp = numpy.array(imgs, dtype = theano.config.floatX)
    num_imgs = imgs_tmp.shape[0]
    shuffled_indices = numpy.random.permutation(numpy.arange(num_imgs))
    sampled_indices = shuffled_indices[:min(max_num_sample, num_imgs)]
    num_sampled = sampled_indices.size
    aug_data = numpy.ndarray(shape=(num_sampled, 32*32*3), dtype = theano.config.floatX)
    index = 0
    noise = None
    for i in sampled_indices:
        if noise_dist == 0:
            noise = random.gauss(0, 0.1)
        elif noise_dist == 1:
            noise = random.uniform(-0.1,0.1)
        img = imgs_tmp[i,:].reshape(3,32,32).transpose(1,2,0) + noise
        aug_data[index] = img.transpose(2,0,1).flatten()
        index = index + 1
    aug_labels = numpy.array(labels[sampled_indices], dtype = labels.dtype)
    return (aug_data, aug_labels)
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(learning_rate=0.1, n_epochs=200, num_samples = 5000, nkerns=[32, 64], batch_size=500):
    train_nn_data_aug(noise_injection, learning_rate, batch_size, nkerns, n_epochs, True, num_samples)

#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 10e-5
    #   covariance matrix
    cov = numpy.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = numpy.linalg.eig(cov)
    #   D = diag(d) ^ (-1/2)
    D = numpy.diag(1. / numpy.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = numpy.dot(numpy.dot(E, D), E.T)
    X_white = numpy.dot(X, W)
    return X_white

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False, sqrt_bias=0.1, min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X

def MY_lenet():
    verbose = True
    rng = numpy.random.RandomState(12345)
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    batch_size = 200
    learning_rate = 0.1
    n_epochs = 500
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    # Open a file to save all the information during the training
    target = open("MyLet.txt","w")
    
    # Do global contrast normalization and whitening
    '''
    print("Doing normalization and whitening...")
    target.write("Doing normalization and whitening...\n")
    train_x_tmp = train_set_x.get_value()
    train_x = train_x_tmp.reshape(n_train_batches*batch_size, 3, 32*32)
    valid_x_tmp = valid_set_x.get_value()
    valid_x = valid_x_tmp.reshape(n_valid_batches*batch_size, 3, 32*32)
    test_x_tmp = test_set_x.get_value()
    test_x = test_x_tmp.reshape(n_test_batches*batch_size, 3, 32*32)
    train_x_norm_white = numpy.empty_like(train_x, dtype = train_x.dtype)
    valid_x_norm_white = numpy.empty_like(valid_x, dtype = valid_x.dtype)
    test_x_norm_white = numpy.empty_like(test_x, dtype = test_x.dtype)
    for i in range(0,2):
        train_x_norm_white[:,i,:] = zca_whiten(global_contrast_normalize(train_x[:,i,:], scale=1., subtract_mean=True, use_std=True, sqrt_bias=0., min_divisor=1e-8))
        valid_x_norm_white[:,i,:] = zca_whiten(global_contrast_normalize(valid_x[:,i,:], scale=1., subtract_mean=True, use_std=True, sqrt_bias=0., min_divisor=1e-8))
        test_x_norm_white[:,i,:] = zca_whiten(global_contrast_normalize(test_x[:,i,:], scale=1., subtract_mean=True, use_std=True, sqrt_bias=0., min_divisor=1e-8))
    train_set_x.set_value(train_x_norm_white.reshape(n_train_batches*batch_size, 32*32*3))
    valid_set_x.set_value(valid_x_norm_white.reshape(n_valid_batches*batch_size, 32*32*3))
    test_set_x.set_value(test_x_norm_white.reshape(n_valid_batches*batch_size, 32*32*3))
    print("Norm and Whitening finish...")
    print("num of nan:", numpy.sum(numpy.isnan(train_set_x.get_value())))
    target.write("Norm and Whitening finish...\n")
    '''
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    #target.write('... building the model\n')
    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    mlpkern = 150
    nkerns = [200,250,300]
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 6, 6),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )
    '''
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, mlpkern, 30, 30),
        filter_shape=(mlpkern, mlpkern, 1, 1),
        poolsize=(2, 2)
    )
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, mlpkern, 15, 15),
        filter_shape=(nkerns[1], mlpkern, 3, 3),
        poolsize=(2, 2)
    )
    
    layer5 = LeNetConvPoolLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, nkerns[1], 6, 6),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )
    
    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, mlpkern, 13, 13),
        filter_shape=(mlpkern, mlpkern, 1, 1),
        poolsize=(2, 2)
    )
    layer8 = LeNetConvPoolLayer(
        rng,
        input=layer7.output,
        image_shape=(batch_size, mlpkern, 6, 6),
        filter_shape=(nkerns[2], mlpkern, 3, 3),
        poolsize=(1, 1)
    )
    
    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer8.output,
        image_shape=(batch_size, nkerns[2], 4, 4),
        filter_shape=(10, nkerns[2], 1, 1),
        poolsize=(2, 2)
    )
    
    layer12 = GlobalAvgPoolLayer(
        input=layer5.output,
        image_shape=(2,2)
    )
    print("Using GlobalAvgPoolLayer to do classification...")
    '''
    layer12_input = layer2.output.flatten(2)
    layer12 = HiddenLayer(
        rng,
        input=layer12_input,
        n_in=nkerns[2] * 2 * 2,
        n_out=1024,
        activation=T.tanh
    )
    
    layer13 = HiddenLayer(
        rng,
        input=layer12.output,
        n_in=1024,
        n_out=512,
        activation=T.tanh
    )
    layer14 = LogisticRegression(input=layer13.output, n_in=512, n_out=10)
    cost = layer14.negative_log_likelihood(y)
    
    params = layer14.params + layer13.params + layer12.params \
    + layer2.params + layer1.params + layer0.params
    #cost = layer12.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer14.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer14.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    #layer14.params + layer13.params + layer12.params +
    # create a list of all model parameters to be fit by gradient descent
    

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.5), name='momentum')
    lr = theano.shared(numpy.cast[theano.config.floatX](0.1), name='lr')
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param + param_update))
        updates.append((param_update, momentum * param_update - lr / batch_size * T.grad(cost, param)))
    '''
    
    ## adaGrad
    epsilon = theano.shared(numpy.cast[theano.config.floatX](0.1), name = 'epsilon')
    delta = theano.shared(numpy.cast[theano.config.floatX](1.0), name = 'delta')
    updates = []
    for param in  params:
        accum = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - epsilon / (delta + T.sqrt(accum)) * T.grad(cost, param)))
        updates.append((accum, accum + T.sqr(T.grad(cost, param))))
    
    # end-snippet-1

    # early-stopping parameters
    patience = 20000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    #print("Call train_nn")
    print("Start training now")
    '''train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    '''
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    data_aug_flip, labels_aug_flip = flip_image(train_set_x.eval(), train_set_y.eval(), 40000)
    data_aug_gpu_flip = theano.shared(numpy.asarray(data_aug_flip, dtype = theano.config.floatX))
    labels_aug_gpu_flip = T.cast(theano.shared(numpy.asarray(labels_aug_flip, dtype = theano.config.floatX)), "int32")

        
    train_aug_flip_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data_aug_gpu_flip[index * batch_size: (index + 1) * batch_size],
            y: labels_aug_gpu_flip[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    data_aug_trans, labels_aug_trans = translate_image(train_set_x.eval(), train_set_y.eval(), 1, 1, 40000, 0)
    data_aug_gpu_trans = theano.shared(numpy.asarray(data_aug_trans, dtype = theano.config.floatX))
    labels_aug_gpu_trans = T.cast(theano.shared(numpy.asarray(labels_aug_trans, dtype = theano.config.floatX)), "int32")
    aug_data_train_batches = 40000 // batch_size    

    train_aug_trans_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data_aug_gpu_trans[index * batch_size: (index + 1) * batch_size],
            y: labels_aug_gpu_trans[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # train flip data first
        print("Training flip data...")
        for minibatch_index in range(aug_data_train_batches):
            cost_ij = train_aug_flip_model(minibatch_index)
        print("Training flip data over...")
        
        print("Training translation data...")
        x = random.randint(-3,2) 
        y = random.randint(-2,3)
        data_aug_trans, labels_aug_trans = translate_image(train_set_x.eval(), train_set_y.eval(), x, y, 40000, 0)
        data_aug_gpu_trans = theano.shared(numpy.asarray(data_aug_trans, dtype = theano.config.floatX))
        labels_aug_gpu_trans = T.cast(theano.shared(numpy.asarray(labels_aug_trans, dtype = theano.config.floatX)), "int32")
        for minibatch_index in range(aug_data_train_batches):
            cost_ij = train_aug_trans_model(minibatch_index)
        print("Training translation data over...")
        
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
                target.write(('training @ iter = {0}\n'.format(iter)))

            cost = train_model(minibatch_index)
            

            if (iter + 1) % validation_frequency == 0:
                
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100))
                    target.write('epoch %i, minibatch %i/%i, validation error %f %%\n' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))
                        target.write(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%\n') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    target.write('Optimization complete.')
    target.write('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    target.write(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    target.close()
   
#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN():
    verbose = True
    rng = numpy.random.RandomState(12345)
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    batch_size = 200
    learning_rate = 0.1
    n_epochs = 500
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    # Open a file to save all the information during the training
    target = open("MyCNN.txt","w")
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    images = x.reshape((batch_size, 3, 32, 32))
    mat = numpy.random.rand(32,32)
    mask = mat > 0.3
    tmp = numpy.zeros((3,32,32), dtype = theano.config.floatX)
    tmp[0] = mask
    tmp[1] = mask
    tmp[2] = mask
    #tmp_t = theano.shared(tmp, borrow = True)
    border = 'half'
    tmp_t = theano.shared(tmp, borrow = True)
    layer0_input = tmp_t * images
    
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(2, 2),
        conv_border = border
    )
    
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )
    
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        poolsize=(2, 2),
        conv_border = border
    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )
    
    #bu = bilinear_upsampling(layer4.output, ratio = 2, batch_size=batch_size, num_input_channels=256, )
    bu = layer4.output.repeat(2, axis=2)
    bu = bu.repeat(2, axis=3)
    
    layer5 = LeNetConvPoolLayer(
        rng,
        input=bu,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )
    layer6 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )
    add_1 = layer6.output + layer3.conv_output
    #bu_2 = bilinear_upsampling(add_1, 2, batch_size=batch_size, num_input_channels=128)
    bu_2 = add_1.repeat(2, axis = 2)
    bu_2 = bu_2.repeat(2, axis = 3)
    layer7 = LeNetConvPoolLayer(
        rng,
        input=bu_2,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    )    
    layer8 = LeNetConvPoolLayer(
        rng,
        input=layer7.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    ) 
    
    add_2 = layer8.output + layer1.conv_output
    layer9 = LeNetConvPoolLayer(
        rng,
        input=add_2,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),
        poolsize=(1, 1),
        conv_border = border
    ) 
    
    cost = T.mean((layer9.output - images)**2)
    
    params = layer9.params + layer8.params + layer7.params \
    + layer6.params + layer5.params + layer4.params \
    + layer3.params + layer2.params + layer1.params + layer0.params
    

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [],
        [images, layer0_input, layer9.output, cost],
        givens={
            x: test_set_x[0:200]
        }
    )

    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    '''
    grads = T.grad(cost, params)
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
    momentum = theano.shared(numpy.cast[theano.config.floatX](0.5), name='momentum')
    lr = theano.shared(numpy.cast[theano.config.floatX](0.1), name='lr')
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param + param_update))
        updates.append((param_update, momentum * param_update - lr / batch_size * T.grad(cost, param)))
    '''
    
    ## adaGrad
    epsilon = theano.shared(numpy.cast[theano.config.floatX](0.07), name = 'epsilon')
    delta = theano.shared(numpy.cast[theano.config.floatX](1.0), name = 'delta')
    updates = []
    for param in  params:
        accum = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - epsilon / (delta + T.sqrt(accum)) * T.grad(cost, param)))
        updates.append((accum, accum + T.sqr(T.grad(cost, param))))
    

    # early-stopping parameters
    patience = 100*200 # look as this many examples regardless
    print("patience: ", patience)
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    #print("Call train_nn")
    print("Start training now")
    '''train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
    '''
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
 
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
     
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 1000 == 0) and verbose:
                print('training @ iter = ', iter)
                target.write(('training @ iter = {0}\n'.format(iter)))

            cost = train_model(minibatch_index)    
            if (iter + 1) % validation_frequency == 0:
                
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss))
                    target.write('epoch %i, minibatch %i/%i, validation error %f\n' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
           % (best_validation_loss * 100., best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    target.write('Optimization complete.')
    target.write('Best validation error of %f %% obtained at iteration %i, ' %
          (best_validation_loss * 100., best_iter + 1))
    target.write(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    target.close()
    
    result = test_model()
    test_images = result[0]
    corrupted_images = result[1] 
    recons_images= result[2]
    print("recons:", recons_images[0])
    cost = result[3] 

    # plot 8 images
    print("Original : Corrupted : Recontructed")
    f, axes = plt.subplots(8,3,figsize=(20,20))
    for i in range(8):
        plt.axes(axes[i,0])
        plt.imshow(test_images[i].transpose(1,2,0))
        
        plt.axes(axes[i,1])
        plt.imshow(corrupted_images[i].transpose(1,2,0)) 
        
        plt.axes(axes[i,2])
        plt.imshow(recons_images[i].transpose(1,2,0))
    plt.savefig('mycnn.png')

"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

#Problem4
#Implement the convolutional neural network depicted in problem4 
def my_cnn2(batch_size = 200, n_epochs = 1, learning_rate = 0.01, patience = 12000):
    
    # load data
    ds_rate=None
    datasets = load_data(ds_rate=ds_rate,theano_shared=True)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
        
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches  //= batch_size
   
    
    rng = numpy.random.RandomState(23455)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
        
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    layerX_input = x.reshape((batch_size, 3, 32, 32))
    
    mat = numpy.random.rand(32,32)
    mask = mat > 0.3
    tmp = numpy.zeros((3,32,32), dtype = theano.config.floatX)
    tmp[0] = mask
    tmp[1] = mask
    tmp[2] = mask
    border = 'half'
    tmp_t = theano.shared(tmp, borrow = True)
    layerX_output = tmp_t * layerX_input
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layerX_output,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),  # (number of output feature maps, number of input feature maps, height, width)
        poolsize=(1, 1)
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)

    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),  
        poolsize=(2, 2),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 64, 16, 16)

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)
    
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),  
        poolsize=(2, 2),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 128, 8, 8)   
    
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),  
        poolsize=(1, 1),
        conv_border = border        
    )
    # 4D output tensor is thus of shape (batch_size, 256, 8, 8)    
    
    bu = layer4.output.repeat(2, axis=2)
    bu = bu.repeat(2, axis=3)
    # 4D output tensor is thus of shape (batch_size, 256, 16, 16)  
    
    layer6 = LeNetConvPoolLayer(
        rng,
        input=bu,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)      

    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer6.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 128, 16, 16)
    add_1 = layer7.output + layer3.conv_output
    bu_2 = add_1.repeat(2, axis = 2)
    bu_2 = bu_2.repeat(2, axis = 3)
    #layer8 = UpSampleLayer(input=layer7.output+layer3.output_x)
    # 4D output tensor is thus of shape (batch_size, 128, 32, 32)
    
    layer9 = LeNetConvPoolLayer(
        rng,
        input=bu_2,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
    
    layer10 = LeNetConvPoolLayer(
        rng,
        input=layer9.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
    
    layer11 = LeNetConvPoolLayer(
        rng,
        input=layer10.output+layer1.conv_output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),  
        poolsize=(1, 1),
        conv_border = border
    )
    # 4D output tensor is thus of shape (batch_size, 3, 32, 32)  
    

    cost = T.mean((layer11.output - layerX_input)**2)
 
 
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [],
        [layerX_input,layerX_output,layer11.output,cost],
        givens={
            x: test_set_x[0:100]
        }
    )
    
    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent    
    params = layer11.params + layer10.params + layer9.params + layer7.params + layer6.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    
   
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)    
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost, 
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    
    print('... training the model')   
    
    # early-stopping parameters
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_cost = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    verbose = True

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_cost = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_cost = numpy.mean(validation_cost)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation cost %f' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_cost))

                # if we got the best validation score until now
                if this_validation_cost < best_validation_cost:

                    # save best validation score and iteration number
                    best_validation_cost = this_validation_cost
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break
                
    TEST_MODEL_RESULT = test_model()
    GT_Images_T = TEST_MODEL_RESULT[0]
    Drop_Images_T = TEST_MODEL_RESULT[1] 
    Reconstructed_Images_T = TEST_MODEL_RESULT[2]
    cost_list = TEST_MODEL_RESULT[3] 
    

    # plot 8*3 images
    print("Ground Truth, Corrupted Images, and Recontructed Images:")
    f, axarr = plt.subplots(8,3,figsize=(20,20))
    for i in range(8):
        plt.axes(axarr[i,0])
        plt.imshow(np.transpose(GT_Images_T[i],(1,2,0)))
        
        plt.axes(axarr[i,1])
        plt.imshow(np.transpose(Drop_Images_T[i],(1,2,0))) 
        
        plt.axes(axarr[i,2])
        plt.imshow(np.transpose(Reconstructed_Images_T[i],(1,2,0)))

        
    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
           % (best_validation_loss * 100., best_iter + 1))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
    
