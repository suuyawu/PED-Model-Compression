import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import scipy.misc

var = 0.001

################## KL-based Upper Bound on Entropy of Mixture of Gaussians ###################
def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + np.transpose(x2) - 2*np.dot(X, np.transpose(X))
    return dists

def get_shape(x):
    dims = np.float32(np.shape(x)[1])
    N = np.float32(np.shape(x)[0])
    return dims, N

def logsumexp(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return scipy.misc.logsumexp(x, axis=axis, keepdims=keepdims)

def entropy_estimator_kl(x, flag):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    if flag == "up":
        dists2 = dists / (2*var)
    else:
        dists2 = dists / (4*(2*var))
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -np.mean(lprobs)
    return dims/2 + h

def kde_condentropy(output):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

def hot_vec_lable(labels, k):
    labelixs = {}
    for i in range(k):
        labelixs[i] = labels == i
        labelixs[i] =  labelixs[i].numpy()
    return labelixs

def cal_info(layerdata, labels, k):
    labelixs = hot_vec_lable(labels, k)
    H_LAYER = entropy_estimator_kl(layerdata, "up")
    H_LAYER_GIVEN_OUTPUT = 0
    H_LAYER_GIVEN_INPUT = kde_condentropy(layerdata)
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * entropy_estimator_kl(layerdata[ixs,:], "low")
    return H_LAYER - H_LAYER_GIVEN_OUTPUT, 0
