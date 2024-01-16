# some additional functions for GLM with discrete choices
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from ssm.util import one_hot
import numpy as onp
npr.seed(65)  # set seed in case of randomization

def binary_similarity(x=None, y=None):
    if np.max(x) > 1:
        raise ValueError("Use binary input")
    if y is None:
        p = 0
    else:
        if not len(x) == len(y):
           raise ValueError("x and y needs to have same length")
        p = 1- np.sum(np.absolute(x-y))/len(x)

    return p

def dissimilarity(p):
    tau = np.sqrt(1-p)
    return tau

def inclusive_value(a, tau, axis=2, keepdims=True):
    b = np.divide(a, tau)
    out = logsumexp(b, axis=axis, keepdims=keepdims)
    return out

def logsumexpiv(ivs, taus, C, axis=2, keepdims=True):
    b = np.multiply(ivs, taus)
    out = np.tile(logsumexp(b, axis=axis, keepdims=keepdims), (1,1,C))
    return out

def calculate_tau(outcome, od, C):
    if not outcome.ndim == 2:
        raise NotImplementedError
    
    one_hot_outcomes = one_hot(outcome, C)
    taus = []

    for j, (key, value) in enumerate(od.items()): 
        # calculate tau for outcomes within a nest
        outcomes_across_trials = one_hot_outcomes[:,:,value]
        oats2 = outcomes_across_trials.reshape(-1, len(value)).T
        if len(value) == 1:
            p_i = binary_similarity(oats2)
        else:
            p_i = binary_similarity(oats2[0,:], oats2[1,:])

        tau_i = dissimilarity(p_i)
        taus.append(tau_i)

    return taus

def calculate_NestedMultinominalLogit(arr, od, taus_list: list, C):
    # calculate nested logit
    # https://journals.sagepub.com/doi/10.1177/1536867X0200200301
    if not arr.ndim == 3:
        raise NotImplementedError

    taus = np.zeros(C)
    IVs_arr = onp.zeros(arr.shape) # different between trials
    IVs_for_logsumexpvi = []
    for j, (key, value) in enumerate(od.items()): 
        outcomes_across_trials = arr[:,:,value]
        # onp.take(arr, value, axis=2, mode='raise')
        
        # retrieve tau for outcomes within a nest
        tau_i = taus_list[j]
        np.put(taus, value, np.repeat([tau_i], len(value)), mode='raise')

        # calculate IV for outcomes within a nest
        iv_i = inclusive_value(outcomes_across_trials, tau_i, keepdims=True)

        tmp = np.zeros(C)
        np.put(tmp, value, np.repeat([1], len(value)), mode='raise')
        tmp_2 = np.expand_dims(np.tile(tmp, (arr.shape[0], 1)), axis=1)
        IVs_arr += np.multiply(tmp_2, iv_i) 
        
        IVs_for_logsumexpvi.append(iv_i)

    taus_arr = np.expand_dims(np.tile(taus, (arr.shape[0], 1)), axis=1)
    IVs_for_logsumexpvi2 = np.concatenate(IVs_for_logsumexpvi, axis=2)
    lsei = logsumexpiv(IVs_for_logsumexpvi2, taus_list, C, axis=2, keepdims=True)
    out1 = np.add(np.divide(arr, taus_arr), np.multiply((taus_arr-1), IVs_arr))
    out2 = np.add(out1, -lsei) # a/tau - IV + tau IV - logsumexpiv
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    
    return out2

def nested_categorical_logpdf(data, logits, mask=None):
    D = data.shape[-1]
    C = logits.shape[-1]
    assert data.dtype in (int, np.int8, np.int16, np.int32, np.int64)
    assert np.all((data >= 0) & (data < C))
    assert logits.shape[-2] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    # logits = logits - logsumexp(logits, axis=-1, keepdims=True)      # (..., D, C)
    x = one_hot(data, C)                                             # (..., D, C)
    lls = np.sum(x * logits, axis=-1)                                # (..., D)
    return np.sum(lls * mask, axis=-1)                               # (...,)