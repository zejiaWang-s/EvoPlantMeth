from __future__ import division, print_function

from tensorflow.keras import backend as K
from .utils import get_from_module
from .data import CPG_NAN

def contingency_table(y, z):
    y, z = K.round(y), K.round(z)

    def count_matches(a, b):
        tmp = K.concatenate([a, b])
        return K.sum(K.cast(K.all(tmp, -1), K.floatx()))

    ones, zeros = K.ones_like(y), K.zeros_like(y)
    y_ones, y_zeros = K.equal(y, ones), K.equal(y, zeros)
    z_ones, z_zeros = K.equal(z, ones), K.equal(z, zeros)

    return count_matches(y_ones, z_ones), count_matches(y_zeros, z_zeros), \
           count_matches(y_zeros, z_ones), count_matches(y_ones, z_zeros)

def prec(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp + K.epsilon())

def tpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn + K.epsilon())

def tnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp + K.epsilon())

def fpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn + K.epsilon())

def fnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp + K.epsilon())

def f1(y, z):
    _tpr, _prec = tpr(y, z), prec(y, z)
    return 2 * (_prec * _tpr) / (_prec + _tpr + K.epsilon())

def mcc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) / (K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + K.epsilon())

def acc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fp + fn + K.epsilon())

def _sample_weights(y, mask=None):
    if mask is None: return K.ones_like(y)
    return 1 - K.cast(K.equal(y, mask), K.floatx())

def _cat_sample_weights(y, mask=None):
    return 1 - K.cast(K.equal(K.sum(y, axis=-1), 0), K.floatx())

def cat_acc(y, z):
    weights = _cat_sample_weights(y)
    _acc = K.cast(K.equal(K.argmax(y, axis=-1), K.argmax(z, axis=-1)), K.floatx())
    return K.sum(_acc * weights) / (K.sum(weights) + K.epsilon())

def mse(y, z, mask=CPG_NAN):
    weights = _sample_weights(y, mask)
    return K.sum(K.square(y - z) * weights) / (K.sum(weights) + K.epsilon())

def mae(y, z, mask=CPG_NAN):
    weights = _sample_weights(y, mask)
    return K.sum(K.abs(y - z) * weights) / (K.sum(weights) + K.epsilon())

def pcc(y_true, y_pred, mask=CPG_NAN):
    weights = _sample_weights(y_true, mask)
    y_true, y_pred, weights = K.flatten(y_true), K.flatten(y_pred), K.flatten(weights)
    
    n = K.sum(weights)
    mean_true = K.sum(y_true * weights) / (n + K.epsilon())
    mean_pred = K.sum(y_pred * weights) / (n + K.epsilon())
    
    centered_true, centered_pred = y_true - mean_true, y_pred - mean_pred
    cov = K.sum(centered_true * centered_pred * weights)
    
    var_true = K.sum(K.square(centered_true) * weights)
    var_pred = K.sum(K.square(centered_pred) * weights)
    
    return cov / (K.sqrt(var_true) * K.sqrt(var_pred) + K.epsilon())

def gaussian_nll_loss(y_true, y_pred, mask=CPG_NAN):
    weights = _sample_weights(y_true, mask)
    mu, log_var = y_pred[:, 0:1], y_pred[:, 1:2]
    
    y_true, weights = K.reshape(y_true, (-1, 1)), K.reshape(weights, (-1, 1))
    variance = K.exp(log_var) + K.epsilon()
    log_likelihood = 0.5 * log_var + 0.5 * K.square(y_true - mu) / variance
    
    return K.sum(log_likelihood * weights) / (K.sum(weights) + K.epsilon())

def get(name):
    return get_from_module(name, globals())