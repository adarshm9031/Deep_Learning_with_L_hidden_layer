# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:19:35 2019

@author: adarshm
"""
import afunctions as af
import linear_backward as lb
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = af.relu_backward(dA, activation_cache)
        dA_prev, dW, db = lb.linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = af.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = lb.linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db