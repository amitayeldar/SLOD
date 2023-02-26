#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
"""
Created on Mon Feb 20 16:50:30 2023

@author: ofirkarin
"""

def calc_lower_upper_bound(basis_functions_lst,config,grid_coordinates,int_to_cord,dim,step,num_of_fun):
    
    #%% Evaluate basis_functions on the grid
    length_of_grid = grid_coordinates[0].shape[0]
    basis_functions_eval = []
    for fun in basis_functions_lst:
        basis_functions_eval.append(function_eval(fun,grid_coordinates))
    #%% Computes integral for B computation
    config_one_integral = {}
    for con in configs[0]:
        cnt_k = 0
        for psi_k in basis_functions_eval:
            for psi_j_translated in basis_functions_eval:
                for shift_parameter in con:
                    ax = 0
                    psi_j_translated = roll_zeropad(psi_j_translated, shift_parameter,ax )
                    ax = ax + 1
                int_tmp_j = psi_k*psi_j_translated
                for d in range(dim):
                    int_tmp_kj = np.trapz(int_tmp,dx = step) 
            config_one_integral[(k,j,)+con] = int_tmp_kj
            cnt_k +=1
     #%% lambda computation       
    B_quad = np.zeros((num_of_fun,num_of_fun))
    lambda_dic_quad = {}
    for con in configs[0]:
        for k in range(num_of_fun):
            for s in range(k,num_of_fun):
                for j in range(num_of_fun):
                    B_quad[k,s] += config_one_integral[(k,j,)+con]*config_one_integral[(s,j,)+con] 
        B_quad = B_quad + np.transpose(B_quad)
        w = LA.eigh(B_quad)
        lambda_dic_quad[con] = (w[0],w[-1])
    
    B_bil = np.zeros((num_of_fun,num_of_fun))
    lambda_dic_bil = {}
    for con in configs[1]:
        for k in range(num_of_fun):
            for s in range(k,num_of_fun):
                for j in range(num_of_fun):
                    B_bil[k,s] += config_one_integral[(k,j,)+con]*config_one_integral[(s,j,)+con] 
        B_bil = B_bil + np.transpose(B_bil)
        w = LA.eigh(B_bil)
        lambda_dic_bil[con] = (max(abs(w[0]),abs(w[-1])))
                              
    #%% for each configuration compute lower_upper_bounds
    lower_bound = {}
    upper_bound = {}
    for con_len in config:
        for cons in con_len:
            for con in cons:
                S_lower_vec = ()
                S_upper_vec = ()
                for con_i in con:
                    con_removed = con.remove(con_i)
                    con_couples = [(a, b) for idx, a in enumerate(con_removed) for b in con_removed[idx + 1:]]
                    lambda_min,lambda_max = lambda_dic_quad[con_i]
                    S_lower = lambda_min
                    S_upper = lambda_max
                    for cup in con_couples:
                       S_lower = S_lower - 2*lambda_dic_bil[cup]
                       S_upper = S_upper + 2*lambda_dic_bil[cup]
                    S_lower_vec +=(S_lower,)   
                    S_upper_vec +=(S_upper,)
            lower_bound[con] = S_lower_vec
            upper_bound[con] = S_upper_vec
    
                    
                 

def function_eval(fun,grid_coordinates,int_to_cord):
    """
    This function sample a given function "fun" on the grid_coorginates
    The output is an array of size grid_coorginates[0].
    
    """
    fun_on_grid = np.zeros(grid_coordinates[0].shape)
    it = np.nditer(grid_coordinates[0],flags=['multi_index'])
    with it:
            while not it.finished:
                fun_on_grid(it.multi_index) = fun(int_to_cord(it.multi_index))
                it.iternext()
    return fun_on_grid 

def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res            
    
                                                  
                                                 

                
        
        

