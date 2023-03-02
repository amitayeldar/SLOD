#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:59:11 2023

@author: amitayeldar
"""

from functools import partial
import numpy as np

def fb_basis_1d(num_of_fun): 
    basis_lst = []
    for idx in range(1,num_of_fun+1):
        f = partial(modified_fb_1d,idx)
        basis_lst+=[f]
    return basis_lst
        
def modified_fb_1d(idx,x):
    res = (1/(np.sqrt(np.pi)))*np.cos(idx*x)
    return res
    
    

