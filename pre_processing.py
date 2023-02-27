#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:17:35 2023

@author: ofirkarin
"""
import numpy as np ;
m = 3 ;
d = 3 ;
B = 4 ;
grid_shape = [m for dim in range(d)]
step = 2*B/(m-1);
list_of_divided_intervals = [] ;
for i in range(d):
    list_of_divided_intervals.append(np.linspace(-B,B,m))

grid_coordinates = np.meshgrid(*list_of_divided_intervals, indexing="ij")

def int_coor_to_real_coor(y):
    return [grid_coordinates[i][y] for i in range(d)]

def real_to_int(y):
    int_list = [];
    for i in range(d):
        int_list.append(int((y[i]+B)/step))
    return int_list

pnt_coords = int_coor_to_real_coor(np.unravel_index(range(m**d),grid_shape))