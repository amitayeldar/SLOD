#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:17:35 2023

@author: ofirkarin
"""
import numpy as np
import itertools
import math

m = 5
d = 2
B = 2
grid_shape = [m]*d
step = 2*B/(m-1)
num_of_pts_in_grid = m**d
list_of_divided_intervals = []
max_int_grid = int((m-1)/2)

# for i in range(d):
#    list_of_divided_intervals.append(np.linspace(-B,B,m))
    
grid_int_coordinates = np.meshgrid(*[list(range(-max_int_grid,max_int_grid+1))]*(d), indexing="ij")

def grid_ind_to_int_coor(y):
    return [grid_int_coordinates[i][y] for i in range(d)]

def int_coor_to_grid_ind(y):
    return np.add(y,max_int_grid)

# def real_coor_to_grid_ind(y):
#     int_list = [];
#     for i in range(d):
#         int_list.append(int((y[i]+B)/step))
#     return int_list

pnt_int_coords = grid_ind_to_int_coor(np.unravel_index(range(num_of_pts_in_grid),grid_shape))


configs = []
configs_min_dist = []
config_iscentered = []
zero_ind = np.ravel_multi_index(int_coor_to_grid_ind([0]*d),grid_shape)

# configs with 1 pt
configs.append(np.arange(num_of_pts_in_grid))
configs_min_dist.append(np.empty(num_of_pts_in_grid,dtype=int)) # dist list for configs with 1 pt
config_iscentered.append(None)


# configs with 2 pts
configs.append(np.empty([math.comb(num_of_pts_in_grid, 2),2],dtype=int))
configs_min_dist.append(np.empty(math.comb(num_of_pts_in_grid, 2)))
pairs_to_idx = dict()
config_iscentered.append(np.empty(math.comb(num_of_pts_in_grid, 2),dtype=bool)) # risky
for config_idx,config in enumerate(itertools.combinations(range(num_of_pts_in_grid), 2)):
    configs[1][config_idx] = config
    configs_min_dist[1][config_idx] = step*max(np.abs([np.diff(pnt_int_coords[dim][list(config)]) for dim in range(d)]))
    pairs_to_idx[config] = config_idx
    if zero_ind in config:
        config_iscentered[1][config_idx] = True
    

# configs with more than 2 pts
for num_of_pts_in_config in range(3,3**d+1):
    num_of_configs = math.comb(num_of_pts_in_grid, num_of_pts_in_config)
    configs.append(np.empty([num_of_configs,num_of_pts_in_config],dtype=int))
    configs_min_dist.append(np.empty(num_of_configs))
    config_iscentered.append(np.empty(num_of_configs,dtype=bool)) # risky
    for config_idx,config in enumerate(itertools.combinations(range(num_of_pts_in_grid), num_of_pts_in_config)):
        configs[num_of_pts_in_config-1][config_idx] = config
        configs_min_dist[num_of_pts_in_config-1][config_idx] = step*min([configs_min_dist[1][pairs_to_idx[pair]] for pair in itertools.combinations(config, 2)])
        if zero_ind in config:
            config_iscentered[num_of_pts_in_config-1][config_idx] = True
            
            
def get_valid_configs(num_of_pts_in_config,dist,only_cent=False):
    valid_for_dist = [configs_min_dist[num_of_pts_in_config-1] >= dist]
    if only_cent:
        valid_configs = np.all([valid_for_dist,config_iscentered[num_of_pts_in_config]],0)
        return configs[num_of_pts_in_config-1][valid_configs]
    else:
        return configs[num_of_pts_in_config-1][valid_for_dist]

        

#for config_size in range(1,3**d+1):
 #   configs.append(list(itertools.combinations(range(m**d), config_size)))