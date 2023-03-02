#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:17:35 2023

@author: ofirkarin
"""
import numpy as np
import itertools
import math
import sys

m = 65 # odd number
d = 2
B = 2
grid_shape = [m]*d
step = 2*B/(m-1)
int_B = int(B/step)
num_of_pts_in_grid = m**d
max_int_grid = int((m-1)/2)

# convert an array of integer positions with shape = (N,dim) to array of grid indecies with shape (N,dim)
def int_pos_to_grid_idx(int_pos,max_int_grid):
    return np.add(int_pos,max_int_grid)

# convert an array of grid indecies with shape = (N,dim) to array of integer positions with shape (N,dim)
def grid_idx_to_int_pos(int_pos,max_int_grid):
    return np.subtract(int_pos,max_int_grid)

# return an array of integer positions for all points in the grid, shape (num_of_pnts_in_grid,dim)
def get_pnts_int_pos(dim,max_int_grid):
    m = max_int_grid*2+1
    grid_shape = [m]*dim
    num_of_pts_in_grid = m**dim
    return np.subtract(np.unravel_index(np.arange(num_of_pts_in_grid),grid_shape),max_int_grid).T

# use this to convert pnt idx to integer coordinates
pnt_idx_to_int_pos = get_pnts_int_pos(d,max_int_grid)

def get_configs(num_of_pnts_in_config,prev_configs,min_dist_int):
    # initialize config list with shape (num_of_possible_configs,num_of_pnts_in_config)
    configs = dict()
    grid_idx_to_pnt_idx = np.arange(num_of_pts_in_grid).reshape(grid_shape)
    one_percent_of_loops = max([int(len(prev_configs)/100),1]) # for printing
    for small_config_idx,small_config in enumerate(prev_configs): # go over all small configs with shape (num_of_pnts_in_config-1,)
        
        grid_idx_to_add = np.ones(grid_shape,dtype=bool) # set a grid of pnts to add to the config, initialize with all True
        grid_idx_to_add[np.unravel_index(np.arange(small_config[-1]+1),grid_shape)] = False # set all grid idx for points with pnt idx smaller than the largest pnt idx of the small config to False, to prevent duplicates
        
        for pnt_idx in small_config: # go over all pnts in the small config
            pnt_grid_idx = np.unravel_index(pnt_idx,grid_shape) # get its grid idx
            box_idx = [np.arange(max([pnt_grid_idx[dim]-min_dist_int,0]),min([pnt_grid_idx[dim]+min_dist_int,2*max_int_grid+1])) for dim in range(d)] # get all grid idx for the box of radius integer B
            grid_idx_to_add[np.ix_(*box_idx)] = False # set all the grid idx in the box as false
    
        pnt_idx_to_add = grid_idx_to_pnt_idx[grid_idx_to_add]
        num_of_pnts_to_add = len(pnt_idx_to_add)
        configs[small_config_idx] = pnt_idx_to_add

        
        if small_config_idx % one_percent_of_loops == 0: # print after 1 precent of loops are done
            print("-",end="")
    
    print("\n Done with dictionary!")

    # set final output array
    num_of_valid_configs = sum(list(map(len,configs.values())))
    final_configs = np.empty([num_of_valid_configs,num_of_pnts_in_config],dtype=int)
    config_idx = 0
    for small_config_idx,pnt_idx_to_add in configs.items():
        num_of_pnts_to_add = len(pnt_idx_to_add)
        final_configs[config_idx:config_idx+num_of_pnts_to_add] = np.concatenate([np.repeat(prev_configs[small_config_idx][None,:],num_of_pnts_to_add,0),pnt_idx_to_add[:,None]],1)
        config_idx += num_of_pnts_to_add
        if small_config_idx % one_percent_of_loops == 0: # print after 1 precent of loops are done
            print("-",end="")
        
    
    print("\n Done! configs with {} points take {} GB of memory".format(num_of_pnts_in_config,final_configs.nbytes/(10**9)))
    return final_configs

configs1 = np.arange(num_of_pts_in_grid)[:,None]
        