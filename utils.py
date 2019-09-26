# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:13:30 2019

@author: victor
"""

import numpy as np
from matplotlib import pyplot as plt

def euclidian(a, b):
    """
    Calculate the euclidian distance between two points.
    
    Args:
        a: Point a
        b: Point b
    
    Returns:
        Return the euclidian distance between the two points.
    """
    
    x = np.asarray(a)
    y = np.asarray(b)
    
    return np.sqrt(np.sum(np.power(x - y, 2)))


def plot_plain_separator(model, x, grid_size=1000, grid_range=(-5, 15), save=None):
    x_lab = np.linspace(grid_range[0], grid_range[1], num=grid_size)
    y_lab = np.linspace(grid_range[0], grid_range[1], num=grid_size)
    x1, x2 = np.meshgrid(x_lab, y_lab)
    x_grid = np.transpose(np.vstack([x1.flatten(), x2.flatten()]))
    
    z = model.predict(x_grid)
    
    z = z.reshape([1000,1000])
    plt.contour(x1, x2, z, levels=[0], colors=('black',), linewidths=(2,))
    if save:
        plt.savefig(fr'C:\Users\victo\Documents\Works\Artigo_1_RNA\Imagens\{save}.png')
        
