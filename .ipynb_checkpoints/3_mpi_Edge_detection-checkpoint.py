#!/usr/bin/env python
# coding: utf-8

# # Code to detect edges in an image
# ### Splits image into pieces
# #### Feb 28, 2019

# Import modules

import sys
import datetime
import subprocess as sp
import argparse
import pandas as pd
import numpy as np
import time
from IPython.display import display

import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage import feature


# mpi part of the code
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# ## Modules
# Generate noisy image of a square
def f_create_image(size):
 '''
 Create image
 Steps: make blank image, add brightness in a square and rotate it.
 '''
 
 im = np.zeros((size, size))
 factor=int(size/4)
#     print(factor)

 im[factor:-factor, factor:-factor] = 1
 im = ndi.rotate(im, 45, mode='constant',reshape=False)
 im = ndi.gaussian_filter(im, 4)
 im += 0.2 * np.random.random(im.shape)
#     print(im.shape)

 return im

# Detect edges
def f_detect_edge(im,sigma=3):
 # Compute the Canny filter for two values of sigma
 edges1 = feature.canny(im)
 edges2 = feature.canny(im, sigma=sigma)
 
 return edges1,edges2


# Plot functions
def f_plot_single_image(im):
 
 plt.figure()
 plt.imshow(im, cmap=plt.cm.gray)
 plt.axis('off')


def f_plot_all(im,edges1,edges2):
 '''
 Function to plot the 3 images: original, simple detection, best detection.
 Reads in 3 arrays
 '''
 fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                     sharex=True, sharey=True)

 ax1.imshow(im, cmap=plt.cm.gray)
 ax1.axis('off')
 ax1.set_title('noisy image', fontsize=20)
 
 ax2.imshow(edges1, cmap=plt.cm.gray)
 ax2.axis('off')
 ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

 ax3.imshow(edges2, cmap=plt.cm.gray)
 ax3.axis('off')
 ax3.set_title('Canny filter, $\sigma=2$', fontsize=20)

 fig.tight_layout()
#     plt.show()


# ## Edge detection using image split

if __name__=='__main__':
    num_pieces=4
    length=256
    ratio=int(length/num_pieces)

    print("Rank",rank)
    t0=time.time()

    data=np.empty((64,256),dtype=np.float64)


    if rank==0:

        im_combined_lst_1,im_combined_lst_2=[],[]
        
        # Create and split data
        im=f_create_image(length)
        split_lst=np.split(im,num_pieces)
        
        # Send pieces to other cores
        for i in [1,2,3]: comm.Send(split_lst[i], dest=i, tag=i+12)

        # Perform one operation on rank=0
        t1=time.time()
        edges1,edges2=f_detect_edge(split_lst[0],sigma=2)
        t2=time.time()
        time.sleep(2)
        t3=time.time()
        print("Time for loop",rank,":",t3-t1)   
        im_combined_lst_1.append(edges1)
        im_combined_lst_2.append(edges2)

        # Receive jobs back from other cores
        for i in [1,2,3]:
            ed1=np.empty((64,256),dtype=np.bool)
            ed2=np.empty((64,256),dtype=np.bool)
            comm.Recv(ed1,source=i,tag=16+2*(i-1))
            comm.Recv(ed2,source=i,tag=16+2*(i-1)+1)
            im_combined_lst_1.append(ed1)
            im_combined_lst_2.append(ed2)
        
        assert not np.array_equal(im_combined_lst_2[2],im_combined_lst_2[3]), "the two lists should not be equal"

        # Combine data back in rank 0
        im_combined1=np.vstack(im_combined_lst_1)
        im_combined2=np.vstack(im_combined_lst_2)

        f_plot_all(im,im_combined1,im_combined2)
        plt.savefig('edge_detection_split.pdf')

        t_end=time.time()

        print("Total time:",t_end-t0)

    if rank in [1,2,3]:
    #     data=np.empty((ratio,ratio), dtype=np.float64)
        i=rank
        comm.Recv(data, source=0,tag=12+i)
        t1=time.time()
        edges1,edges2=f_detect_edge(data,sigma=2)
        t2=time.time()
        time.sleep(2)
        t3=time.time()
        print("Time for one loop",t3-t1)
        comm.Send(edges1, dest=0, tag=16+2*(i-1))
        comm.Send(edges2, dest=0, tag=16+2*(i-1)+1)
    
    ### Done ###