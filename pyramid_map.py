# pyramid_map -- This program is designed to plot the n-cell partition for TCEs satisfying the assumptions of the 
# ...polygonal invariant curves paper by Cockram, N., Ashwin, P. and Rodrigues, A.
# Plots are restricted to the largest valid n-stepped pyramid, and this program can plot every n-stepped cap for the TCE. 
# Copyright (C) 2024 Noah Cockram

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#---------

#- The notes further down the code which are important to read.  Mainly at the beginning of the Plotting functions section,
#- ...but also at the end of the Plot_Cells function.

import numpy as np
import matplotlib.pyplot as plt
import pyTCE_v1 as tce
from copy import copy
from tqdm import tqdm


#--- Functions for determining whether a point (or which points in an array) are within an n-stepped pyramid.

def is_in_pyramid(x, l, rho, alpha, N):
    """This function determines whether a point x is within the N-stepped pyramid for the values of l = lambda, alpha = alpha and rho given."""

    eta = rho - l
    SIN = np.sin(alpha) #We use the cosines, sines and cotangent so much it is more efficient to precompute them.
    COS = np.cos(alpha)
    COT = COS/SIN
    
    #Here, we are testing if a point is within the highest n-stepped pyramid by observing that 
    #...the pyramid can be obtained as a union of trapezia stacked on top of each other.
    #...So we test for membership in this pyramid by testing for membership in each of the trapezia, 
    #...and if none of them return True, then we return False.
    if (x[1] <= (x[0]+rho)*COT) and (x[1] <= (l-x[0])*COT) and (x[1] <= N*eta*(1+COS)*COT):
        return True
    for k in range(N):
        if (x[1] <= (x[0] + eta*(N - k)*(1 + COS))/COT + N*eta*(1+COS)/SIN - (N - k)*eta*SIN) and (N*eta*(1 + COS)/SIN - (N - k)*eta*SIN < x[1] <= N*eta*(1 + COS)/SIN - (N - k - 1)*eta*SIN) and (x[1] <= -(x[0] + eta - eta*(N - k)*(1 + COS))/COT + N*eta*(1+COS)/SIN - (N - k)*eta*SIN):
            return True
    return False


def are_in_pyramid(points, l, rho, alpha):
    """This function computes the highest N for which the TCE with parameters l, rho and alpha admits an N-stepped cap.  The function then applies is_in_pyramid to an arrays of points, returning a boolean array.
    points is an array of 2D vectors."""
    
    eta = rho - l
    N = 1   #Here we compute the highest n-stepped pyramid that exists for the given values of lambda, eta, rho, and alpha.
    if eta > 0 and l > 2*eta*(1+np.cos(alpha)):
        for k in range(1,1000): #Unless you are working with values of lambda and rho exceptionally close together, you don't need to change the 1000 value.
            if l < 2*(k+1)*eta*(1+np.cos(alpha)):
                N = k
                break

    truth_array = np.zeros(points.shape[0], dtype=bool)
    for j, point in enumerate(points):
        truth_array[j] = is_in_pyramid(point, l, rho, alpha, N)
    
    return truth_array


#--- Plotting functions
#NOTE: There are several modes for plotting which must be changed manually through commenting and uncommenting.
#Some software (e.g. Visual Studio) allow quick (un)commenting via highlighting the lines you wish to (un)comment and pressing Ctrl+/


def Plot_Cells(ax, alpha, l, rho, num_iter, box_limits, resolution, **kwargs):
    """This function computes the n-cells, where n = num_iter, of the TCE with parameters alpha, l, rho restricted to the highest n-stepped pyramid for those parameters.
    We are assuming the TCE to abide by the constraints in the polygonal invariant curves paper.
    ax is an instance of the matplotlib Axes class,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    **kwargs are passed to the ax.scatter function."""

    #reformatting the variables into ones which can be parsed by the TCE function.
    translation_vec = np.array([[-rho,0],[l-rho,0],[l-rho,0],[l,0]])
    cone_angles = np.array([np.pi/2-alpha,alpha,alpha,np.pi/2-alpha])
    rotation_vec = np.array([0,alpha,-alpha,0])

    #Generating a uniform grid of points, then keeping only the points in the grid that are within the n-stepped pyramid.
    Full_Grid = tce.Generate_Lattice(box_limits, resolution)
    mask = are_in_pyramid(Full_Grid, l, rho, alpha)
    Grid = Full_Grid[mask]

    points = copy(Grid)
    
    initial_points = copy(points)
    points_size = points.shape[0]

    TCE_pts = np.zeros((points_size,2))

    for i in range(points_size):
        TCE_pts[i], _ = tce.TCE(points[i], cone_angles, rotation_vec, translation_vec)

    count = np.zeros((points.shape[0],4))
    
    for _ in tqdm(range(1,num_iter+1)): #We iterate the TCE the desired number of times, recording the cone we land in to determine the colouring of the point.
        for i in range(points_size):
            points[i], j = tce.TCE(points[i], cone_angles, rotation_vec, translation_vec)
                
            count[i,j] += 1 #Here we count the number of times a point visits each cone during its orbit.
    
    colour_palette = np.array([[0.2,0.2,0.2],[1,0,0],[0,1,0],[0,0,1]])  #These are the colours we assign to each cone.  Feel free to change them around.
    colour_stuff = np.array([[count[i,j]*colour_palette[j] for j in range(4)] for i in range(points_size)])
    colour_values = np.sum(colour_stuff, axis=1)/num_iter

    #- The first plot displays the n-cells in their original positions
    ax.scatter(initial_points[:,0],initial_points[:,1],c=colour_values,**kwargs)
    #- The second plot displays the n-cells after one iterate of the TCE
    # ax.scatter(TCE_pts[:,0],TCE_pts[:,1],c=colour_values,**kwargs)
    #- The third plot displays the n-cells after n iterations of the TCE.
    # ax.scatter(points[:,0],points[:,1],c=colour_values,**kwargs)


def Plot_n_step_curve(ax, alpha, l, rho, n, boundary=False):
    """This function plots the n-stepped cap (the polygonal invariant curve with n-steps) for the TCE with parameters alpha, l and rho, 
    assuming that the TCE abides by the constraints in the polygonal invariant curves paper.
    ax is an instance of the matplotlib Axes class,
    boundary is a boolean which asks whether you want to plot the rest of the boundary of the n-stepped pyramid as well as the curve."""

    eta = rho - l
    SIN = np.sin(alpha)
    COS = np.cos(alpha)
    COT = COS/SIN
    ell_n = l - 2*n*eta*(1 + COS)
    
    points = np.zeros((4*n+4,2)) #Storing the vertices of n-stepped cap in this array
    points[0] = [n*eta*(1 + COS) - rho, n*eta*(1 + COS)*COT]
    points[1] = points[0] + np.array([ell_n, 0])

    for k in range(1,4*n+2): #This generates the vertices of the curve iteratively
        if k % 2 == 1:
            points[k+1] = points[k] + np.array([eta, 0])
        elif k % 2 == 0 and 1 < k <= 2*n+1:
            points[k+1] = points[k] + np.array([eta*COS, eta*SIN])
        else:
            points[k+1] = points[k] + np.array([eta*COS, -eta*SIN])
    
    points[-1] = points[-2] + np.array([ell_n, 0])

    ax.plot(points[:3,0], points[:3,1], 'r') #These four lines plot the curve, segmented in colour according to the cone each segment is in.
    ax.plot(points[2:(2*n+3),0], points[2:(2*n+3),1], 'g')
    ax.plot(points[(2*n+2):(4*n+3),0], points[(2*n+2):(4*n+3),1], 'b')
    ax.plot(points[(4*n+2):(4*n+4),0], points[(4*n+2):(4*n+4),1], 'm')
    
    if boundary:    #If boundary is set to True, then the exterior boundary of the n-stepped pyramid will also be plotted.
        ax.plot([-rho, points[0,0]], [0, points[0,1]],'k')
        ax.plot([0, points[2,0]], [0, points[2,1]], 'k')
        ax.plot([0, points[2*n+2,0]], [0, points[2*n+2,1]], 'k')
        ax.plot([0, points[4*n+2,0]], [0, points[4*n+2,1]], 'k')
        ax.plot([l, points[-1,0]], [0, points[-1,1]], 'k')

    return None


#--- Example initialisation of variables

# alpha = np.pi/6
# l = 1 - 1/(7+6*np.cos(alpha))+0.005
# rho = 1

# box_limits = [-rho,l,0,1]

# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
# ax.set_aspect(1)
# ax.set_xlim(box_limits[0],box_limits[1])
# ax.set_ylim(box_limits[2],box_limits[3])


#--- Example execution of plotting functions 
#NOTE: the Plot_n_step_curve function must be executed once for each valid value of n.
#Additionally, be mindful about changing the maximum value in the range of the for loop when you change the value of alpha, l and rho,
#...because the function will attempt to plot a curve even when one doesn't exist.  This will give you strange results!

# Plot_Cells(ax, alpha, l, rho, 1, box_limits, 1e-3, s=0.05, marker='o')

# for n in range(1,4):
#     Plot_n_step_curve(ax, alpha, l, rho, n, top=False)

# plt.show()