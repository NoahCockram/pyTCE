# trapezium_map -- This program is designed to compute and plot the n-cell partition for the first return map of
#...a TCE to the trapezium defined by the intersection of an n-stepped pyramid with the last cone.  It can also generate first return orbits.
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

#- The notes further down the code are important to read.  Mainly the comments at the Plotting functions section,
#- ...and at the end of the Plot_Cells function.

import numpy as np
import matplotlib.pyplot as plt
import pyTCE_v1 as tce
import matplotlib.cm as cm
from copy import copy
from tqdm import tqdm


#--- Computing orbits

def is_in_trapezium(x, cone_angles, translation_vec, N):
    """This function determines whether the point x lies within the trapezium defined by the intersection between the N-stepped pyramid and the last cone for the TCE.
    x is a 2D vector in the plane,
    cone_angles and translation_vec are parameters for the TCE,
    N is the number of steps of the pyramid that the trapezium is coming from."""
    if x[1] < -x[0]/np.tan(cone_angles[1]) and x[1] <= (x[0] - translation_vec[0,0])/np.tan(cone_angles[1]) and x[1] <= N*(1+np.cos(cone_angles[1]))*abs(translation_vec[1,0])/np.tan(cone_angles[1]):
        return True
    else:
        return False

def fr_to_trapezium(x, cone_angles, rotation_vec, translation_vec, N, max_iter=1000):
    """This function computes the first return map of the TCE with parameters cone_angles, rotation_vec and translation_vec to the trapezium.
    N is the number of steps of the pyramid that the trapezium is coming from,
    max_iter is the maximum number of iterations of the TCE that the first return map allows before halting and returning a value."""
    Cs = 0

    for n in range(max_iter):   #At each run of the for loop, we iterate the point under the TCE
        x, c = tce.TCE(x, cone_angles, rotation_vec, translation_vec)  
        Cs += c #The cones that x lands in will determine the colour of the point.

        if is_in_trapezium(x, cone_angles, translation_vec, N):  #This checks whether the point has re-entered the trapezium
            return x, n+1+Cs   #If after n iterations, we return to the middle cone, we stop and return the point y = TCE^n(x) and the time n.

    return np.array([0, -1]), -1  #It is normally impossible for these values to be a valid output, so if the function times out, this will indicate it.


#--- Plotting functions
#NOTE: There are multiple plotting modes for the Plot_Cells function which must be switched (commented/uncommented) manually.

def Plot_Cells(ax, cone_angles, rotation_vec, translation_vec, num_iter, box_limits, resolution, colour_map, max_iter=1000, **kwargs):
    """This function computes and plots the n-cells, where n = num_iter of the first return map to the trapezium defined as the intersection between the n-stepped pyramid and the last cone in the TCE.
    ax is an instance of the matplotlib Axes class,
    cone_angles, rotation_vec and translation_vec are parameters of the TCE,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    colour_map is a colourmap chosen from the matplotlib library,
    max_iter is the maximum number of iterations of the TCE that the first return map allows before halting and returning a value,
    **kwargs are passed to the ax.scatter function."""

    l = -translation_vec[0,0]
    eta = -translation_vec[1,0]
    
    N = 1   #Here we compute the highest n-stepped pyramid that exists for the given values of lambda, eta, rho, and alpha.
    if eta > 0 and l > 2*eta*(1+np.cos(cone_angles[1])):
        for k in range(1,1000): #Unless you are working with values of lambda and rho exceptionally close together, you don't need to change the 1000 value.
            if l < 2*(k+1)*eta*(1+np.cos(cone_angles[1])):
                N = k
                break

    XY = tce.Generate_Lattice(box_limits, resolution)   #See the pyTCE file for information on this function.
    #...Briefly, it generates a uniform grid of points separated by distance equal to resolution.

    points = np.empty((0,2))

    for pt in tqdm(XY): #A point in the grid is added to the points array only if it lies within the pyramid.
        if is_in_trapezium(pt, cone_angles, translation_vec, N):
            points = np.append(points,[pt], axis=0)
    
    points_size = points.shape[0]

    initial_points = copy(points)
    points_size = points.shape[0]

    first_return_times = np.zeros(points_size)

    for _ in tqdm(range(1,num_iter+1)): #We compute the iterates of the first return map for each point in our grid.
        for i in range(points_size):
            points[i], fht = fr_to_trapezium(points[i], cone_angles, rotation_vec, translation_vec, N, max_iter)
            
            first_return_times[i] += fht
    
    colour_values = np.mod(2*first_return_times/(max(first_return_times)+2),1)  #We determine the colour values to go with each point.

    #- The first plot displays the n-cells in the original positions
    ax.scatter(initial_points[:,0],initial_points[:,1],c=colour_map(colour_values),**kwargs)
    #- The second plot displays the n-cells after n iterates of the first return map.
    # ax.scatter(points[:,0],points[:,1],c=colour_map(colour_values),**kwargs)


#--- Example of initialisation of variables

# alpha = np.pi/5
# cone_angles = np.array([np.pi/2 - alpha, alpha, alpha, np.pi/2 - alpha])
# permutation = np.array([0,2,1,3])
# rotation_vec = tce.Rotation_Angles(cone_angles, permutation)

# l = 1 - 1/(7+6*np.cos(alpha))+0.005
# eta = 1 - l
# rho = 1
# translation_vec = np.zeros((cone_angles.shape[0],2))
# translation_vec[1:-1,0] = -eta
# translation_vec[0,0] = -rho
# translation_vec[-1,0] = l

# box_limits = [-rho,0,0,(rho/2)/np.tan(alpha)]   #This setting for the boundary box always shows the entire trapezium.
# num_points = 500

# colour_map = cm.get_cmap('viridis') #Check the available colourmaps for a list of choices.  My favourite for Plot_TCE is 'Blues'.

# translation_vec = np.array([[-rho,0],[-eta,0],[-eta,0],[l,0]])
# box_limits = [-rho,0,0,(rho/2)/np.tan(alpha)]
# N = 1000

# fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(8,6)) #Initialising axes.  Feel free to change figsize to suit your screen.
# ax.set_aspect(1)


# #--- Example execution of the plot functions

# #Plotting the n-cell partition for the first return map
# Plot_Cells(ax, cone_angles, rotation_vec, translation_vec, 1, box_limits, 1.5e-3, colour_map, s=0.15, marker='o')

# N = 1   #Here we compute the highest n-stepped pyramid that exists for the given values of lambda, eta, rho, and alpha.
#         #...then we plot the part of each k-stepped cap passing through the trapezium for reference.
# if eta > 0 and l > 2*eta*(1+np.cos(cone_angles[1])):
#     for k in range(1,1000): #Unless you are working with values of lambda and rho exceptionally close together, you don't need to change the 1000 value.
#         #Plotting the part of the k-stepped cap which passes through the trapezium.
#         ax.plot([k*(1+np.cos(alpha))*eta - rho, -k*(1+np.cos(alpha))*eta], [k*(1+np.cos(alpha))/np.tan(alpha)*eta, k*(1+np.cos(alpha))/np.tan(alpha)*eta], 'r')
        
#         if l < 2*(k+1)*eta*(1+np.cos(cone_angles[1])):
#             N = k
#             break

# ax.set_xlim(box_limits[0], box_limits[1])
# ax.set_ylim(box_limits[2], box_limits[3])
# plt.show()