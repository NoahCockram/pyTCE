# pyTCE -- This program is designed to compute and plot the dynamical orbits of Translated Cone Exchange Transformations
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

#- There are notes further down the code which are important to read.  Mainly at the beginning of the sections marked by '#---'
#- ...but also try to read the comments at the end of the Plot_TCE, Plot_TCE_Cells and Plot_FR_Cells functions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import copy
from tqdm import tqdm

#--- The Cone class is made for the easy parametrisation and plotting of cones in the plane.  
#    I use it primarily for demonstrations and figures, but it has no use in any further code.

class Cone:
    def __init__(self, pos, width, angle):  #I defined this class in case I wanted to make demonstrations/figures to showcase what the cones do under the action of a TCE
        self.pos = pos  #Any cone is defined by three parameters, the position of its vertex, and the angles of the two boundary rays originating from it from the horizontal.
        self.width = width  #The angle of the cone at its vertex, measured in the anti-clockwise direction.
        self.angle = angle  #The anti-clockwise angle between the horizontal and one of the boundary lines.
        #Note that the first boundary line is determined by self.angle, and the second one is determined from the first using self.width
    
    def vertices(self, radius): #This function generates the vertices of the triangle created by cutting off the boundary lines at a radius from the vertex
        """The vertices are ordered as follows: the origin of the cone, 
        the vertex with the smaller angle relative to the horizontal line going through the origin, then the vertex with the larger angle."""
        v1 = self.pos
        v2 = self.pos + radius*np.array([np.cos(self.angle), np.sin(self.angle)])
        v3 = self.pos + radius*np.array([np.cos(self.angle + self.width), np.sin(self.angle + self.width)])
        return v1, v2, v3
    
    def rotate(self, rotation_angle):   #This rotates the entire cone.
        self.angle += rotation_angle
    
    def translate(self, translation_vector):   #This translates the entire cone.
        self.pos += translation_vector
    
    def hollow_plot(self, ax, radius, **kwargs):    #This plots the boundary of the cone, consisting of the vertex and its two boundary rays (up to the radius)
        v1, v2, v3 = self.vertices(radius)
        ax.plot([v2[0],v1[0],v3[0]], [v2[1],v1[1],v3[1]],**kwargs)

    def filled_plot(self, ax, radius, **kwargs):    #This function plots a filled-in triangle with vertices determined by self.vertices(radius)
        v1, v2, v3 = self.vertices(radius)
        ax.fill([v2[0],v1[0],v3[0]], [v2[1],v1[1],v3[1]],**kwargs)


#--- TCE computation     

def Rotation_Angles(cone_angles, permutation):
    """This function computes the rotation angles for our piecewise isometry. 
    cone_angles is a numpy array containing the angle widths of each cone, in anti-clockwise order (right-to-left),
    permutation is a list of the integers 0, ... , d-1 representing the new ordering of the cones after they are exchanged,
    where d is the number of cones in the partition."""

    d = np.size(cone_angles)
    rotation_vec = np.zeros(d)
    
    for j in range(d): #Loop to find rotation angle for each cone
        permuted_angle_sum = 0
        for k in range(d): #The sum of the angles of cones that appear before the jth cone after permutation
            if permutation[k] < permutation[j]:
                permuted_angle_sum += cone_angles[k]
        
        angle_sum = 0
        for k in range(j): #The sum of the angles of cones that appear before the jth cone before permutation
            angle_sum += cone_angles[k]
        
        rotation_vec[j] = permuted_angle_sum - angle_sum #The difference is the rotation angle
    
    return rotation_vec


def TCE(x, cone_angles, rotation_vec, translation_vec):
    """This function performs one iteration of a TCE.
    x is a numpy array representing a vector in the plane,
    cone_angles is a numpy array representing the angle widths of the cones in anti-clockwise order,
    rotation_vec is a numpy array containing the angles each cone rotates under the TCE, again in anti-clockwise order,
    translation_vec is a numpy array containing the amount each cone shifts under the TCE, again in anti-clockwise order.
    
    Note: translation_vec has the same shape as rotation_vec and cone_angles, i.e. the translation of every cone is listed individually,
    despite the fact that cones 1, ... , d-2 are grouped together in the mathematical definition.
    """

    if x[1] < 0: #If x is not in the closed upper half plane, then this function doesn't work.
        print(x)
        raise Exception('The point x must lie in the closed upper half plane.')
    
    x_argument = np.arctan2(x[1],x[0]) #The argument of the point x.

    current_cone = 0    #This is the argument of the right boundary line of the cone j, starting from cone 0.
    for j in range(np.size(cone_angles)):
        current_cone += cone_angles[j]

        if x_argument <= current_cone:  #The function compares the angles of the cone boundary lines,
            #...until the first time where a boundary line is "on the other side" of x.  Then we know which cone x is in.
            J = j
            rotation_matrix = np.array([[np.cos(rotation_vec[j]),-np.sin(rotation_vec[j])], #2D rotation matrix
                                        [np.sin(rotation_vec[j]), np.cos(rotation_vec[j])]])
            # return np.dot(rotation_matrix,x) + translation_vec[j], j #We return the local isometry applied to the point x.
            new_point = np.dot(rotation_matrix,x)
            break
    
    newx_argument = np.arctan2(new_point[1],new_point[0])
    current_cone = 0
    for j in range(np.size(cone_angles)):
        current_cone += cone_angles[j]

        if newx_argument <= current_cone:
            return new_point + translation_vec[j], J


def First_Return_Pc(x, cone_angles, rotation_vec, translation_vec, max_iter=1000):
    """This function computes one iterate of the first return map of the TCE to the middle cone Pc.
    x is a numpy array representing a vector in the plane,
    cone_angles is a numpy array representing the angle widths of the cones in anti-clockwise order,
    rotation_vec is a numpy array containing the angles each cone rotates under the TCE, again in anti-clockwise order,
    translation_vec is a numpy array containing the amount each cone shifts under the TCE, again in anti-clockwise order,
    max_iter is the maximum number of iterations of the TCE that the function allows before halting and returning a value."""

    y = copy(x) #Note that this function does work for points outside of the middle cone, 
                #...simply returning the first point in the trajectory where the point enters the middle cone.

    angle_sums = 0

    for n in range(max_iter):
        if all(x == np.array([0, -1])):
            break
        y, _ = TCE(y, cone_angles, rotation_vec, translation_vec)  #At each run of the for loop, we iterate the point under the TCE

        y_argument = np.arctan2(y[1],y[0])  #Calulating the argument of the point.

        if y_argument >= cone_angles[0] and y_argument <= np.pi - cone_angles[-1]:  #This checks whether the point has (re-)entered the middle cone.
            angle_sums = np.mod(angle_sums,1)
            return y, n+1 
            #If after n iterations, we return to the middle cone, we stop and return the point y = TCE^n(x) and the time n. 
            #Note: the index n starts from 0 to N-1, so we add 1.

    return np.array([0, -1]), -1, 0     #It is normally impossible for these values to be a valid output, 
                                        #...so if the function times out, this will be the indication of that.


#--- Functions for generating points

def Generate_Random_Points(box_limits, num_points):
    """This function generates an array of vectors uniformly distributed within a bounding box.
    box_limits is a list of the form [x_min, x_max, y_min, y_max],
    num_points is the number of points being generated."""

    x_min, x_max, y_min, y_max = box_limits
    choices = np.random.rand(2, num_points) #generating uniformly distributed 2D vectors within the box [0,1)x[0,1)
    points = np.multiply(choices, np.array([x_max - x_min, y_max - y_min])[:,None]).T + np.array([x_min, y_min]) #Deforming the randomly generated points into the bounding box
    return points


def Generate_Lattice(box_limits, resolution):
    """This function generates a square grid of points within a bounding box.
    box_limits is a list of the form [x_min, x_max, y_min, y_max],
    resolution is the distance between adjacent points in the grid."""

    xmin, xmax, ymin, ymax = box_limits
    xs = np.arange(xmin, xmax, resolution)
    ys = np.arange(ymin, ymax, resolution)

    XX, YY = np.meshgrid(xs,ys) #This creates a grid of points, where XX contains the x-coordinates and YY contains the y-coordinates.  I want to put each x-coordinate with each y-coordinate

    XX = XX.flatten()   #I flatten them to make them easier to plot.
    YY = YY.flatten()

    numel = XX.shape[0]  #This is the size of the array of (x,y)-coordinates.

    XY = np.zeros((numel,2)) #Now I create an array to store the coordinates as pairs instead of being in separate arrays. e.g. XY[0] = [XX[0],YY[0]].
    XY[:numel,0] = XX
    XY[:numel,1] = YY

    return XY


#--- Plotting functions (NOTE: Plots may not automatically show after the execution of these functions unless you execute plt.show() afterwards.)

def Plot_TCE(ax, cone_angles, rotation_vec, translation_vec, box_limits, num_points, num_iter, colour_map, **kwargs):
    """This function plots a graph of the orbits of uniformly distributed points under the TCE with parameters cone_angles, rotation_vec and translation_vec.
    Each point and its iterates are given an arbitrary distinct colour.
    ax is an instance of the matplotlib Axes class,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    num_points is the number of points being iterated,
    num_iter is the number of iterates being computed for each point,
    colour_map is a colourmap chosen from the matplotlib library,
    **Kwargs are passed to the ax.scatter function."""
    
    points = Generate_Random_Points(box_limits, num_points) #We rescale the elements of each vector so that the resulting vectors are within the box [xmin, xmax] x [ymin, ymax]

    orbit = np.zeros((num_points*num_iter, 2))  #Preset the array which will store the trajectories of the points we have chosen.  The shape makes it easier to plot.
    orbit[:num_points,:] = points  #We start the array with the initial points (time t=0).

    plot_colours = np.zeros(num_points*num_iter) 
    plot_colours[:num_points] = list(range(1,num_points+1))

    for n in tqdm(range(1,num_iter)):
        for j in range(num_points):
            orbit[n*num_points+j], _  = TCE(orbit[(n-1)*num_points+j], cone_angles, rotation_vec, translation_vec)  #We calculate the next point in the trajectory.
            plot_colours[n*num_points + j] = j+1  #Ensuring that each trajectory has the same colour value as its initial point, and distinct trajetories have distinct colours.

    plot_colours = plot_colours/(num_points+2)
    
    ax.scatter(orbit[600*num_points:,0],orbit[600*num_points:,1],c=colour_map(plot_colours[600*num_points:]),**kwargs)
    #Some orbits start with a transient part where they drift around before entering a periodic island, 
    #...but this makes the final image look grainier/noisier, so we omit the first 600 iterates of each trajectories.


def Plot_TCE_Cells(ax, cone_angles, rotation_vec, translation_vec, num_iter, box_limits, resolution, **kwargs):
    """This function plots the n-cells for the TCE with parameters from cone_angles, rotation_vec, translation_vec, where n = num_iter.
    ax is an instance of the matplotlib Axes class,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    **kwargs are passed to the ax.scatter function."""

    d = np.size(cone_angles)

    XY = Generate_Lattice(box_limits, resolution)
    numel = XY.shape[0]

    colour_choices = [[1,0,0],[0,1,0],[0,0,1]] #These are the colour choices assigned to the first, middle and last cones
    
    points = copy(XY)
    colour_values = np.zeros((numel,3))

    for _ in tqdm(range(1,num_iter+1)):
        for i in range(numel):
            XY[i], j = TCE(XY[i], cone_angles, rotation_vec, translation_vec)
            if j == 0: #The cones the first, middle and last cones are each assigned a different colour from colour_choices.
                jstar=0
            elif j == d-1:
                jstar=2
            else:
                jstar=1
            colour_values[i] += colour_choices[jstar] #The colour of an orbit is determined by the average colour of cones the point visits.

    colour_values *= 1/num_iter

    ax.scatter(points[:,0],points[:,1],c=colour_values,**kwargs)



def Plot_FR_Cells(ax, cone_angles, rotation_vec, translation_vec, num_iter, box_limits, resolution, colour_map, max_iter=1000, **kwargs):
    """This function plots the n-cells, where n = num_iter, for the first return map of the TCE with parameters cone_angles, rotation_vec, translation_vec, to the middle cone.
    The colouring of a point is determined by the sequence of first return times to the middle cone in its orbit up to num_iter.
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    colour_map is a colourmap chosen from the matplotlib library,
    max_iter is the maximum number of iterations of the TCE that the first return map allows before halting and returning a value,
    **kwargs are passed to the ax.scatter function."""

    XY = Generate_Lattice(box_limits, resolution)

    points = np.empty((0,2))

    for pt in tqdm(XY): #A point in the grid is added to the points array only if it is within the middle cone.
        argument = np.arctan2(pt[1],pt[0])

        if argument >= cone_angles[0] and argument <= np.pi - cone_angles[-1]:
            points = np.append(points,[pt], axis=0)
    
    points_size = points.shape[0]
    
    initial_points = copy(points)   #This array stores the original state of the points for the purpose of plotting.
    TCE_pts = np.zeros((points_size,2))     #This array will store the points after transformation by the exchange part of the TCE, 
                                            #and plotting them reveals the partition of the first return map into alternating rhombi. 

    for i in range(points_size):
        TCE_pts[i], _ = TCE(points[i], cone_angles, rotation_vec, translation_vec)

    first_return_times = np.zeros(points_size) #The first return times will determine the colouring of each point.

    colour_values = np.zeros(points_size)

    for _ in tqdm(range(num_iter)):
        for i in range(points_size):
            
            points[i], fht = First_Return_Pc(points[i], cone_angles, rotation_vec, translation_vec, max_iter=max_iter)

            first_return_times[i] += fht
    
    colour_values = np.mod(1/2 + 8*first_return_times/max(first_return_times), 1) #The coefficients here can be modified to shift the colour values for the plot
    #The advantage of this colouring is that it is fairly easy to calculate, however it is not the most effective at distinguishing between cells.

    #- The first plot will show the n-cells in their original positions
    ax.scatter(initial_points[:,0],initial_points[:,1],c=colour_map(colour_values),**kwargs)
    #- The second plot will show the n-cells after permuting the cones in the middle cone
    # ax.scatter(TCE_pts[:,0]-translation_vec[1,0],TCE_pts[:,1],c=colour_map(colour_values),**kwargs)
    #- the third plot will show the n-cells after n iterates of the first return map.
    # ax.scatter(points[:,0],points[:,1],c=colour_map(colour_values),**kwargs)


#--- Example initialisation of variables
    
#NOTE: It is recommended with this code that you run it via a script rather than on the command line

# cone_angles = np.array([np.pi/2-0.7,0.8,0.6,np.pi/2 - 0.7])
# permutation = np.array([0,2,1,3])
# rotation_vec = Rotation_Angles(cone_angles, permutation)

# l = (np.sqrt(5)-1)/2
# eta = 1 - l
# rho = 1
# translation_vec = np.zeros((cone_angles.shape[0],2))
# translation_vec[1:-1,0] = -eta
# translation_vec[0,0] = -rho
# translation_vec[-1,0] = l

# box_limits = [-rho,l,0,0.7]
# num_points = 500
# num_iter = 1250 #Try to keep num_iter above 600, because in the Plot_TCE function, the first 600 iterates are removed from the plot as transients (to remove noise.)
# colour_map = cm.get_cmap('viridis') #Check the available colourmaps for a list of choices.  My favourite for Plot_TCE is 'Blues'.
    
# fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize=(8,6)) #Initialising axes.  Feel free to change figsize to suit your screen.


#--- Example execution of the plot functions

# Plot_TCE(ax1, cone_angles, rotation_vec, translation_vec, box_limits, num_points, num_iter, colour_map, s=0.1, alpha=1, marker='o')
# # Plot_TCE_Cells(ax1, cone_angles, rotation_vec, translation_vec, 10, box_limits, 2e-3, s=0.1, marker='o')
# # Plot_FR_Cells(ax1, cone_angles, rotation_vec, translation_vec, 1, box_limits, 2.5e-3, colour_map, max_iter=10000, s=0.3, marker='o')

# ax1.set_aspect(1)    #This ensures that there is no artificial stretching/squishing in the axes for the final image.
# ax1.set_xlim(box_limits[0], box_limits[1]) #You can change these values if you wish, but keep in mind only the trajectories of points starting in box_limits are generated.
# ax1.set_ylim(box_limits[2], box_limits[3]) #So if the trajectories don't reach the part of the image you want to view, you will need to change box_limits.
# plt.show()
