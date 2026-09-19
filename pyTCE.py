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
#- ...but also try to read the comments at the end of the plot_tce, plot_tce_Cells and plot_first_return_cells functions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import copy
from tqdm import tqdm

#--- The Cone class is made for the easy parametrisation and plotting of cones in the plane.  
#    I use it primarily for demonstrations and figures, but it has no use in any further code.

class Cone:
    def __init__(self, vertex, width, angle):  #I defined this class in case I wanted to make demonstrations/figures to showcase what the cones do under the action of a TCE
        self.__vertex = vertex  #Any cone is defined by three parameters, the position of its vertex, and the angles of the two boundary rays originating from it from the horizontal.
        self.__width = width  #The angle of the cone at its vertex, measured in the anti-clockwise direction.
        self.__angle = angle  #The anti-clockwise angle between the horizontal and one of the boundary lines.
        #Note that the first boundary line is determined by self.angle, and the second one is determined from the first using self.width
    
    @property
    def vertex(self):
        return self.__vertex

    @vertex.setter
    def vertex(self, new_vertex):
        self.__vertex = new_vertex

    @property
    def width(self):
        return self.__width

    @width.setter
    def vertex(self, new_width):
        if new_width <= 0:
            raise ValueError("The width of the cone must be positive")
        self.__width = new_width

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def vertex(self, new_angle):
        self.__angle = new_angle

    def vertices(self, radius): #This function generates the vertices of the triangle created by cutting off the boundary lines at a radius from the vertex
        """The vertices are ordered as follows: the origin of the cone, 
        the vertex with the smaller angle relative to the horizontal line going through the origin, then the vertex with the larger angle."""
        v1 = self.vertex
        v2 = self.vertex + radius*np.array([np.cos(self.angle), np.sin(self.angle)])
        v3 = self.vertex + radius*np.array([np.cos(self.angle + self.width), np.sin(self.angle + self.width)])
        return v1, v2, v3
    
    def rotate(self, rotation_angle):   #This rotates the entire cone.
        self.angle += rotation_angle
    
    def translate(self, translation):   #This translates the entire cone.
        self.vertex += translation
    
    def hollow_plot(self, ax, radius, **kwargs):    #This plots the boundary of the cone, consisting of the vertex and its two boundary rays (up to the radius)
        v1, v2, v3 = self.vertices(radius)
        ax.plot([v2[0],v1[0],v3[0]], [v2[1],v1[1],v3[1]],**kwargs)

    def filled_plot(self, ax, radius, **kwargs):    #This function plots a filled-in triangle with vertices determined by self.vertices(radius)
        v1, v2, v3 = self.vertices(radius)
        ax.fill([v2[0], v1[0], v3[0]], [v2[1], v1[1], v3[1]], **kwargs)


#--- TCE computation     

def rotation_angles(cone_angles, permutation):
    """This function computes the rotation angles for our piecewise isometry. 
    cone_angles is a numpy array containing the angle widths of each cone, in anti-clockwise order (right-to-left),
    permutation is a list of the integers 0, ..., number_of_cones - 1 representing the new ordering of the cones after they are exchanged,
    where d is the number of cones in the partition."""

    number_of_cones = np.size(cone_angles)
    cone_rotations = np.zeros(number_of_cones)
    
    for j in range(number_of_cones): #Loop to find rotation angle for each cone
        permuted_angle_sum = 0
        for k in range(number_of_cones): #The sum of the angles of cones that appear before the jth cone after permutation
            if permutation[k] < permutation[j]:
                permuted_angle_sum += cone_angles[k]
        
        angle_sum = 0
        for k in range(j): #The sum of the angles of cones that appear before the jth cone before permutation
            angle_sum += cone_angles[k]
        
        cone_rotations[j] = permuted_angle_sum - angle_sum #The difference is the rotation angle
    
    return cone_rotations


def tce(point, cone_angles, rotation, translation):
    """This function performs one iteration of a TCE.
    point is a numpy array representing a vector in the plane,
    cone_angles is a numpy array representing the angle widths of the cones in anti-clockwise order,
    rotation is a numpy array containing the angles each cone rotates under the TCE, again in anti-clockwise order,
    translation is a numpy array containing the amount each cone shifts under the TCE, again in anti-clockwise order.
    """

    if point[1] < 0: # If point is not in the closed upper half plane, then this function doesn't work.
        print(point)
        raise Exception('point must have a non-negative y-coordinate.')
    
    point_angle = np.arctan2(point[1],point[0]) #The argument of the point x.

    current_cone = 0    # This is the argument of the right boundary line of the cone j, starting from cone 0.
    for j in range(np.size(cone_angles)):
        current_cone += cone_angles[j]

        if point_angle <= current_cone:
            # Compare the angles of the cone boundary lines with the angle of the point until we know which cone the point is in.
            rotation_matrix = np.array([[np.cos(rotation[j]),-np.sin(rotation[j])], # Rotation matrix for the jth cone
                                        [np.sin(rotation[j]), np.cos(rotation[j])]])
            return np.dot(rotation_matrix, point) + translation[j], j # The rotated point 


def first_return(point, cone_angles, rotation, translation, max_iter=1000):
    """This function computes one iterate of the first return map of the TCE to the middle cone, that is to say the upper half plane without the first and last cones.
    point is a numpy array representing a vector in the plane,
    cone_angles is a numpy array representing the angle widths of the cones in anti-clockwise order,
    rotation is a numpy array containing the angles each cone rotates under the TCE, again in anti-clockwise order,
    translation is a numpy array containing the amount each cone shifts under the TCE, again in anti-clockwise order,
    max_iter is the maximum number of iterations of the TCE that the function allows before halting and returning a value."""

    point_copy = copy(point) #Note that this function does work for points outside of the middle cone, 
                #...simply returning the first point in the trajectory where the point enters the middle cone.

    angle_sums = 0

    for n in range(max_iter):
        if all(point == np.array([0, -1])):
            break
        point_copy, _ = tce(point_copy, cone_angles, rotation, translation)  #At each run of the for loop, we iterate the point under the TCE

        point_argument = np.arctan2(point_copy[1],point_copy[0])  #Calulating the argument of the point.

        if point_argument >= cone_angles[0] and point_argument <= np.pi - cone_angles[-1]:  #This checks whether the point has (re-)entered the middle cone.
            angle_sums = np.mod(angle_sums,1)
            return point_copy, n+1 
            #If after n iterations, we return to the middle cone, we stop and return the point tce^n(point) and the time n. 
            #Note: the index n starts from 0 to N-1, so we add 1.

    return np.array([0, -1]), -1   #It is impossible for this to be a valid output, so this serves as an indication that the function has reached max_iter without returning to the middle cone.


#--- Functions for generating points

def generate_random_points(box_limits, num_points):
    """This function generates an array of vectors uniformly distributed within a bounding box.
    box_limits is a list of the form [x_min, x_max, y_min, y_max],
    num_points is the number of points being generated."""

    x_min, x_max, y_min, y_max = box_limits
    choices = np.random.rand(2, num_points) #generating uniformly distributed 2D vectors within the box [0,1)x[0,1)
    points = np.multiply(choices, np.array([x_max - x_min, y_max - y_min])[:,None]).T + np.array([x_min, y_min]) #Deforming the randomly generated points into the bounding box
    return points


def generate_grid(box_limits, resolution):
    """This function generates a uniform grid of points within a bounding box.
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


#--- Plotting functions

def plot_tce(ax, cone_angles, rotation, translation, box_limits, num_points, num_iter, colour_map, **kwargs):
    """This function plots a graph of the orbits of uniformly distributed points under the TCE with parameters cone_angles, rotation and translation.
    Each point and its iterates are given an arbitrary distinct colour.
    ax is an instance of the matplotlib Axes class,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    num_points is the number of points being iterated,
    num_iter is the number of iterates being computed for each point,
    colour_map is a colourmap chosen from the matplotlib library,
    **Kwargs are passed to the ax.scatter function."""

    #We rescale the elements of each vector so that the resulting vectors are within the box with limits [xmin, xmax] in x-coordinate and [ymin, ymax]in y-coordinate.
    points = generate_random_points(box_limits, num_points)

    orbit = np.zeros((num_points*num_iter, 2))  #Preset the array which will store the trajectories of the points we have chosen.  The shape makes it easier to plot.
    orbit[:num_points, :] = points  #We initialise the array with the initial points (time t=0).

    plot_colours = np.zeros(num_points*num_iter) 
    plot_colours[:num_points] = list(range(1, num_points + 1))

    for n in tqdm(range(1, num_iter)):
        for j in range(num_points):
            orbit[n*num_points + j], _  = tce(orbit[(n-1)*num_points + j], cone_angles, rotation, translation)  #We calculate the next point in the trajectory.
            plot_colours[n*num_points + j] = j + 1  #Ensuring that each trajectory has the same colour value as its initial point, and distinct trajetories have distinct colours.

    plot_colours = plot_colours/(num_points + 2)
    
    ax.scatter(orbit[600*num_points:, 0], orbit[600*num_points:, 1], c=colour_map(plot_colours[600*num_points:]), **kwargs)
    #Some orbits start with a transient part where they drift around before entering a periodic island, 
    #...but this makes the final image look grainier/noisier, so we omit the first 600 iterates of each trajectories.


def plot_tce_cells(ax, cone_angles, rotation, translation, num_iter, box_limits, resolution, **kwargs):
    """This function plots the n-cells for the TCE with parameters from cone_angles, rotation, translation, where n = num_iter.
    ax is an instance of the matplotlib Axes class,
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    **kwargs are passed to the ax.scatter function."""

    number_of_cones = np.size(cone_angles)

    grid = generate_grid(box_limits, resolution)
    numel = grid.shape[0]

    colour_choices = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] #These are the colour choices assigned to the first, middle and last cones
    
    points = copy(grid) #We want to store the original positions of the grid, using the iterated grid points to indicate the colours of the original grid points.
    colour_values = np.zeros((numel, 3))

    for _ in tqdm(range(1, num_iter + 1)):
        for i in range(numel):
            grid[i], j = tce(grid[i], cone_angles, rotation, translation)
            if j == 0: #The cones the first, middle and last cones are each assigned a different colour from colour_choices.
                jstar = 0
            elif j == number_of_cones - 1:
                jstar = 2
            else:
                jstar = 1
            colour_values[i] += colour_choices[jstar] #The colour of an orbit is determined by the average colour of cones the point visits.

    colour_values *= 1/num_iter

    ax.scatter(points[:,0],points[:,1],c=colour_values,**kwargs)



def plot_first_return_cells(ax, cone_angles, rotation, translation, num_iter, box_limits, resolution, colour_map, max_iter=1000, **kwargs):
    """This function plots the n-cells, where n = num_iter, for the first return map of the TCE with parameters cone_angles, rotation, translation, to the middle cone.
    The colouring of a point is determined by the sequence of first return times to the middle cone in its orbit up to num_iter.
    box_limits is a list of the form [xmin, xmax, ymin, ymax],
    resolution is the distance between adjacent points in the grid,
    colour_map is a colourmap chosen from the matplotlib library,
    max_iter is the maximum number of iterations of the TCE that the first return map allows before halting and returning a value,
    **kwargs are passed to the ax.scatter function."""

    grid = generate_grid|(box_limits, resolution)

    points = np.empty((0,2))

    for point in tqdm(grid): #A point in the grid is added to the points array only if it is within the middle cone.
        argument = np.arctan2(point[1], point[0])

        if argument >= cone_angles[0] and argument <= np.pi - cone_angles[-1]:
            points = np.append(points, [point], axis=0)
    
    points_size = points.shape[0]
    
    initial_points = copy(points)   #This array stores the original state of the points for the purpose of plotting.
    tce_points = np.zeros((points_size, 2))     #This array will store the points after transformation by the exchange part of the TCE, 
                                            #and plotting them reveals the partition of the first return map into alternating rhombi. 

    for i in range(points_size):
        tce_points[i], _ = tce(points[i], cone_angles, rotation, translation)

    first_return_times = np.zeros(points_size) #The first return times will determine the colouring of each point.

    colour_values = np.zeros(points_size)

    for _ in tqdm(range(num_iter)):
        for i in range(points_size):
            
            points[i], first_return_time = first_return(points[i], cone_angles, rotation, translation, max_iter=max_iter)

            first_return_times[i] += first_return_time
    
    colour_values = np.mod(1/2 + 8*first_return_times/max(first_return_times), 1) #The coefficients here can be modified to shift the colour values for the plot
    #The advantage of this colouring is that it is fairly easy to calculate, however it is not the most effective at distinguishing between cells.

    #- The first plot will show the n-cells in their original positions
    ax.scatter(initial_points[:, 0], initial_points[:, 1], c=colour_map(colour_values), **kwargs)
    #- The second plot will show the n-cells after permuting the cones in the middle cone
    # ax.scatter(tce_points[:,0]-translation_vec[1,0],tce_points[:,1],c=colour_map(colour_values),**kwargs)
    #- the third plot will show the n-cells after n iterates of the first return map.
    # ax.scatter(points[:,0],points[:,1],c=colour_map(colour_values),**kwargs)


#--- Example initialisation of variables
    
#NOTE: It is recommended with this code that you run it via a script rather than on the command line

# cone_angles = np.array([np.pi/2 - 0.7, 0.8, 0.6, np.pi/2 - 0.7])
# permutation = np.array([0, 2, 1, 3])
# rotation = rotation_angles(cone_angles, permutation)

# l = (np.sqrt(5)-1)/2
# eta = 1 - l
# rho = 1
# translation = np.zeros((cone_angles.shape[0], 2))
# translation[1:-1, 0] = -eta
# translation[0, 0] = -rho
# translation[-1, 0] = l

# box_limits = [-rho, l, 0, 0.7]
# num_points = 500
# num_iter = 1250 #Try to keep num_iter above 600, because in the plot_tce function, the first 600 iterates are removed from the plot as transients (to remove noise.)
# colour_map = cm.get_cmap('Blues') #Check the available colourmaps for a list of choices.  My favourite for plot_tce is 'Blues'.
    
# fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 8)) #Initialising axes.  Feel free to change figsize to suit your screen.


#--- Example execution of the plot functions

# plot_tce(ax1, cone_angles, rotation, translation, box_limits, num_points, num_iter, colour_map, s=0.1, alpha=1, marker='o')
# plot_tce_cells(ax1, cone_angles, rotation, translation, 10, box_limits, 2e-3, s=0.1, marker='o')
# plot_first_return_cells(ax1, cone_angles, rotation, translation, 1, box_limits, 2.5e-3, colour_map, max_iter=10000, s=0.3, marker='o')

# ax1.set_aspect(1)    #This ensures that there is no artificial stretching/squishing in the axes for the final image.
# ax1.set_xlim(box_limits[0], box_limits[1]) #You can change these values if you wish, but keep in mind only the trajectories of points starting in box_limits are generated.
# ax1.set_ylim(box_limits[2], box_limits[3]) #So if the trajectories don't reach the part of the image you want to view, you will need to change box_limits.
# plt.show()
