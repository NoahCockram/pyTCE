import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
from pyTCE import tce
import os

def compute_pixel_encodings(d, cone_angles, rotation, translation, resolution, bounding_box, max_iter):
    """This function discretises the chosen region into a grid of resolution x resolution squares.  
    Then, in each grid square we uniformly choose one point and iterate it under the TCE either max_iter times, or until we enter a grid square which has already been visited.  
    We then assign each visited grid square a frequency based on the number of visits to each cone.
    resolution must divide into both bbox[1]-bbox[0] and bbox[3]-bbox[2]"""

    x_size = int((bounding_box[2] - bounding_box[0])//resolution)
    y_size = int((bounding_box[3] - bounding_box[1])//resolution)

    visited = np.zeros((x_size,y_size), dtype=bool) # An array to track which grid squares have been previously visited.
    encodings = np.empty((x_size,y_size), dtype=object) # An array to store the computed encoding of point (up to the final iterate).

    for i in tqdm(range(x_size)):
        for j in range(y_size):
            if visited[i,j]:
                continue
            
            point = bounding_box[:2] + resolution*(np.array([i,j]) + np.random.rand(2)) # Uniformly chosen point in the (i,j)-th grid square.

            enc = ''
            visited[i,j] = True
            coord_list = [[i,j]] # An array to trace the grid squares we visited in the orbit of point.

            for _ in range(max_iter):
                point, cone = tce(point, d, cone_angles, rotation, translation)
                enc += str(cone)
                coords = ((point - bounding_box[:2])/resolution).astype(int) #computing the grid square that contains the current iterate.

                if coords[0] >= x_size or coords[1] >= y_size or coords[0] < 0 or coords[1] < 0:
                    # If the point escapes the bounding box during iteration, we ignore the grid square where it lands, since we won't be plotting that.
                    continue
                else:
                    if visited[coords[0],coords[1]]: #If our point enters a grid square visited by another point, we stop iterating
                        if encodings[coords[0],coords[1]] is not None: # If the encoding here is None, then the previous visit to this square came from earlier in this point's trajectory
                            enc = encodings[coords[0],coords[1]]
                        break
                    else:
                        coord_list += [coords]
                        visited[coords[0],coords[1]] = True
            
            for c in coord_list:
                encodings[c[0],c[1]] = enc
    
    # Here we save the frequency data as a txt file for later use. Be aware that file sizes can become large when the resolution is about 10^(-4) smaller than the sidelengths.
    # We also add a header row of this data which contains all of the parameters for the computation.
    with open('encoding_data.txt', 'w') as data_file:
        format_line = np.empty(y_size, dtype=object)
        format_line[0] = str(x_size)
        format_line[1] = str(y_size)
        format_line[2] = str(d)
        format_line[3] = str(bounding_box[0])
        format_line[4] = str(bounding_box[1])
        format_line[5] = str(bounding_box[2])
        format_line[6] = str(bounding_box[3])
        format_line[7] = str(resolution)
        format_line[8] = str(cone_angles[1])
        format_line[9] = str(-translation[0,0])
        format_line[10] = str(translation[-1,0])
        format_line[11] = str(max_iter)

        np.savetxt(data_file, np.vstack((format_line,encodings)),fmt='%s')

    return None

def plot_pixel_encodings(ax, colouring='freq', restrict_to_pyramid=0, weight=[1,1,1,1], **kwargs):
    """Plots the data file produced by ComputeTCE using a specified colouring scheme - a function which turns the encodings of each grid square into colours.
    The available colourings are:
    'freq' - calculates the frequency of appearance of each cone in the encoding of each point, each frequency is multiplied by the colour of its respective cone, and added together.
    'linear_tail_heavy' - The colours of each cone are added together after being weighted by the sum of indices at which the cone appears in the encoding.
    'exponential_tail_heavy' - The colours of each cone are added with weights equal to an exponential sum of the indices at which the cone appears in the encoding."""
    if not os.path.isfile('encoding_data.txt'):
        raise Exception('There is no data to plot.')
    
    data = np.loadtxt('encoding_data.txt', dtype=object)

    x_size = int(data[0, 0])
    y_size = int(data[0, 1])
    d = int(data[0, 2])
    bbox = data[0, 3:7].astype(float)
    resolution = float(data[0, 7])
    theta = float(data[0, 8])
    rho = float(data[0, 9])
    l = float(data[0, 10])
    max_iter = int(data[0, 11])

    new_data = data[1:]

    colour_list = np.array([[0,0,0], [1,1,0], [0,0,1], [1,1,1]])
    colour_array = np.zeros((y_size, x_size, 3))

    if colouring == 'freq':
        for i in tqdm(range(x_size)):
            for j in range(y_size):
                weighted_count = np.array([weight[k]*new_data[i, j].count(str(k)) for k in range(d)])
                colour_array[j, i] = np.dot(colour_list.T, weighted_count).T/sum(weighted_count)
    elif colouring == 'linear_tail_heavy':
        for i in tqdm(range(x_size)):
            for j in range(y_size):
                pixel_encoding = new_data[i, j]
                encoding_sum = np.zeros(d)
                for n in range(max_iter):
                    encoding_sum[int(enc[n])] += n
                weighted_sum = pixel_encoding*weight
                colour_array[j,i] = np.dot(colour_list.T, weighted_sum).T/sum(weighted_sum)
    elif colouring == 'square_tail_heavy':
        for i in tqdm(range(x_size)):
            for j in range(y_size):
                pixel_encoding = new_data[i,j]
                encoding_square_sum = np.zeros(d)
                for n in range(max_iter):
                    encoding_square_sum[int(enc[n])] += n*n
                weighted_square_sum = encoding_square_sum*weight
                colour_array[j,i] = np.dot(colour_list.T, weighted_square_sum).T/sum(weighted_square_sum)
    elif colouring == 'exponential_tail_heavy':
        b = 1.1
        logb = np.log(b)
        N = np.floor(64*np.log(2)/logb)
        for i in tqdm(range(x_size)):
            for j in range(y_size):
                pixel_encoding = new_data[i,j]
                encoding_exponential_sum = np.zeros(d)
                for n in range(int(max(0,len(pixel_encoding)-N)),len(pixel_encoding)):
                    encoding_exponential_sum[int(pixel_encoding[n])] += np.exp((n-len(pixel_encoding))*logb)
                weighted_exponential_sum = encoding_exponential_sum*weight
                colour_array[j,i] = np.dot(colour_list.T, weighted_exponential_sum).T/sum(weighted_exponential_sum)

    ax.imshow(colour_array, **kwargs)

    return None