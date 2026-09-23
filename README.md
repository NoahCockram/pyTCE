Within this repository, you will find the programs which Noah Cockram has used to generate the figures in his papers with Peter Ashwin and Ana Rodrigues.

The functions simulated with pyTCE are part of a collection known as the Translated Cone Exchanges (TCEs), which are a family of piecewise isometries that break the upper half of a flat 2D plane into cones from the point (0,0), shuffle them according to rotations about (0,0), and then shift those cones to the left and right. For those initiated into mathematics, the introductions of Cockram et al.'s papers will suffice for a cursory background on TCEs.

pyTCE is a dependency for pyramid_map and trapezium_map (which you should be able to see from the 'import pyTCE as tce' lines near the beginning of both. Therefore, try to keep all files in the same folder.

As said elsewhere, it is highly recommended that one runs these programs via a script (or simply by running the files themselves after uncommenting specific parts). This is because there is a lot of variable initialisation, and the plotting functions do not show the figure after they have executed, unless you add 'plt.show()' at the end. I have done it this way because in my experience with these programs, I have often wanted to plot additional details after the plotting functions have executed.

All of these program files have (in comments) examples of variable initialisation and execution of the plotting functions for your convenience. Feel free to simply use these and experiment with parameters from there.

Be aware that at the end of some of the plotting functions, specifically those plotting n-cell partitions, you can manually change the 'mode' of plotting. What this means is that, roughly speaking, the positions of the n-cells at specific times in its orbit under the TCE are stored locally in the function, and so you can plot what the n-cell partition looks like at those points in time.  however, to reiterate, changing the mode of plotting must be done manually via commenting and uncommenting.
