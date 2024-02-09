Within this repository, you will find the programs which Noah Cockram has used to generate the figures in his papers with Peter Ashwin and Ana Rodrigues.

pyTCE is a dependency for pyramid_map and trapezium_map (which you should be able to see from the 'import pyTCE_v1 as tce' lines near the beginning of both.
Therefore, try to keep all files in the same folder.

As said elsewhere, it is highly recommended that one runs these programs via a script (or simply by running the files themselves after uncommenting specific parts).
This is because there is a lot of variable initialisation, and the plotting functions do not show the figure after they have executed, unless you add 'plt.show()' at the end.
I have done it this way because in my experience with these programs, I have often wanted to plot additional details after the plotting functions have executed.

All of these program files have (in comments) examples of variable initialisation and execution of the plotting functions for your convenience.
Feel free to simply use these and experiment with parameters from there, and if you must execute these programs through the command line,
you can contain these parts at the end within a main() function, and then simply enter 'main()' on the command line.

Be aware that at the end of some of the plotting functions, specifically those plotting n-cell partitions, you can manually change the 'mode' of plotting.
What this means is that, roughly speaking, the positions of the n-cells at specific times in its orbit under the TCE are stored locally in the function,
and so you can plot what the n-cell partition looks like at those points in time.  however, to reiterate, changing the mode of plotting must be done
manually via commenting and uncommenting.
