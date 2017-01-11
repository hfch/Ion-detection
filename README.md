# Ion detection simulation
Simulation of ions in a Rydberg atoms ionization and detection process.

The Lua file export the Potential Array solution files of electrode in the Rydberg detector.
Then the potential array data are rearranged with Python code into a HDF5 file for later application.

The potential shapes are then ploted in python and compared with SIMION, as well as single ion trajectory.

### Python Simulation
Most of the python simulation are first done with ipython notebook, then some code are transformed into python scripts.

Documentation of ipython notebooks.

1. Compare exported potentials with SIMION plots to make sure the exported data are used in right way.

2. Find the initial atom center by flying single ions in a 3D grid space. 
The ion should reach the detector's plane center and the Time of Flight must be close to experimental result, which is 6.6 us.

 Then narrow down the space step by step.
 
3. By multiplying the electrode voltages with a factor, the time of flight of ions should change correpondingly. 
 The 3rd notebook compares the experimental results and simulation of the time of flight. 
 
 One should note that the experimental results are not exactly the TOF of ions, they are actually the time delay between the ion signal trace and electric field ramp.
 
4. Study the influence of Coulumbic interaction in two ions case on the trajectories and arrival position of ions on the detector.

5. Study the influence of Coulumbic interaction in two ions case on the trajectories and arrival position of ions on the detector.
 With right voltages settings.
 
6. First trial of simulation of interacting ions.

7. Most recent and reliable simulation of interacting ions.  In version 3, the interpolation method is changed to increase the simulation speed.
 
