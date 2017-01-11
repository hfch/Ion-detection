
# coding: utf-8

# Calculate potentials and compare it with SIMION

# ### Import potentials and coordinates from HDF5 file

# In[1]:

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# In[2]:

#Potential field are stored in a HDF5 file
f=h5py.File('PA_v4.hdf5','r')


# In[3]:

#coordinates x,y,z in unit of mm
x=f["Coordinates/X"].value/2.
y=f["Coordinates/Y"].value/2.
z=f["Coordinates/Z"].value/2.


# ### Set electrodes potentials and calculate spatial potential values
# 19 electrodes are set to certain potentials

# In[4]:

#set values of 19 electrodes, for ionization field of state n=30
electrode_set_potentials=[1010.,1010.,-1090.,-1000.,-50.,-2000.,0.,100.,900.,-1700.,1010.,1010.,-1090.,-1000.,0.,0.,0.,0.,0.];


# In[5]:

# generate potential by setting each electrode to its potential and adding up all electrodes' potential
potentials=np.zeros((len(x),len(y),len(z)))
for i in range(19):
    electroden=f["Potential Arrays/electrode"+str(i+1)].value
    potentials+=electrode_set_potentials[i]*electroden/10000.


# ### Interpolate potential arrays and plot cross sections for comparison

# In[6]:

#generate potential interpolation

from scipy.interpolate import RegularGridInterpolator as rgi
p_interpolation=rgi((x,y,z),potentials)


# In[7]:

nx=np.arange(min(x),max(x),0.2)
nz=np.arange(min(z),max(z),0.2)
ny=(max(y)+min(y))/2.
X,Z=np.meshgrid(nx,nz,indexing='ij')
Y=ny


# In[8]:

interpY_p=np.zeros((len(nx),len(nz)))
for i in range(len(nx)):
    for j in range(len(nz)):
        interpY_p[i,j]=p_interpolation([nx[i],ny,nz[j]])[0]


plt.contour(X,Z,interpY_p)
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Potential cross section at Y= 102mm")
plt.show()


# In[11]:

#load SIMION potential contour and surface plot of Y=102mm
from PIL import Image as im
simion_contour=im.open('./Plot comparison/Simion potential contour n=30.PNG')
plt.imshow(simion_contour)

