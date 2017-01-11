
# coding: utf-8

# Warning 1: This file has to be placed in the same directory of the PA array files.
#Warning 2: Depending on PA array size, it may take long time to run this code. For 19 PA files of grid size of 121*121*173, it took more than 1 hour to finish.

# Export potential arrays of 19 electrodes into a HDF5 file.
# 
# The Potential arrays data of the 19 electrodes generated with SIMION are rearranged and stored into a hdf5 fle called "PA_v4.hdf5", in order to make use of the data more efficiently.
# In "PA_v4.hdf5" file, there are two data groups "Coordinates" and "Potential Arrays", and a dataset "iselectrode".
# "Coordinates" are contains 3 one-dimentional arrays ( called X, Y, Z ), which are the range of the volume of interest. The unit is in mm.
# "Potential Arrays" contains 19 three-dimention arrays, which are the potential solution of the 19 electrodes. The unit is V.
# "iselectrode" dataset is a three-dimension array, representing if the grid point is electrode or not. 0 means non-electrode grid point and 1 means electrode grid point.



import numpy as np
import pandas as pd
import h5py

# In[2]:

#import electrode 1 data
electrode1=pd.read_csv('electrode1.csv',header=None);
electrode1.columns=['x','y','z','iselectrode','potential']
#electrode1


# In[3]:

#get 3 dimensional grid points' coordinates
xx=electrode1.x.values;
yy=electrode1.y.values;
zz=electrode1.z.values;


# In[5]:

x=np.arange(min(xx),max(xx)+1)
y=np.arange(min(yy),max(yy)+1)
z=np.arange(min(zz),max(zz)+1)


# In[6]:

potential=electrode1.potential.values.reshape(len(x),len(y),len(z),order='C')


# In[16]:

#create HDF5 to store 19 electrodes' potential array data
f=h5py.File('PA_v4.hdf5','w')


# In[17]:

#create group named "Coordinates"
coor=f.create_group("Coordinates")


# In[18]:

#create datasets of X,Y,Z coordinates
xset=coor.create_dataset("X",(len(x),),dtype=np.float64)


# In[19]:

xset[...]=x


# In[20]:

yset=coor.create_dataset("Y",(len(y),),dtype=np.float64)
yset[...]=y
zset=coor.create_dataset("Z",(len(z),),dtype=np.float64)
zset[...]=z


# In[21]:

type(zset.value)


# In[22]:

#create potential array group to store potentials data
pa=f.create_group("Potential Arrays")


# In[23]:

import glob
directories=glob.glob('./electrode*'+'.csv')


# In[24]:

for n in range(len(directories)):
    vars()['ele'+str(n+1)+'set']=pa.create_dataset("electrode"+str(n+1),(len(x),len(y),len(z)),dtype=np.float64)
    data=np.loadtxt('electrode'+str(n+1)+'.csv',delimiter=',')
    potentials=data[:,4]
    vars()['ele'+str(n+1)+'set'][...]=potentials.reshape(len(x),len(y),len(z),order='C')


# In[26]:

#store grid points electrode properties, whether they are electrode points
iselectrode=electrode1.iselectrode.values


# In[27]:

iselectrode=iselectrode.reshape(len(x),len(y),len(z),order='C')


# In[28]:

isset=f.create_dataset("iselectrode",(len(x),len(y),len(z),),dtype=np.int16)


# In[29]:

isset[...]=iselectrode


# In[ ]:



