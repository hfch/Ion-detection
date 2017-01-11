
# coding: utf-8

# Calculate single ion tajectory in Python with exported potentials and compare with SIMION

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

# In[10]:

#generate potential interpolation

from scipy.interpolate import RegularGridInterpolator as rgi
p_interpolation=rgi((x,y,z),potentials)


# ### Calculate single ion trajectory

# Define velocity iteration function and trajectory iteration function for single ion.

# In[6]:

#single ion's position and velocity iteration functions

def R(ri,vi,dt):                  #delta_t in unit of micro-second, displacement in unit of mm, v in unit of mm/us
    x_new=ri[0]+vi[0]*dt
    y_new=ri[1]+vi[1]*dt
    z_new=ri[2]+vi[2]*dt
    return np.array([x_new,y_new,z_new])

def v(ri,vi,delta_t,dr):
    dvx=-100./(1.0364*87.) * (p_interpolation([ri[0]+dr,ri[1],ri[2]])[0]-p_interpolation([ri[0]-dr,ri[1],ri[2]])[0])/(dr*2) #- for positive ions, + for electrons
    dvy=-100./(1.0364*87.) * (p_interpolation([ri[0],ri[1]+dr,ri[2]])[0]-p_interpolation([ri[0],ri[1]-dr,ri[2]])[0])/(dr*2)
    dvz=-100./(1.0364*87.) * (p_interpolation([ri[0],ri[1],ri[2]+dr])[0]-p_interpolation([ri[0],ri[1],ri[2]-dr])[0])/(dr*2)
    vx_new=vi[0]+dvx * dt
    vy_new=vi[1]+dvy * dt
    vz_new=vi[2]+dvz * dt
    return np.array([vx_new,vy_new,vz_new])


# In[7]:

#Starting point r0 and initial velocity v0
r0=[102.5,102.5,65]
v0=[0,0,0]


# **Distance step when calculating gradient has significant fluence on the trajectories**

# In[8]:

dr=0.5       #dr[mm] is used when calculating gradients
dt=0.001     #dt[us]


# In[11]:

v_iter=np.zeros((1,3))
r_iter=np.zeros((1,3))
r_iter[0]=r0
v_iter[0]=v0
while (r_iter[-1]>[x[0],y[0],z[0]]).all and (r_iter[-1]<[x[-1],y[-1],z[-1]]).all() and (r_iter[-1][2]+11*r_iter[-1][0]<=1499):
    v_iter=np.append(v_iter,[v(r_iter[-1],v_iter[-1],dt,dr)],0)
    r_iter=np.append(r_iter,[R(r_iter[-1],v_iter[-1],dt)],0)


# In[19]:

#import simion simulated single ion trajectory
simion_trajectory=pd.read_csv('./flying ion test v2.csv',header=None)

simion_trajectory.columns=['TOF','x','y','z','vx','vy','vz']
sim_x=simion_trajectory.x.values
sim_y=simion_trajectory.y.values
sim_z=simion_trajectory.z.values



xt=r_iter[:,0]
yt=r_iter[:,1]
zt=r_iter[:,2]


# In[21]:

from scipy.interpolate import interp1d


trj2=plt.figure()
plt.plot(xt,zt,label="Py")
plt.plot(sim_x,sim_z,label="Simion")
plt.title("Comparison of Single ion trajectory in a Y-view")
plt.xlabel("X/mm")
plt.ylabel("Z/mm")
plt.legend()
plt.show()


# In[28]:

trj3=plt.figure()
plt.plot(yt,zt,label="Py")
plt.plot(sim_y,sim_z,label="Simion")
plt.title("Comparison of Single ion trajectory in a Y-view")
plt.xlabel("Y/mm")
plt.ylabel("Z/mm")
#plt.legend()
plt.show()

