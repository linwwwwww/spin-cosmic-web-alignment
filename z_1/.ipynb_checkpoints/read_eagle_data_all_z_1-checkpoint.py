import numpy as np
import h5py
import eagle_python as eagle
import read_eagle as read_particle
from multiprocessing import Pool
import multiprocessing



#read data
r200=np.zeros([1])
mass=np.zeros([1])
pos=np.zeros([1,3])
GN=np.zeros([1])
Vel=np.zeros([1,3])
starmass=np.zeros([1])
FirstID=np.zeros([1])
starspin=np.zeros([1,3])
for i in np.arange(256):
    f = h5py.File('/Simulations/Eagle/RefL0100N1504/groups_023_z000p503/eagle_subfind_tab_023_z000p503.'+np.str(i)+'.hdf5', 'r')
    r200=np.concatenate((r200,f['FOF']['Group_R_Crit200']))
    pos=np.concatenate((pos,f['FOF']['GroupCentreOfPotential']))
    mass=np.concatenate((mass,f['FOF']['Group_M_Crit200']))
    GroupNumber=f["Subhalo"]["GroupNumber"]
    GN=np.concatenate((GN,GroupNumber))

    Vel=np.concatenate((Vel,f["Subhalo"]["Velocity"])) 
    starmass=np.concatenate((starmass,f['Subhalo']['Stars']['Mass']))    
    FirstID=np.concatenate((FirstID,f['FOF']['FirstSubhaloID']))
    starspin=np.concatenate((starspin,f['Subhalo']['Stars']['Spin'])) 
    f.close()
r200 = np.delete(r200, 0, 0)
Vel = np.delete(Vel, 0, 0)
pos = np.delete(pos,0,0)
mass = np.delete(mass,0,0)
GN = np.delete(GN,0,0)
starmass=np.delete(starmass,0,0)
FirstID=np.delete(FirstID,0,0)
starspin=np.delete(starspin,0,0)

Vel=Vel[np.int32(FirstID)]#选择中心halo
starmass=starmass[np.int32(FirstID)]
starspin=starspin[np.int32(FirstID)]

num=(np.where(mass>=1))#M200>10^10 M_sun
r200=r200[num]
pos=pos[num]
mass=mass[num]
Vel=Vel[num]
starmass=starmass[num]
starspin=starspin[num]

select=(np.where(starmass>=1e-1))#stellar mass>10^9 M_sun
r200=r200[select]
pos=pos[select]
mass=mass[select]
Vel=Vel[select]
starmass=starmass[select]
starspin=starspin[select]

np.save("/data/dell5/userdir/hy/EAGLE-spin/galaxy_properties/z_0.5/stellarspin1e-1_201014_z_0.5.npy",starspin)
np.save("/data/dell5/userdir/hy/EAGLE-spin/galaxy_properties/z_0.5/stellarmass1e-1_201014_z_0.5.npy",starmass)

print("finish 1")
print("len(mass)=%s\n"%(len(mass)))

#use the particle data to compute spin

basePath= '/Simulations/Eagle/RefL0100N1504/'
snapnum=23


def parts_in_region(basePath,snapnum, partType, centre: 'cMpc/h', region_length:'cMpc/h', fields):
    #print('\n##\n#region_length is box\'s length of selected region, not radius',)
    #fname   = basePath+"/snapshot_0%d_z000p000/snap_0%d_z000p000.0.hdf5" % (snapnum, snapnum)
    fname   = eagle.snapshot.snapPath(basePath,snapnum)
    itype = partType
    result = {}
    # make sure fields is not a single element
    if isinstance(fields, str):
        fields = [fields]
    
    # Open the snapshot
    snap = read_particle.EagleSnapshot(fname)
    # Specify the region to read (coords. are in comoving Mpc/h)
    xmin = centre[0]- 0.5*region_length
    xmax = centre[0]+ 0.5*region_length
    ymin = centre[1]- 0.5*region_length
    ymax = centre[1]+ 0.5*region_length
    zmin = centre[2]- 0.5*region_length
    zmax = centre[2]+ 0.5*region_length
    snap.select_region(xmin, xmax, ymin, ymax, zmin, zmax)
    
    
    #print ("# Number of particles in this region = %d" % snap.count_particles(itype))
    # Read positions and IDs of particles of type itype in the specified region.
    for field in fields:
        result[field]= snap.read_dataset(itype, field)
    snap.close()
    return result
print("finish 2")

parttype=1
cores = multiprocessing.cpu_count()

def Jcompute(i):
 
    halop=parts_in_region(basePath,snapnum,parttype,pos[i],r200[i],['Coordinates','Velocity'])      
    halopr=halop['Coordinates']-pos[i]
    halopv=halop['Velocity']-Vel[i]
    
    r3d=np.sqrt(np.sum(halopr*halopr, axis = 1))
    halopr=halopr[np.where(r3d < r200[i]),:][0]
    halopv =halopv[np.where(r3d < r200[i]),:][0]
    Jall=np.cross(halopr,halopv)
    J=Jall.sum(axis=0)
    
    return J
if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    with Pool(cores) as p:
        J=p.map(Jcompute, range(len(mass)))

print("finish 3")
Galneedtype=[
    ( 'Pos'                  , (np.float32, 3)), 
    ( 'Spin'            , (np.float32, 3)),
    ( 'Mass'           , np.float32),
]
halo=np.zeros(len(mass),dtype=Galneedtype)
halo['Pos']=pos
halo["Spin"]=J
halo['Mass']=mass

np.save("/data/dell5/userdir/hy/EAGLE-spin/galaxy_properties/z_0.5/halo_pos_J_mass_z_0.5_201014.npy",halo)
