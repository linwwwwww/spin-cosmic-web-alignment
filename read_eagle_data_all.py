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
SGN=np.zeros([1])
Vel=np.zeros([1,3])
starmass=np.zeros([1])
CentreOfMass=np.zeros([1,3])
for i in np.arange(256):
    f = h5py.File('/Simulations/Eagle/RefL0100N1504/groups_028_z000p000/eagle_subfind_tab_028_z000p000.'+np.str(i)+'.hdf5', 'r')
    r200=np.concatenate((r200,f['FOF']['Group_R_Crit200']))

    pos=np.concatenate((pos,f['FOF']['GroupCentreOfPotential']))
    mass=np.concatenate((mass,f['FOF']['Group_M_Crit200']))
    GroupNumber=f["Subhalo"]["GroupNumber"]
    GN=np.concatenate((GN,GroupNumber))
    SubGroupNumber=f["Subhalo"]["SubGroupNumber"]
    SGN=np.concatenate((SGN,SubGroupNumber))
    Vel=np.concatenate((Vel,f["Subhalo"]["Velocity"])) 
    starmass=np.concatenate((starmass,f['Subhalo']['Stars']['Mass']))    
    CentreOfMass=np.concatenate((CentreOfMass,f["Subhalo"]["CentreOfMass"])) 
    
r200 = np.delete(r200, 0, 0)
Vel = np.delete(Vel, 0, 0)
pos = np.delete(pos,0,0)
mass = np.delete(mass,0,0)
GN = np.delete(GN,0,0)
SGN = np.delete(SGN,0,0)
starmass=np.delete(starmass,0,0)
CentreOfMass=np.delete(CentreOfMass,0,0)

main=np.int32(np.where(SGN==0))
GN=GN[main]
Vel=Vel[main] 
starmass=starmass[main]
CentreOfMass=CentreOfMass[main]

GN=np.int32(GN-1)

GN=np.int32(GN)
r200=r200[GN]
pos=pos[GN]
mass=mass[GN] 

num=(np.where(mass>=1e-2))
r200=r200[num]
pos=pos[num]
mass=mass[num]
Vel=Vel[num]
starmass=starmass[num]
CentreOfMass=CentreOfMass[num]
np.save("starmass.npy",starmass)
np.save("CentreOfMass.npy",CentreOfMass)
print("finish 1")


#use the particle data to compute spin

basePath= '/Simulations/Eagle/RefL0100N1504/'
snapnum=28


fields=['MassType','HalfMassRad','CentreOfMass','CentreOfPotential','Velocity']



subhalo_info= eagle.groupcat.loadSubhalos(basePath,snapnum,fields=fields)
mstars= subhalo_info['MassType'][:,4]
halfr_star=  subhalo_info['HalfMassRad'][:,4]





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
print("len(mass)=%s\n"%(len(mass)))
def Jcompute(i):
    halop=parts_in_region(basePath,snapnum,parttype,pos[i],r200[i],['Coordinates','Velocity'])
    halopr=halop['Coordinates']-pos[i]
    halopv=halop['Velocity']-Vel[i]
    Jall=np.cross(halopr,halopv)
    
    J=Jall.sum(axis=0)
    if i%10000==0:
        print(np.str(i)+"\r")
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

np.save("halo_pos_J_mass.npy",halo)
