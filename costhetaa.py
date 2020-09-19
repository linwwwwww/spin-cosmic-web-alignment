import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

def costhetaa(massline,showG,w,v,halo):

    y=np.zeros([len(massline)])
    error=np.zeros([len(massline)])
    for i in np.arange(len(massline)-1):
        print(i)
        massmin=massline[i]       
        massmax=massline[i+1]
        num=(np.where((halo['Mass']>=massmin)&(halo['Mass']<massmax)))
        
        halos=halo[num]
        
        #高质量：
        #向上取整求halo的坐标
        posround=np.rint(halos["Pos"])

        #四舍五入求halo的坐标
        #posround=np.rint(halo["pos"])
        posround[np.where(posround==500)]=499

        posround=posround.astype(np.int16)
 
        #在filament中的halo占所有halo的概率
        showGww=showG[posround[:,0],posround[:,1],posround[:,2]]#取出halo的对应位置的值
        poshf=np.where(showGww==2)#在filament上的halo的序号
        fraction=(np.shape(poshf))[1]/len(posround)
        print(fraction)

        #取出在filament中相关的halo信息
        halof=halos[poshf]
        posroundf=np.rint(halof["Pos"])
        posroundf=posroundf.astype(np.int16)
        
        
        wwwa=w[posroundf[:,0],posroundf[:,1],posroundf[:,2]]#halo对应的
        vwwa=v[posroundf[:,0],posroundf[:,1],posroundf[:,2]]

        
        sort=np.argsort(wwwa)
        for j in range(len(wwwa)):
            a=wwwa[j]
            b=vwwa[j]
            wwwa[j]=a[sort[j]]
            vwwa[j]=b[sort[j]]

        vector=vwwa[:,0]
        
        spinvalues=np.sqrt(np.square(halof["Spin"][:,0])+(np.square(halof["Spin"][:,1]))+np.square(halof["Spin"][:,2]))
        spinx=halof["Spin"][:,0]/spinvalues
        spiny=halof["Spin"][:,1]/spinvalues
        spinz=halof["Spin"][:,2]/spinvalues
        
        vectorvalues=np.sqrt(np.square(vector[:,0])+(np.square(vector[:,1]))+np.square(vector[:,2]))
        vector[:,0]=vector[:,0]/vectorvalues
        vector[:,1]=vector[:,1]/vectorvalues
        vector[:,2]=vector[:,2]/vectorvalues
        costheta=(vector[:,0]*spinx+vector[:,1]*spiny+vector[:,2]*spinz)
        
        error[i]=np.std(costheta)/np.sqrt(len(costheta))
        y[i]=np.median(abs(costheta))
        #draw_histogram(abs(costheta))
        
    a=np.ones([len(massline)])*10e9
    x=(massline)*a
    plt.plot(np.log10(x[0:-1]),y[0:-1])
    #plt.title("eagle_halo_spin*$v_f$ Rs=1.36Mpc/h Threshold=0.0012 stellarmass")
    plt.errorbar(np.log10(x[0:-1]),y[0:-1],yerr=error[0:-1],fmt='o',ecolor='r',color='b',elinewidth=1,capsize=1)
    plt.ylim([0.4,0.6])
    return y,error
