#30 min with 52cpus in LMEM1
#the script uses a maximum of 40GB mem

#%reset -f
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dask as da
import glob
import time
from tqdm import tqdm #to see progressbar for loops
from scipy.interpolate import interp1d #1d interp
import xesmf as xe #for spatial interpolation in projected or lon-lat coords
#for projections
from pyproj import Proj, transform, Transformer

#run this cell just once----
#from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster 
#this seems that is working---
client = Client(processes=False,n_workers=1,threads_per_worker=52,memory_limit='120GB')
#this produce memory limit problems:
#client = Client(processes=False,n_workers=12,threads_per_worker=1,memory_limit='4GB')
#
#this seems that is working, but with lots of warnings related to mem issues?---
#this produce the same result as Client, but we can'not see progress neither strem
#client = LocalCluster(processes=False,n_workers=1,threads_per_worker=24,memory_limit='48GB')
#
#this is not calling any partition, just running with 1 core in the node we are now---
#cluster = SLURMCluster(queue='normal',cores=24,memory='48GB',processes=1,interface="lo")
#cluster = SLURMCluster(queue='LMEM1',cores=2,memory='1GB',project='test',interface='lo',scheduler_options={'interface': 'lo'})
#cluster = SLURMCluster(queue='LMEM1',cores=4,memory='2GB',processes=1, interface='lo')
#cluster = SLURMCluster()
#cluster.scale(jobs=2)
#cluster.scale(memory='2GB')
#cluster.adapt(maximum_jobs=2)
#print(cluster.job_script())
#client = Client(cluster)
#
# open dashboard with this if link doesn't work
# http://localhost:8787/status


#----------
home_dir="/export/lv4/user/jfajardourbina/"
ml_dir=f"{home_dir}dws_ulf_getm_2D_depth_avg/experiments_post_proc/lagrangian_simulation_36years/machine_learning_github/Lagrangian_ML/"
dir_wind=f"{home_dir}dws_ulf_getm_2D_depth_avg/data/atmosphere/" #winds
dir_displacement="net_displacement/"
dir_topo=f"{home_dir}dws_ulf_getm_2D_depth_avg/experiments_post_proc/analysis_eulerian_data_36years/data_bathy_grid/" #topo data
file_topo="DWS200m.2012.v03.nc"
file_wind0="UERRA.2009.nc4" #any wind file
#
savee='everyM2' #saving track data every m2
deploy='everyM2'#deploy set of particles every m2
minTsim=60 #mimimum time of simulation (days)
maxTsim=91 #maximum time of simulation (days)
dir_tracks = f"{home_dir}dws_ulf_getm_2D_depth_avg/experiments_post_proc/lagrangian_simulation_36years/exp-deployHighVolume_coords-xcyc_save-{savee}_deploy-{deploy}_Tsim-{minTsim}-{maxTsim}d/tracks/"
#
npa_per_dep=12967 #number of particles per deployment
m2=int(12.42*3600+2) #period in seconds
nt_interp=283*2 #interpolate wind data every 9.43 min from 1h original data (factor of m2=44714)
ref_time=np.datetime64("1980-01-01") #reference time for time interpolation, could be any value
dx=400/1e3;dy=400/1e3 #particle grid resolution
#
#paths for output data
dir_post_proc_data=f"{ml_dir}post_proc_data/" #to save wind interp files
dir_interp_wind="wind/"
file_interp_wind_root="wind_avg_std_during_1M2_and_interp_to_particle_grid_for_convlstm.nc"


#--------
dsw=xr.open_dataset(dir_wind+file_wind0) #open any wind data
dsw.close()

dsto=xr.open_dataset(dir_topo+file_topo) #topo file
xct0=dsto.xc.min().values/1e3; yct0=dsto.yc.min().values/1e3 #=(0,0)


#--------
#open grid of displacements (use for convlstm)---
file_displacement=sorted(glob.glob(f'{dir_post_proc_data}{dir_displacement}*.nc',recursive=True))[0]
ds_dis=xr.open_dataset(file_displacement); ds_dis.close()
xcdis0,ycdis0=ds_dis.x,ds_dis.y; del ds_dis
xcdis,ycdis=np.meshgrid(xcdis0,ycdis0)
#
#or build it---
#xmin=x0.min();xmax=x0.max();ymin=y0.min();ymax=y0.max()
#extend_grid=10 #so from particle min max positions extend grid 10*dx (to not have problems with convolution)
#xgrid=np.arange(xmin-dx*1e3*extend_grid,xmax+dx*1e3*(extend_grid+1),dx*1e3,dtype='float32')
#ygrid=np.arange(ymin-dy*1e3*extend_grid,ymax+dy*1e3*(extend_grid+1),dy*1e3,dtype='float32')
#xgrid,ygrid=np.meshgrid(xgrid,ygrid)


#define the transformations----------
#1)
#from epgs:28992(DWS) to epgs:4326(LatLon with WGS84 datum used by GPS and Google Earth)
proj = Transformer.from_crs('epsg:28992','epsg:4326',always_xy=True)
#2)
#from epgs:4326(LatLon with WGS84) to epgs:28992(DWS) 
inproj = Transformer.from_crs('epsg:4326','epsg:28992',always_xy=True)
#inproj_old=Proj("EPSG:28992") #old method (has errors 10-20m when contrast with the rotated coords)

#lon,lat to 28992(DWS)-projection--------------------

#bathymetry--------
xct=dsto.lonc.values;  yct=dsto.latc.values #lon,lat units
xctp,yctp,z = inproj.transform(xct,yct,xct*0.)
#[xctp,yctp] = inproj_old(xct,yct) #old method
xctp=(xctp)/1e3; yctp=(yctp)/1e3 
#first projected point to correct the coordinates of model local meter units
xctp0=xctp[0,0]; yctp0=yctp[0,0]


#local meter model units to 28992(DWS)-projection and lon-lat--------------

#matrix rotation -17degrees-----
ang=-17*np.pi/180
angs=np.ones((2,2))
angs[0,0]=np.cos(ang); angs[0,1]=np.sin(ang)
angs[1,0]=-np.sin(ang); angs[1,1]=np.cos(ang)

#bathymetry----
#original topo points in meter
xct2,yct2=np.meshgrid(dsto.xc.values,dsto.yc.values)
xy=np.array([xct2.flatten(),yct2.flatten()]).T
#rotate
xyp=np.matmul(angs,xy.T).T/1e3
xyp0=xyp[0,:] #the first rotated point in the topo data in meter =0,0
#correction from rotation to projection:
#1)substact the first rotated topo point in meter, but give tha same as xyp0=[0,0]
#2)add the first projected point of the case (lon,lat model units to projection)
xyp=xyp-xyp0 
xyp[:,0]=xyp[:,0]+xctp0; xyp[:,1]=xyp[:,1]+yctp0 
xyp=np.reshape(xyp,(len(dsto.yc.values),len(dsto.xc.values),2))
xctp2=xyp[...,0]; yctp2=xyp[...,1] #km
#
#contrast projections (lon,lat model units to meter) with rotated case
#around 0 meter diff with new method
#10 meter difference in average and maximum of 20 with old method
a=xctp-xctp2; b=yctp-yctp2
print(np.abs(a).max()*1e3, np.abs(b).max()*1e3, np.abs(a).mean()*1e3, np.abs(b).mean()*1e3) 


#particle grid of displacements (use for convlstm)------
xy=np.array([xcdis.flatten(),ycdis.flatten()]).T
ny,nx=xcdis.shape
#rotate
xyp=np.matmul(angs,xy.T).T/1e3
#correction from rotation to projection:
#1)substact the first rotated topo point in meter, but give tha same as xyp0=[0,0]
#2)add the first projected point of the case (lon,lat model units to meter)
xyp=xyp-xyp0 
xyp[:,0]=xyp[:,0]+xctp0; xyp[:,1]=xyp[:,1]+yctp0 
xyp=np.reshape(xyp,(ny,nx,2))
xcdisp=xyp[...,0]; ycdisp=xyp[...,1] #km
#
#get coordinates in lon-lat units (WGS84 ) 
xcdisp_lon, ycdisp_lat, _ = proj.transform(xcdisp*1e3,ycdisp*1e3, ycdisp*0.)


#for spatial interpolation using lon-lat-----

#build the input grid (lon-lat of original wind file)---
ds_in = xr.Dataset()
ds_in.coords["lon"] = dsw.lon.astype('float32')
ds_in["lon"].attrs['long_name'] = 'longitude'
ds_in.coords["lat"] = dsw.lat.astype('float32')
ds_in["lat"].attrs['long_name'] = 'latidude'
print(ds_in)
print()

#build the output grid (lon-lat of particle displacement)---
#this grid is used for the interpolation
ds_out = xr.Dataset()
ds_out.coords["lon"] = (("yc","xc"),xcdisp_lon.astype('float32'))
ds_out["lon"].attrs['long_name'] = 'longitude'
ds_out.coords["lat"] = (("yc","xc"),ycdisp_lat.astype('float32'))
ds_out["lat"].attrs['long_name'] = 'latidude'
#ds_out=ds_out.drop(["xc","yc"])
print(ds_out)

#regridder-----
#only need to run once
regridder = xe.Regridder(ds_in,ds_out,"patch") #special smooth iterpolator from this package
#regridder_bilinear = xe.Regridder(ds_in,ds_out,"bilinear")
#regridder_nearest = xe.Regridder(ds_in,ds_out,"nearest_s2d") #classical nearest


#for temporal interpolation-----
def interp1d_fun(x,tin,tout):
    f = interp1d(tin,x,axis=-1,kind='linear')
    return f(tout)

def xr_interp1d(x,tin,tout,idim,odim):
    #x: xarray with chunks
    #idim: input coordinate that will be changed by output odim
    #odim: output coordinate
    ds_interp1d = xr.apply_ufunc(
             interp1d_fun,x,
             input_core_dims=[[idim]],
             output_core_dims=[[odim]],
             output_dtypes=[np.float32],
             dask_gufunc_kwargs={'output_sizes':{odim:len(tout)}},
             kwargs={'tin':tin,'tout':tout}, #input to the above function
             dask='parallelized',
             #vectorize=True,
             )
    return ds_interp1d


#rotate wind from projection to model coordinates---
def projection_to_model_local_coords(x,y,ang=17*np.pi/180):
    return np.cos(ang)*x + np.sin(ang)*y, -np.sin(ang)*x + np.cos(ang)*y


#-----
files_wind=sorted(glob.glob(f'{dir_wind}/**/*.nc4',recursive=True))


for file_wind in tqdm(files_wind):

    year=int(str(file_wind)[-8:-4])
    print(year)

    
    #open wind data------
    dsw=xr.open_dataset(file_wind,chunks={'time':-1,'lon':-1,'lat':-1})[["u10","v10"]];dsw.close() #winds
    tw = dsw.time.values #contains data for the full 1st day of the next year
    #del these long attributes
    del dsw.attrs["history_of_appended_files"], dsw.attrs["history"]
  

    #spatial interpolation-----
    dsw_int = regridder(dsw)
  

    #temporal interpolation-----

    #first track of this year---
    month_sim=1
    file_track=f'tracks_{year}{month_sim:02d}_coords-xcyc_save-{savee}_deploy-{deploy}_Tsim-{minTsim}-{maxTsim}d.nc'
    file_track_path=f'{dir_tracks}{year}/{file_track}'  
    dst=xr.open_dataset(file_track_path)
    t0=dst.time.isel(traj=0,obs=0).values 
    x0=dst.x.isel(traj=range(npa_per_dep),obs=0).values
    y0=dst.y.isel(traj=range(npa_per_dep),obs=0).values
    dst.close(); del dst
    #
    #first track of the following year---
    if file_wind!=files_wind[-1]:
        file_track=f'tracks_{year+1}{month_sim:02d}_coords-xcyc_save-{savee}_deploy-{deploy}_Tsim-{minTsim}-{maxTsim}d.nc'
        file_track_path=f'{dir_tracks}{year+1}/{file_track}'  
        t1=xr.open_dataset(file_track_path).time.isel(traj=0,obs=0).values
    #last track of this year (for the final simulated month)---
    else: 
        #for the final year we can not open the next year simulation
        #we only have tracks until october, so we can get the wind for the last interval of displacement
        last_year_tracks=sorted(glob.glob(f'{dir_tracks}{year}/*.nc',recursive=True))
        end_month=len(last_year_tracks)
        file_track=f'tracks_{year}{end_month:02d}_coords-xcyc_save-{savee}_deploy-{deploy}_Tsim-{minTsim}-{maxTsim}d.nc'    
        file_track_path=f'{dir_tracks}{year}/{file_track}'  
        t1=xr.open_dataset(file_track_path).time.isel(traj=-1,obs=0).values + np.timedelta64(m2,'s')
    #
    #times to get wind data for this year---
    #however if we can not find a factor "nt_interp" of m2, use 10min
    #we wont have the same amount of interp data every m2, but it is better to skip 10min of 1 sample than 1h(original data)
    #nt_interp=283*2 #interpolate wind data every 9.43 min from 1h original data (factor of m2=44714)
    #t_interp:
    # - high reolution times to compute avg and std during the interval of net displacement
    # - the last data could be close to the beginning of next year, or the same year for the final month (October) of the simulation
    #t_dep: 
    # - times of displacement for the current year, referenced to the initial time of the m2 interval.
    t_interp=np.arange(t0,t1+np.timedelta64(1,'s'),nt_interp,dtype='datetime64[s]') 
    t_dep=np.arange(t0,t1,m2,dtype='datetime64[s]') #only for this year


    #1d interp----
    #reference with respect to ref_time (so convert timedelta64 to float)
    t_interp0=(t_interp-ref_time) / np.timedelta64(1,'s')  #dates after interpolation (factor of m2)
    tw0=(tw-ref_time) / np.timedelta64(1,'s') #dates of original winds (every 1h)
    #
    dsw_int=xr_interp1d(dsw_int.chunk({'time':-1,'xc':10,'yc':10}),tw0,t_interp0,idim='time',odim='time_int').transpose("time_int","yc","xc")
    #add time, xc and yc coords
    dsw_int.coords["time_int"]=t_interp
    dsw_int.coords["xc"] = ("xc",xcdis0.values.astype('float32')) #model coords in m
    dsw_int.coords["yc"] = ("yc",ycdis0.values.astype('float32'))
    

    #reshape with xarray---
    #
    #check time dimensions
    nt_interval=int(m2/nt_interp) #points in the m2 interval (right border of interval open)
    nt_dep=(len(t_interp)-1)//nt_interval #=len(t_dep), final shape after mean or std in the m2 interval. "-1" because we also don't consider the right border of the last interval in the avg
    #times after avg or std are referenced with the date of deployment (the begin of the m2 interval of the displacement)
    print("check times:",nt_interval,nt_dep,len(t_dep),nt_interval*nt_dep,len(dsw_int.time_int)-1)
    #
    #https://stackoverflow.com/questions/59504320/how-do-i-subdivide-refine-a-dimension-in-an-xarray-dataset
    #steps:
    # - assign_coords: create coords time_dep and time_interval 
    # - stack: create a coord and index called multi_time which is related to the original temporal size of the data,
    #          that now match a 2d-MultiIndex(nt_dep,nt_interval) which is defined using the new time_dep and time_interval coords,
    #          and will order the above coords keeping constant time_dep in every time_interval(0:78); which is consistent with how dsw_t_interp was created.
    # - reset_index().rename: del the the old time coord, and rename time index as multi_time to remove the old time index.
    # - unstack(): use the above 2d-MultiIndex to reshape the data original 1d time data into time_dep, time_interval,
    #              however, the new dimensions are send by default to the last index, 
    # - transpose: to fix above issue for the dimensions of variables, however, can not fix the order that dims are shown after Dimensions:
    #
    dsw_int=dsw_int.isel(time_int=slice(0,-1)
            ).assign_coords(time_dep=t_dep,time_interval=range(nt_interval)
            ).stack(multi_time=("time_dep","time_interval")
            ).reset_index("time_int",drop=True).rename(time_int="multi_time"
            ).unstack(dim="multi_time").transpose("time_dep","time_interval","yc","xc")
    dsw_int #still time in the last on the title of dimensions
    #
    #instead of above we could also try resample of xarray---
    #and then perform avg, std, but not working well
    #res=int(nt_interp+m2)
    #dsout_m2_avg=dsout.resample(time=f'{res}s',closed="right")#.mean(dim='time');
    #print(t_dep[:5])
    #for i in dsout_m2_avg: print(i)
    
    
    #rotate wind from projection to model coordinates---
    dsw_int["u10"],dsw_int["v10"]=projection_to_model_local_coords(dsw_int.u10,dsw_int.v10)

    
    #compute wind speed, direction,... (mean and std) based on Farrugia and Micallef (2017)---
    wd = np.arctan2(dsw_int.v10,dsw_int.u10) #wd for the interp times
    ws = (dsw_int.u10**2 + dsw_int.v10**2)**.5 #ws for the interp times
    u10_vec = dsw_int.u10.mean(dim='time_interval')
    v10_vec = dsw_int.v10.mean(dim='time_interval')
    #
    dsw_int["wd_mean"] = np.arctan2(v10_vec,u10_vec)
    dsw_int["ws_mean"] = ws.mean(dim='time_interval')
    dsw_int["ws_mean_vec"] = (u10_vec**2 + v10_vec**2)**.5 
    dsw_int["wd_std"] = ( (ws*(2*np.arctan(np.tan(0.5*(wd-dsw_int["wd_mean"]))))**2).mean(dim='time_interval') / dsw_int["ws_mean"] )**.5
    #use abs because there is 1 case with very small negative value -1e-7
    dsw_int["ws_std"] = ( abs(((ws*np.cos(wd-dsw_int["wd_mean"]))**2).mean(dim='time_interval') - dsw_int["ws_mean_vec"]**2) )**.5
    #
    #del u10 and v10
    del dsw_int["u10"], dsw_int["v10"], dsw_int["time_interval"]

    
    #call computations---
    dsw_int=dsw_int.compute()
    
    
    #save data---
    dsw_int=dsw_int.rename(time_dep="time") #rename dim time_dep
    #global coords and attrs---
    dsw_int["time"].attrs['description'] = 'initial date of the M2 interval of the net particle displacement'
    dsw_int["yc"].attrs['long_name'] = 'yc'
    dsw_int["yc"].attrs['description'] = 'the same as the net particle displacement grid y-axis'
    dsw_int["yc"].attrs['units'] = 'm'
    dsw_int["xc"].attrs['long_name'] = 'xc'
    dsw_int["xc"].attrs['description'] = 'the same as the net particle displacement grid x-axis'
    dsw_int["xc"].attrs['units'] = 'm'
    #
    dsw_int.attrs["spatial_info"] = "1) xESMF (method: patch) was used to interpolate wind components to the net displacement particle-grid (using lon-lat coords). 2) Then the wind was projected (rotated) to the local model axes."    
    dsw_int.attrs["temporal_info"] = f"Wind components were linearly interpolated to {nt_interp}s (factor of M2={m2}s), and then the avg and std in the M2 interval of the net displacement were computed."
    dsw_int.attrs["std of wind speed and direction"] = "Based on Farrugia and Micallef (2017)."
    #
    #variables---
    #
    dsw_int["wd_mean"].attrs['long_name'] = 'M2-mean wind direction'
    dsw_int["wd_mean"].attrs['units'] = 'radian'
    dsw_int["wd_mean"].attrs['description'] = 'Farrugia and Micallef (2017): eq(7)'
    #
    dsw_int["ws_mean"].attrs['long_name'] = 'M2-mean wind speed'
    dsw_int["ws_mean"].attrs['units'] = 'm/s'
    dsw_int["ws_mean"].attrs['description'] = 'eq(9)'
    #
    dsw_int["ws_mean_vec"].attrs['long_name'] = 'M2-mean wind speed with vectorial method'
    dsw_int["ws_mean_vec"].attrs['units'] = 'm/s'
    dsw_int["ws_mean_vec"].attrs['description'] = 'eq(8)'
    #
    dsw_int["wd_std"].attrs['long_name'] = 'M2-std of wind direction'
    dsw_int["wd_std"].attrs['units'] = 'radian'
    dsw_int["wd_std"].attrs['description'] = 'eq(18): square root of along wind variance'
    #
    dsw_int["ws_std"].attrs['long_name'] = 'M2-std of wind speed'
    dsw_int["ws_std"].attrs['units'] = 'm/s'
    dsw_int["ws_std"].attrs['description'] = 'eq(25)'

    #
    file_out_nc=f"{year}_{file_interp_wind_root}" 
    dir_out_nc=dir_post_proc_data+dir_interp_wind
    dsw_int.to_netcdf(dir_out_nc+file_out_nc)
    dsw_int.close(); del dsw_int; del dsw
    
    
client.close()