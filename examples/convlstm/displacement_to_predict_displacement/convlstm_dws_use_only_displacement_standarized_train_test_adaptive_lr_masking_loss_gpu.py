#- This simulation with gpu (with the below parameters) took 14h
#- In this experiment we also set lr from 0.01 to 0.0025
#  but here with masking is like the no masking case (exp2a-d) with 0.03 to 0.0075
#  thefactor of corecction is approx 3.
#  So: probably we should set the next time for masking case: lr=0.005-0.001

# ssh no100
# screen -S exp2e
# cd /export/lv4/user/jfajardourbina/dws_ulf_getm_2D_depth_avg/experiments_post_proc/lagrangian_simulation_36years/machine_learning_github/Lagrangian_ML/notebooks/convlstm
# conda activate phd_parcelsv221
# python3 convlstm_dws_exp2e_use_only_displacement_standarized_5std_train_test_adaptive_lr_masking_loss_gpu.py &
# to comeback: screen -r exp2e

import os
import sys
import numpy as np
import torch
import torch.nn.functional
from torch.autograd import Variable
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib as mpl
import glob
import xarray as xr
import dask as da
from tqdm import tqdm

# import convlstm
sys.path.append("../../../src")
import convlstm


#path to files---
homee = "/export/lv4/user/jfajardourbina/"
dir_post_proc_data=f"{homee}dws_ulf_getm_2D_depth_avg/experiments_post_proc/lagrangian_simulation_36years/machine_learning_github/Lagrangian_ML/post_proc_data/"
dir_displacement="net_displacement"
dir_interp_wind="wind"

#for output after train and test---
exp="exp2e"
dir_convlstm_model_out="ouput_convlstm_model_data/"
case_train="training"; file_out_train=f"{exp}_train.nc"
case_test="testing"; file_out_test=f"{exp}_test.nc"

#for plotting---
#dir_wind="{homee}dws_ulf_getm_2D_depth_avg/data/atmosphere/" #winds
dir_dws_bound=f"{homee}dws_ulf_getm_2D_depth_avg/experiments_post_proc/analysis_eulerian_data_36years/data_dws_boundaries/" #DWS boundarie with contour0
file_dws_bound0="dws_boundaries_contour0.nc"
file_grid=f"{homee}dws_ulf_getm_2D_depth_avg/experiments_post_proc/analysis_eulerian_data_36years/data_bathy_grid/" #topo data
#dir_vel = f'{homee}dws_ulf_getm_2D_depth_avg/data/velocity/' #vel data
#file_vel="RE.DWS200m.uvz.20090301.nc" #any vel file
#file_wind="UERRA.2009.nc4" #any wind file
#
#savee='everyM2' #saving track data every m2
#deploy='everyM2'#deploy set of particles every m2
#minTsim=60 #mimimum time of simulation (days)
#maxTsim=91 #maximum time of simulation (days)
#dir_tracks = f"{homee}dws_ulf_getm_2D_depth_avg/experiments_post_proc/lagrangian_simulation_36years/exp-deployHighVolume_coords-xcyc_save-{savee}_deploy-{deploy}_Tsim-{minTsim}-{maxTsim}d/tracks/"
#
#parameters
#npa_per_dep=12967 #number of particles per deployment
m2=int(12.42*3600+2) #period in seconds
#dx=400/1e3;dy=400/1e3 #particle grid reso


#Hyper-parameter of neural network---
run_model = True #True = run model; False = open results from old run
input_channels = 2 # number of input channels: u10,v10 wind
output_channels = 2 #number of output channels: dx, dy displacement
hidden_channels = [6, 3, output_channels] # the last digit is the output channel of each ConvLSTMCell (so we are using 3 layers)
#hidden_channels = [4, output_channels] # the last digit is the output channel of each ConvLSTMCell (so we are using 2 layers)
kernel_size = 3 #3, does not work with kernel=2
batch_size = 1 #not used in this tutorial
num_epochs = 3000
#learning parameters:
adaptive_learning=True  #False: lr=learning_rate;  True: lr=[learning_rate - learning_rate_end]
#learning_rate = 0.0025 ##too slow convergence if used since the beginning of simulation
learning_rate = 0.01 #initial lr
learning_rate_end=0.0025 #final lr
#
std_fac=5 #standarize using "std_fac" times the standard deviation
#
#if: hidden_channels = [6, 3, output_channels]
#the model will create 6GB of data in GPU memory after 400 training time steps
#so, after nt_steps=2000 (around 3y) we will exceed the mem limit of GPU (around 30GB)
#2.5years for training needs approx 26GB for the above model and with: input_channels = 2; output_channels = 2; kernel_size = 3
#this is because in every time step the graph of computations is stored in the cummulative lost (after calling the model), to perform then a backpropagation
#for this reason is sometimes important to use mini_batches and perform backpropagation after finish with 1.
#then use the next mini_batch and so on until using all the data and finishes 1 eppoch.


#use cuda if possible---
print ("Pytorch version {}".format(torch.__version__))
# check if CUDA is available
use_cuda = torch.cuda.is_available()
# use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device to be used for computation: {}".format(device))
print(f"{torch.cuda.get_device_name(0)}")


#open files----
#open DWS contours
dsb0=xr.open_dataset(dir_dws_bound+file_dws_bound0)
bdr_dws0=dsb0.bdr_dws.values #points that define DWS with contour0

#open topo file
dsto=xr.open_dataset(file_grid+"DWS200m.2012.v03.nc")
xct0=dsto.xc.min().values/1e3; yct0=dsto.yc.min().values/1e3 #=(0,0)
mask_topo=dsto.bathymetry.copy(); mask_topo=xr.where(np.isfinite(mask_topo),1,0) #mask ocean=1, land=0

#open net displacement files---
files_displacement=sorted(glob.glob(f'{dir_post_proc_data}{dir_displacement}/*.nc',recursive=True))
#files_displacement=files_displacement[29:31] #2009-2010
#concat all the files
dsdis=xr.open_mfdataset(files_displacement,concat_dim="time",parallel='True',chunks={'time': -1},
                      decode_cf=True, decode_times=True)#.load() #this are default decodes
                      #data_vars='minimal', coords='minimal', compat='override') #takes 1second more with this, see https://xarray.pydata.org/en/stable/io.html#reading-multi-file-datasets

#open interp files for wind---
files_interp_wind=sorted(glob.glob(f'{dir_post_proc_data}{dir_interp_wind}/*.nc',recursive=True))
#files_interp_wind=files_interp_wind[29:31]
#concat all the files
dswi=xr.open_mfdataset(files_interp_wind,concat_dim="time",parallel='True',chunks={'time': -1},
                      decode_cf=True, decode_times=True)#.load() #this are default decodes
                      #data_vars='minimal', coords='minimal', compat='override') #takes 1second more with this, see https://xarray.pydata.org/en/stable/io.html#reading-multi-file-datasets


#set training data---
#
#wind (input)
#dswi_train=dswi.sel(time=slice("2009-10-01","2009-12-31"),x=slice(60000,80000),y=slice(60000,70000))
#dswi_train=dswi_train.fillna(0) #fill nan with 0s in case displacement is in land (not neccesary for the above small domain)
#in_u10_train,in_v10_train=da.compute(dswi_train.u10.values.astype('float32'),dswi_train.v10.values.astype('float32'))
#
#displacement (output)
#dsdis_train=dsdis.sel(time=slice("2009-01-01","2012-12-31"))#,x=slice(70000,80000),y=slice(60000,70000))
#dsdis_train=dsdis.sel(time=slice("2009-10-01","2009-12-31"))#,x=slice(70000,80000),y=slice(60000,70000))
dsdis_train=dsdis.sel(time=slice("2009-06-01","2011-12-31"))#,x=slice(70000,80000),y=slice(60000,70000))
#dsdis_train=dsdis_train.fillna(0) #fill nan with 0s in case displacement is on land (not neccesary for the above small domain)
dx_train,dy_train=da.compute(dsdis_train.dx.values.astype('float32'),dsdis_train.dy.values.astype('float32'))
#
#use the past displacement to predict the displacement after M2=12.42h
in_u10_train,in_v10_train=dx_train[0:-1,...],dy_train[0:-1,...]
out_dx_train,out_dy_train=dx_train[1:,...],dy_train[1:,...]
#
del dx_train,dy_train
times_train=dsdis_train.time[0:-1].values
nt_train,ny,nx=out_dx_train.shape
print(times_train[[0,-1]],out_dx_train.shape)


#set testing data---
#
#wind (input)
#dswi_test=dswi.sel(time=slice("2009-10-01","2009-12-31"),x=slice(60000,80000),y=slice(60000,70000))
#dswi_test=dswi_test.fillna(0) #fill nan with 0s in case displacement is in land (not neccesary for the above small domain)
#in_u10_test,in_v10_test=da.compute(dswi_test.u10.values.astype('float32'),dswi_test.v10.values.astype('float32'))
#
#displacement (output)
#dsdis_test=dsdis.sel(time=slice("2013-01-01",None))#,x=slice(70000,80000),y=slice(60000,70000))
#dsdis_test=dsdis.sel(time=slice("2010-01-01","2010-03-31"))#,x=slice(70000,80000),y=slice(60000,70000))
dsdis_test=dsdis.sel(time=slice("2012-01-01",None))#,x=slice(70000,80000),y=slice(60000,70000))
#dsdis_test=dsdis_test.fillna(0) #fill nan with 0s in case displacement is on land (not neccesary for the above small domain)
dx_test,dy_test=da.compute(dsdis_test.dx.values.astype('float32'),dsdis_test.dy.values.astype('float32'))
#
#use the past displacement to predict the displacement after m2
in_u10_test,in_v10_test=dx_test[0:-1,...],dy_test[0:-1,...]
out_dx_test,out_dy_test=dx_test[1:,...],dy_test[1:,...]
#
del dx_test,dy_test
times_test=dsdis_test.time[0:-1].values
nt_test,ny,nx=out_dx_test.shape
print(times_test[[0,-1]],out_dx_test.shape)


#for plotting maps of predictions---
#mask: ocean=1, land=nan
mask=out_dx_train[0,...]*1.; mask[np.isfinite(mask)]=1.; mask[np.isnan(mask)]=np.nan
xx=dsdis_train.x/1e3; yy=dsdis_train.y/1e3; xx,yy=np.meshgrid(xx,yy)

#for masking values on land when computing loss---
mask_torch=torch.tensor(np.where(np.isnan(mask),0,1)[np.newaxis,np.newaxis,...]*np.ones((output_channels,ny,nx)))*1.
mask_numpy=mask_torch.numpy()*1.


def standarization(var,fac=3):
    mean=np.nanmean(var)
    std=np.nanstd(var)*fac #using 3 times std (seems to works better than just 1std)
    var[np.isnan(var)]=0. #fill with 0 in case of nan. This is modifing our input array
    return ((var-mean)/std),mean,std #.astype('float32')

def destandarization(var,mean,std):
    return (var*std+mean) #.astype('float32')


#standarization of data---

#training---
#inputs
in_u10_train, in_u10_mean_train, in_u10_std_train = standarization(in_u10_train,std_fac)
in_v10_train, in_v10_mean_train, in_v10_std_train = standarization(in_v10_train,std_fac)
#outputs
out_dx_train, out_dx_mean_train, out_dx_std_train = standarization(out_dx_train,std_fac)
out_dy_train, out_dy_mean_train, out_dy_std_train = standarization(out_dy_train,std_fac)
print("train info:")
print(f"steps={nt_train}; (ny,nx)=({ny},{nx})")
print("input approx output")
print(f"dx_mean, dx_std*{std_fac}, dy_mean, dy_std*{std_fac}:")
print(in_u10_mean_train, in_u10_std_train, in_v10_mean_train, in_v10_std_train)
print()

#testing---
#inputs
in_u10_test, in_u10_mean_test, in_u10_std_test = standarization(in_u10_test,std_fac)
in_v10_test, in_v10_mean_test, in_v10_std_test = standarization(in_v10_test,std_fac)
#outputs
out_dx_test, out_dx_mean_test, out_dx_std_test = standarization(out_dx_test,std_fac)
out_dy_test, out_dy_mean_test, out_dy_std_test = standarization(out_dy_test,std_fac)
print("test info:")
print(f"steps={nt_test}; (ny,nx)=({ny},{nx})")
print("input approx output")
print(f"dx_mean, dx_std*{std_fac}, dy_mean, dy_std*{std_fac}:")
print(in_u10_mean_test, in_u10_std_test, in_v10_mean_test, in_v10_std_test)


# initialize model---

if run_model:

    #loss functions with and without masking---
    class initialization:
        def __init__(self, masking=False, mask=None):
            self.masking=masking
            self.mask=mask

    class loss_function:

        class mse(initialization):
            #we call this function without using its name
            def __call__(self, predict=torch.zeros(1), target=torch.zeros(1)):
                if self.masking:
                    #masking land points---
                    #
                    #- the masking affect:
                    #    the value of the total loss (that only includes points inside DWS) and hence the last gradient of the backpropagation
                    #    loss=sum(prediction-output)**2/N; dlos/dpred=2*sum(prediction-output)/N,
                    #    with masking N is smaller because we dont consider land points, so seems that its like increasing the overall lr
                    #- similar effect to masking without using it:
                    #    if we use another custom loss like torch.nn.MSELoss(reduction='sum')
                    #    masking is irrelevant since we dont divide with N
                    #
                    #disregard land points (=0) for the mean, so the loss value will increase
                    #mask_torch: 0=land, 1=ocean
                    #however, because we only have particles inside DWS, mask_torch=0 for the land and all points outside DWS
                    loss_val = torch.mean(((predict-target)[self.mask==1])**2)
                else:
                    #original---
                    loss_val = torch.mean((predict-target)**2)  #=torch.nn.MSELoss()
                #
                return loss_val

        class mse_numpy(initialization):
            #we call this function without using its name
            def __call__(self, predict=np.zeros(1), target=np.zeros(1)):
                if self.masking:
                    #masking land points---
                    #disregard land points (=0) for the mean, so the loss value will increase
                    #probably because land points decrease the loss, the model don't perform so well
                    #mask_torch: 0=land, 1=ocean
                    #however, because we only have particles inside DWS, mask_torch=0 all points except inside it
                    loss_val = np.mean(((predict-target)[self.mask==1])**2)
                else:
                    #original---
                    loss_val = np.mean((predict-target)**2)  #=torch.nn.MSELoss()
                #
                return loss_val


    # initialize our model---
    model = convlstm.ConvLSTM(input_channels, hidden_channels, kernel_size).to(device)
    # choose loss function
    #loss_fn = torch.nn.MSELoss()
    #loss_fn = loss_function.mse() #for training (the same as above)
    masking=True
    loss_fn = loss_function.mse(masking=masking,mask=mask_torch) #for training (masking land points)
    #loss_fn_np = loss_function.mse_numpy() #for testing
    loss_fn_np = loss_function.mse_numpy(masking=masking,mask=mask_numpy) #for testing
    # choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # check the model / loss function and optimizer
    print(model)
    print(loss_fn.__class__.__name__) #this works for pytorch, but also for our custom class
    #print(loss_fn) #only works for pytorch
    print(optimizer)


def training(epoch,num_epochs,nt,model):

    predict=[]

    # Clear stored gradient
    model.zero_grad()
    # loop through all timesteps
    for t in range(nt):

        #stack data---
        #
        #old method using torch.autograd.Variable and .view()---
        #
        #data_in=np.stack((in_u10_train[t,...],in_v10_train[t,...]))
        #data_out=np.stack((out_dx_train[t,...],out_dy_train[t,...]))
        #data_in = torch.autograd.Variable(torch.Tensor(data_in).view(-1,input_channels,ny,nx)).to(device)
        #data_out = torch.autograd.Variable(torch.Tensor(data_out).view(-1,input_channels,ny,nx)).to(device)
        #
        #new method using  torch.tensor and np.newaxis (the same results as above)---
        data_in = torch.tensor(np.stack((in_u10_train[t,...],
                                         in_v10_train[t,...]),axis=0)[np.newaxis,...]).to(device)  #(1,input_channels,ny,nx)
        data_out = torch.tensor(np.stack((out_dx_train[t,...],
                                          out_dy_train[t,...]),axis=0)[np.newaxis,...]).to(device)  #(1,input_channels,ny,nx)

        # Forward process---
        #print(torch.cuda.memory_summary()) #check GPU memory usage
        predict0, _ = model(data_in, t) #data_in=(1,input_channels,ny,nx)  pred_y=(1,output_channels,ny,nx)
        #del data_in; torch.cuda.empty_cache() #to delete this variable from GPU memory

        # compute loss (and the cumulative loss from all the time steps)---
        if t == 0:
            loss0 = loss_fn(predict0, data_out) #data_out=(1,output_channels,ny,nx)
            if masking:
                mae0 = np.mean((abs(predict0-data_out).detach().cpu().numpy())[mask_numpy==1])
            else:
                mae0 = np.mean(abs(predict0-data_out).detach().cpu().numpy())
            #mape0=np.mean( abs((predict0-data_out)/data_out).detach().numpy() ) #problems with mape if denominator = 0
        else:
            loss0 += loss_fn(predict0, data_out)
            if masking:
                mae0 += np.mean((abs(predict0-data_out).detach().cpu().numpy())[mask_numpy==1])
            else:
                mae0 += np.mean(abs(predict0-data_out).detach().cpu().numpy())
            #mape0 += np.mean( abs((predict0-data_out)/data_out).detach().numpy() )
        #del data_out; torch.cuda.empty_cache()

    #cumulative loss from all the time steps (the loss we use for backward propagation)---
    if epoch % 50 == 0: print("Train epoch ", epoch, "; mean(MSE(t)) = ", loss0.item()/nt*std_fac**2, "; mean(MAE(t)) = ", mae0/nt*std_fac)

    # Zero out gradient, else they will accumulate between epochs---
    optimizer.zero_grad()

    # Backward pass---
    # Backward propagation is kicked off when we call .backward() on the error tensor
    # Autograd then calculates and stores the gradients for each model parameter in the parameter’s ".grad" attribute.
    loss0.backward()

    # Update parameters---
    optimizer.step() #to initiate gradient descent

    # Clear stored gradient---
    model.zero_grad()

    # save lr
    lr0=optimizer.param_groups[0]["lr"]

    #predict train data for the last epoch, after updating model parameters
    if epoch == num_epochs-1:
        with torch.no_grad():
            for t in range(nt):
                data_in = torch.from_numpy(np.stack((in_u10_train[t,...],
                                                     in_v10_train[t,...]),axis=0)[np.newaxis,...]).to(device)  #(1,input_channels,ny,nx)
                predict0, _ = model(data_in, t) #data_in=(1,input_channels,ny,nx)  predict=(1,output_channels,ny,nx)
                predict0 = predict0.detach().cpu().numpy()
                predict.append(predict0) #save the predictions for the last epoch
            predict=np.reshape(predict,(nt,output_channels,ny,nx)) #save to numpy array outputs (nt,output_channels,ny,nx)

    #print(optimizer.param_groups[0]['lr'])
    #
    # save the general checkpoint
    # torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss0.item()
    #             }, os.path.join(output_path,'Lagrangian_ML_training_checkpoint.pt'))
    # print("The checkpoint of the model and training status is saved.")

    return loss0.item()*1., mae0*1., predict, model, lr0


def testing(epoch,num_epochs,nt,model):
    #this function avoid gradient storage (there is memory increase with time in spite setting requires_grad=False)
    #https://discuss.pytorch.org/t/requires-grad-or-no-grad-in-prediction-phase/35759/2
    with torch.no_grad():

        predict=[]
        # loop through all timesteps
        for t in range(nt):

            # Forward process---
            #by default torch tensor: requires_grad=False---
            data_in = torch.tensor(np.stack((in_u10_test[t,...],
                                             in_v10_test[t,...]),axis=0)[np.newaxis,...],requires_grad=False).to(device)  #(1,input_channels,ny,nx)
            #
            predict0, _ = model(data_in, t) #data_in=(1,input_channels,ny,nx)  pred_y=(1,output_channels,ny,nx)
            predict0 = predict0.detach().cpu().numpy()

            # Compute loss (and the cumulative loss from all the time steps)---
            data_out = np.stack((out_dx_test[t,...],
                                 out_dy_test[t,...]),axis=0)[np.newaxis,...]  #(1,input_channels,ny,nx)
            if t == 0:
                #loss0 = np.mean((predict0-data_out)**2)
                loss0 = loss_fn_np(predict0, data_out) #MSE numpy loss with mask on land points
                if masking:
                    mae0 = np.mean((abs(predict0-data_out))[mask_numpy==1])
                else:
                    mae0 = np.mean(abs(predict0-data_out))
                #mape0=np.mean( abs((predict-data_out)/data_out).detach().numpy() ) #problems with mape if denominator = 0
            else:
                #loss0 += np.mean((predict0-data_out)**2)
                loss0 += loss_fn_np(predict0, data_out) #MSE numpy loss with mask on land points
                if masking:
                    mae0 += np.mean((abs(predict0-data_out))[mask_numpy==1])
                else:
                    mae0 += np.mean(abs(predict0-data_out))

        if epoch % 50 == 0: print("Test epoch ", epoch, "; mean(MSE(t)) = ", loss0/nt*std_fac**2, "; mean(MAE(t)) = ", mae0/nt*std_fac)

        #predict test data for the last epoch
        if epoch == num_epochs-1:
            for t in range(nt):
                data_in = torch.tensor(np.stack((in_u10_test[t,...],
                                                 in_v10_test[t,...]),axis=0)[np.newaxis,...],requires_grad=False).to(device)  #(1,input_channels,ny,nx)
                predict0, _ = model(data_in, t) #data_in=(1,input_channels,ny,nx)  predict=(1,output_channels,ny,nx)
                predict0 = predict0.detach().cpu().numpy()
                predict.append(predict0) #save the predictions for the last epoch
            predict=np.reshape(predict,(nt,output_channels,ny,nx)) #save to numpy array outputs (nt,output_channels,ny,nx)

        return loss0*1., mae0*1., predict

#run simulation----
if run_model:

    #training output data
    lr = np.zeros(num_epochs)
    loss_train = np.zeros(num_epochs) # save for every epoch the sum of loss for all the time steps for training data
    mae_train = np.zeros(num_epochs) #save for every epoch the sum of the mean absolute error
    #problems with mape if denominator = 0
    #mape = np.zeros(num_epochs) #save for every epoch the sum of the mean absolute percentage error

    #testing output data
    loss_test = np.zeros(num_epochs) # save for every epoch the sum of loss for all the time steps for training data
    mae_test = np.zeros(num_epochs) #save for every epoch the sum of the mean absolute error

    #check memory usage of GPU with
    # torch.cuda.memory_reserved()/1e9 #GB  (this is the important one)
    # torch.cuda.memory_allocated()/1e6 #MB
    # print(torch.cuda.memory_summary()) #More info

    #adaptive learning rate
    if adaptive_learning:
        #lrend = lrini*gamma^(epochs-1)
        #gamma = (lrend/lrini)^(1/(epochs-1))
        gamma=(learning_rate_end/learning_rate)**(1/(num_epochs-1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    #epochs
    for epoch in tqdm(range(num_epochs)):
        #training
        loss_train[epoch], mae_train[epoch], predict_train, model, lr[epoch] = training(epoch,num_epochs,nt_train,model)
        #testing
        loss_test[epoch], mae_test[epoch], predict_test = testing(epoch,num_epochs,nt_test,model)
        #adaptive lr
        if adaptive_learning: scheduler.step()


#save above data---
def save_data(case,time,loss,mae,pred,dir_out_nc,file_out_nc,lr=None):

    dsout = xr.Dataset()
    #global coords and attrs---
    dsout.coords["time"] = time
    dsout["time"].attrs['description'] = f'times every m2'
    dsout.coords["y"] = dsdis_train.y
    dsout["y"].attrs['description'] = 'y-position in meter'
    dsout.coords["x"] = dsdis_train.x
    dsout["x"].attrs['description'] = 'x-position in meter'
    dsout.coords["out_channels"] = np.arange(output_channels)
    dsout.coords["epoch"] = np.arange(num_epochs)
    #
    dsout.attrs["info"] = f"Data created after {case}"
    #
    #variables---
    #
    dsout["predict"] = (("time","out_channels","y","x"),pred)
    dsout["predict"].attrs['long_name'] = 'Standarized prediction for the last epoch'
    dsout["predict"].attrs['units'] = ""
    #
    dsout["loss"] = (("epoch"),loss)
    dsout["loss"].attrs['long_name'] = 'Loss = MSE from standarized prediction'
    #
    dsout["mae"] = (("epoch"),mae)
    dsout["mae"].attrs['long_name'] = 'Mean Absolute error = MAE from standarized prediction'
    #
    if case=="training":
        dsout["lr"] = (("epoch"),lr)
        dsout["lr"].attrs['long_name'] = 'Learning rate'
    #
    dsout.to_netcdf(dir_out_nc+file_out_nc)
    dsout.close(); del dsout
#
if run_model:
    #train
    save_data(case_train,times_train,loss_train,mae_train,predict_train,dir_convlstm_model_out,file_out_train,lr)
    #test
    save_data(case_test,times_test,loss_test,mae_test,predict_test,dir_convlstm_model_out,file_out_test)


#open data---
#dsout_train=xr.open_dataset(dir_convlstm_model_out+file_out_train)
#dsout_test=xr.open_dataset(dir_convlstm_model_out+file_out_test)
#
#loss_train=dsout_train.loss.values; mae_train=dsout_train.mae.values; predict_train=dsout_train.predict.values
#loss_test=dsout_test.loss.values; mae_test=dsout_test.mae.values; predict_test=dsout_test.predict.values

#epochs=np.arange(num_epochs)+1

