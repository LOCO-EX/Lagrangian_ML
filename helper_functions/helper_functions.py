import torch
import numpy as np
import xarray as xr


#standarization and normalization---
def standarization(var,fac=3):
    mean=np.nanmean(var)
    std=np.nanstd(var)*fac #using 3 times std (seems to works better than just 1std)
    var[np.isnan(var)]=0. #fill with 0 in case of nan. This is modifing our input array
    return ((var-mean)/std),mean,std #.astype('float32')

def de_standarization(var,mean,std):
    return (var*std+mean) #.astype('float32')

def min_max_normalization(var):
    minn=np.nanmin(var); maxx=np.nanmax(var)
    var[np.isnan(var)]=0. #fill with 0 in case of nan. This is modifing our input array
    return (var-minn)/(maxx-minn),minn,maxx #.astype('float32')

def de_min_max_normalization(var,minn,maxx):
    return  var*(maxx-minn)+minn #.astype('float32')

#loss functions with and without masking---
class initialization:
    def __init__(self, masking=False, mask=None):
        self.masking=masking
        self.mask=mask

        
#loss functions---
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
            return loss_val
        
    class mae(initialization):
        #we call this function without using its name
        def __call__(self, predict=torch.zeros(1), target=torch.zeros(1)):
            if self.masking:
                loss_val = torch.mean(abs(predict-target)[self.mask==1])
            else:
                #original---
                loss_val = torch.mean(abs(predict-target))  #=torch.nn.L1Loss() #MAE
            return loss_val

    class mse_numpy(initialization):
        #we call this function without using its name
        def __call__(self, predict=np.zeros(1), target=np.zeros(1)):
            if self.masking:
                loss_val = np.mean(((predict-target)[self.mask==1])**2)
            else:
                #original---
                loss_val = np.mean((predict-target)**2)  #=torch.nn.MSELoss()
            return loss_val
        
    class mae_numpy(initialization):
        #we call this function without using its name
        def __call__(self, predict=np.zeros(1), target=np.zeros(1)):
            if self.masking:
                loss_val = np.mean(abs(predict-target)[self.mask==1])
            else:
                #original---
                loss_val = np.mean(abs(predict-target)) #=torch.nn.L1Loss() #MAE
            return loss_val
        

#get times for backward propagation after passing to the model a subset of the full data---
def get_times_for_backward(nt,sequence_length_back=30):
    """
    Inputs:
    - nt = Total time steps of training data
    - sequence_length_back = Perform backward propagation after feeding the model with 
                            "sequence_length_back" sequential inputs,
                             (default sequence_length_back = 30).
    Output:
    t_backward = Time steps (every sequence_length_back) relative to t=0 to perform
                 backward propagation. In case nt/sequence_length_back is not integer, 
                 the last t_backward is set to nt-1.
    """
    if nt < sequence_length_back: sequence_length_back = nt  
    t_last = np.mod(nt,sequence_length_back) #remainder in case nt/sequence_length_back is not integer
    t_backward=np.arange(sequence_length_back,nt+1,sequence_length_back)-1
    #iterations = int(nt/sequence_length_back)
    #t_backward=np.arange(iterations)*sequence_length_back+sequence_length_back-1
    if t_backward[-1]!=nt-1: t_backward[-1]+=t_last #add the remainder to include the last time step
    return t_backward


#amount of model parameters (coefficients)---
def count_parameters(model):
    total_params = 0
    for i in range(len(model.hidden_channels)):
        total_params_layer = 0
        for name, parameter in getattr(model,f'cell{i}').named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            print(name,param)
            total_params+=param
            total_params_layer+=param
        print(f"Trainable params for Layer{i}: {total_params_layer}")
    print(f"Total Trainable params: {total_params}")


#save train, test predictions and loss at the end of simulation---
def save_data(case,time,y,x,lat,lon,output_channels,num_epochs,std_fac_dis,loss_name,loss,loss2,pred,dir_out_nc,file_out_nc,lr=None):

    dsout = xr.Dataset()
    #global coords and attrs---
    dsout.coords["time"] = time
    dsout["time"].attrs['description'] = f'times every m2'
    dsout.coords["y"] = y.astype('float32')
    dsout["y"].attrs['description'] = 'y-position in meter'
    dsout.coords["x"] = x.astype('float32')
    dsout["x"].attrs['description'] = 'x-position in meter'
    dsout.coords["lat"] = (("y","x"),lat.astype('float32'))
    dsout["lat"].attrs['long_name'] = 'latidude'
    dsout.coords["lon"] = (("y","x"),lon.astype('float32'))
    dsout["lon"].attrs['long_name'] = 'longitude'
    dsout.coords["out_channels"] = np.arange(output_channels)
    dsout.coords["epoch"] = np.arange(num_epochs)
    dsout.attrs["info"] = f"{case} data, created after training the model"
    #
    #variables---
    #
    dsout["predict"] = (("time","out_channels","y","x"),pred)
    dsout["predict"].attrs['long_name'] = 'Standarized prediction for the last epoch'
    dsout["predict"].attrs['units'] = ""
    #
    if loss_name=="mse":
        dsout["loss_mse"] = (("epoch"),loss)
        dsout["loss_mse"].attrs['long_name'] = f'Mean square error (MSE) Loss from standarized prediction'
        dsout["mae"] = (("epoch"),loss2)
        dsout["mae"].attrs['long_name'] = 'Mean absolute error (MAE) from standarized prediction'
    else:
        dsout["loss_mae"] = (("epoch"),loss)
        dsout["loss_mae"].attrs['long_name'] = f'Mean absolute error (MAE) Loss from standarized prediction'
        dsout["mse"] = (("epoch"),loss2)
        dsout["mse"].attrs['long_name'] = 'Mean square error (MSE) from standarized prediction'                                  
    #
    if case=="training":
        dsout["lr"] = (("epoch"),lr)
        dsout["lr"].attrs['long_name'] = 'Learning rate'
    #
    dsout["std_fac_predict"]=std_fac_dis
    dsout["std_fac_predict"].attrs['long_name'] = "factor that multiply the std of the tagert during its standarization"
    #dsout["std_fac_input_wind"]=std_fac_wind
    #
    dsout.to_netcdf(dir_out_nc+file_out_nc)
    dsout.close(); del dsout

    
#for checking loss during simulation--- 
def save_check_loss(case,nt,loss_name,epoch,num_epochs,loss,loss2,dir_out_nc,file_out_nc,lr=None,dsout=None):

    if epoch==0:
        dsout = xr.Dataset()
        dsout.coords["epoch"] = np.arange(num_epochs)
        dsout.attrs["info"] = f"loss saved to check {case} and testing"
        dsout["nt"]=nt
        dsout["nt"].attrs['long_name'] = "number of time steps"
        if loss_name=="mse":
            dsout["loss_mse"] = (("epoch"),loss)
            dsout["loss_mse"].attrs['long_name'] = f'Mean square error (MSE) Loss from standarized prediction'
            dsout["mae"] = (("epoch"),loss2)
            dsout["mae"].attrs['long_name'] = 'Mean absolute error (MAE) from standarized prediction'
        else:
            dsout["loss_mae"] = (("epoch"),loss)
            dsout["loss_mae"].attrs['long_name'] = f'Mean absolute error (MAE) Loss from standarized prediction'
            dsout["mse"] = (("epoch"),loss2)
            dsout["mse"].attrs['long_name'] = 'Mean square error (MSE) from standarized prediction'        
        if case=="training":
            dsout["lr"] = (("epoch"),lr)
            dsout["lr"].attrs['long_name'] = 'Learning rate'
        #
        return dsout
    else:
        if loss_name=="mse":
            dsout["loss_mse"].values = loss
            dsout["mae"].values = loss2
        else:
            dsout["loss_mae"].values = loss
            dsout["mse"].values = loss2
        if case=="training": dsout["lr"].values = lr
    #
    dsout.to_netcdf(dir_out_nc+file_out_nc)
    if epoch==num_epochs-1: dsout.close(); del dsout

