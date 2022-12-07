# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:35:55 2021

@author: Mxm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timeit import default_timer
from Adam import Adam
import sys
#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
################################################

def compl_mul2d(a, b):
    # (batch, in_channel, x,t), (in_channel, out_channel, x,t) -> (batch, out_channel, x,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        
        batchsize = x.shape[0]  
        size1 = x.shape[-2]     
        size2 = x.shape[-1]     
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])   

        if gridy is None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
            
            out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        
        return x

# The structure of T-FNO
class TemperatureFNO2d(nn.Module):
    def __init__(self, If_Res, modes1, modes2,  width):
        super(TemperatureFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.ifRes = If_Res
       
        self.padding = 9 
        self.fc0 = nn.Linear(1, self.width) 

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        x_in = x
        x_in_ft = torch.fft.rfftn(x_in, dim=[-3,-2])
        x_in_ft[:, self.modes1:, self.modes2:, :] = 0
        x_ifft = torch.fft.irfftn(x_in_ft, s=(x.size(-3), x.size(-2)),dim=(-3, -2))
        x_in = x_ifft
    
        x = self.fc0(x)             
        x = x.permute(0, 3, 1, 2)   
        
        x = F.pad(x, [0,self.padding, 0,self.padding]) # 

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)     
        
        #Residual mapping
        if self.ifRes:
            x = x  + x_in
            
        return x
  
# The structure of α-FNO
class AlphaFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        
        super(AlphaFNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) 

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.BN1 = nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        # incremental treatment for cure cycles
        x_ini = self.incremental_x(x,x.device)
        
        x = self.fc0(x_ini)             
        x = x.permute(0, 3, 1, 2)   
        x = F.pad(x, [0,self.padding, 0,self.padding]) 

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.tanh(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.BN1(x)
        x = torch.tanh(x)
        
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)   
        out = self.activation_alpha(x)
        
        return out
    
    def incremental_x(self,x,device):
        size_t = x.shape[2]
        if size_t%2==0:
            half_x = x[:,:,0:int(size_t/2),:]
        else:
            half_x = x[:,:,0:int(size_t/2)+1,:]
            
        grid_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        grid_x[:,:,0::2,:] = half_x
        grid_x[:,:,1:-1:2,:] = 0.5*(half_x[:,:,0:-1:1,:]+half_x[:,:,1::1,:])
        return grid_x.to(device)
                
    def activation_alpha(self,x):
        initial_alpha = 0.05
        out = initial_alpha+(1-initial_alpha)*self.sig(x)
        return out       


def train_PINO(If_Res, ntrain, Train_Data, Test_Data, hyper_paras):
    
    print('Start training!')
    modes = 16
    width = 32

    train_dataA = Train_Data['dataA']
    train_dataT = Train_Data['dataT']
    train_dataTair = Train_Data['dataTair']
    
    train_dt = hyper_paras['train_dt']
    test_dt = 4 # the time interval of test data
    
    curing_time = train_dataT.shape[2]*4
    thickness = 0.05
    
    x_total = thickness
    t_total = curing_time
    
    #set the input and output of training process
    dataInput = np.repeat((train_dataTair.reshape(train_dataTair.shape[0],1,train_dataTair.shape[1])), train_dataA.shape[1], axis=1)
    T_train_input = dataInput[0:ntrain,::1,::int(train_dt/test_dt)]
    
    A_dataInput_normalizer = GaussianNormalizer(dataInput)
    A_dataInput = A_dataInput_normalizer.encode(dataInput)
    A_train_input = A_dataInput[0:ntrain,::1,::int(train_dt/test_dt)]
    
    train_A = train_dataA[0:ntrain,:,::int(train_dt/test_dt)]
    train_T = train_dataT[0:ntrain,:,::int(train_dt/test_dt)]

    train_nx = T_train_input.shape[1]
    train_nt = T_train_input.shape[2]
    train_dx = x_total/(train_nx-1)
    train_dt = train_dt

    #set the input and output of testing process
    test_dataA = Test_Data['dataA']
    test_dataT = Test_Data['dataT']
    test_dataTair = Test_Data['dataTair']
    
    test_dataInput = np.repeat((test_dataTair.reshape(test_dataTair.shape[0],1,test_dataTair.shape[1])), test_dataA.shape[1], axis=1)
    T_test_input = test_dataInput[:,:,:]
    
    test_A_dataInput = A_dataInput_normalizer.encode(test_dataInput)
    A_test_input = test_A_dataInput[:,:,:]
    test_A = test_dataA[:,:,:]
    test_T = test_dataT[:,:,:]
    
    ntest = T_test_input.shape[0]
    test_nx = test_A.shape[1]
    test_nt = test_A.shape[2]
    
    alpha_data_train = A_train_input
    temperature_data_train  = T_train_input
    alpha_data_test = A_test_input
    temperature_data_test  = T_test_input
    
    # load the hyperparameters list
    iterations = hyper_paras['interation']
    A_epochs = hyper_paras['A_epochs']
    T_epochs = hyper_paras['T_epochs']
    A_lr = hyper_paras['A_lr']
    T_lr = hyper_paras['T_lr']
    batch_size = hyper_paras['Batch_size']
    
    A_step_size = hyper_paras['A_step_size']
    A_gamma = hyper_paras['A_gamma']
    T_step_size = hyper_paras['T_step_size']
    T_gamma = hyper_paras['T_gamma']
    
    W_alpha_ini = hyper_paras['alpha_ini']
    W_alpha_ode = hyper_paras['alpha_ode']
    W_Temperature_ini = hyper_paras['Temperature_ini']
    W_Temperature_upbc = hyper_paras['Temperature_upbc']
    W_Temperature_downbc = hyper_paras['Temperature_downbc']
    W_Temperature_pde = hyper_paras['Temperature_pde']
 
    #-----------Temperature prediction-----------
    temperature_data_train = torch.from_numpy(temperature_data_train[:,::1].astype(np.float32)).cuda()
    temperature_data_test = torch.from_numpy(temperature_data_test[:,::1].astype(np.float32)).cuda()

    temperature_x_train = temperature_data_train.reshape(ntrain,train_nx,train_nt,1)
    temperature_y_train = torch.from_numpy(train_T[:,:,::1].astype(np.float32)).cuda()
    
    temperature_x_test = temperature_data_test.reshape(ntest,test_nx,test_nt,1)
    temperature_y_test = torch.ones((ntest,test_nx,test_nt,1))
    # dataloader for traning
    temperature_sequential_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(temperature_x_train, temperature_y_train), batch_size=batch_size, shuffle=False)
    # dataloader for testing
    temperature_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(temperature_x_test, temperature_y_test), batch_size=batch_size, shuffle=False)
    
    T_model = TemperatureFNO2d(If_Res, modes, modes, width).cuda()
    T_optimizer = Adam(T_model.parameters(), lr=T_lr, weight_decay=1e-4)
    T_scheduler = torch.optim.lr_scheduler.StepLR(T_optimizer, step_size=T_step_size, gamma=T_gamma) 
    
    #-----------DoC prediction-----------
    alpha_data_train = torch.from_numpy(alpha_data_train[:,::1].astype(np.float32)).cuda()
    alpha_data_test = torch.from_numpy(alpha_data_test[:,::1].astype(np.float32)).cuda()
    
    alpha_x_train = alpha_data_train.reshape(ntrain,train_nx,train_nt,1)
    alpha_y_train = torch.from_numpy(train_A[:,:,::1].astype(np.float32)).cuda()

    alpha_x_test = alpha_data_test.reshape(ntest,test_nx,test_nt,1)
    alpha_y_test = torch.from_numpy(test_A[:,::1].astype(np.float32)).cuda()
    # dataloader for traning
    alpha_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(alpha_x_train, alpha_y_train, temperature_x_train), batch_size=batch_size, shuffle=True)
    alpha_sequential_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(alpha_x_train, alpha_y_train), batch_size=batch_size, shuffle=False)
    # dataloader for testing
    alpha_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(alpha_x_test, alpha_y_test), batch_size=batch_size, shuffle=False)
    
    A_model = AlphaFNO2d(modes, modes, width).cuda()
    A_optimizer = Adam(A_model.parameters(), lr=A_lr, weight_decay=1e-4)
    A_scheduler = torch.optim.lr_scheduler.StepLR(A_optimizer, step_size=A_step_size, gamma=A_gamma) 
    
    t1 = default_timer()

    T_ini_loss = np.zeros((T_epochs*iterations))
    T_upbc_loss = np.zeros((T_epochs*iterations))
    T_downbc_loss = np.zeros((T_epochs*iterations))
    T_pde_loss = np.zeros((T_epochs*iterations))
    T_data_loss = np.zeros((T_epochs*iterations))
    
    A_ini_loss = np.zeros((A_epochs*iterations))
    A_ode_loss = np.zeros((A_epochs*iterations))
    A_data_loss = np.zeros((A_epochs*iterations))
    
    for i in range(iterations):
        for eps in range(A_epochs):
            A_model.train()
            A_train_ini = 0
            A_train_ode = 0
            A_train_data = 0
            for x,y,f in alpha_train_loader:
                x, y, f= x.cuda(), y.cuda(),f.cuda()
                A_optimizer.zero_grad()
                A_output = A_model.forward(x).reshape(batch_size,train_nx,train_nt)
                print('---------Alpha Alpha Alpha Alpha---------')
                print("Iteration:{0}, Alpha__Epoch:{1}".format(i,eps))
                
                A_Ini_loss = alpha_initial_loss(A_output)*W_alpha_ini
                A_train_ini+= A_Ini_loss.item()
                print('Alpha_Initial_loss:',A_Ini_loss.data.cpu().numpy())
                    
                out_T = f.reshape(batch_size,train_nx,train_nt)
                A_Ode_loss = alpha_ode_loss(A_output,out_T,train_dt)*W_alpha_ode
                A_train_ode += A_Ode_loss.item()
                print('Alpha_ODE_loss:',A_Ode_loss.data.cpu().numpy())
                print('\n')
                    
                Alpha_Loss = A_Ini_loss + A_Ode_loss
                Alpha_Loss.backward()
                A_optimizer.step()
            
            A_scheduler.step()

            A_train_ini = A_train_ini/(ntrain/batch_size)
            A_train_ode = A_train_ode/(ntrain/batch_size)
            A_train_data = A_train_data/(ntrain/batch_size)
            
            A_ini_loss[i*A_epochs+eps] = A_train_ini
            A_ode_loss[i*A_epochs+eps] = A_train_ode   
            A_data_loss[i*A_epochs+eps] = A_train_data
            
        A_model.eval()
        pre_alpha = torch.ones((ntrain,train_nx,train_nt,1))
        with torch.no_grad():
            for step, (x,y) in enumerate(alpha_sequential_loader):
                out = A_model.forward(x)
                pre_alpha[step*batch_size:(step+1)*batch_size,:,:,:] = out
        
        pre_alpha = pre_alpha.cuda()
        temperature_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(temperature_x_train, temperature_y_train, pre_alpha), batch_size=batch_size, shuffle=True)
        
        for eps in range(T_epochs):
            
            T_model.train()
            T_train_ini = 0
            T_train_bc_up  = 0
            T_train_bc_down  = 0
            T_train_pde = 0
            T_train_data = 0
            
            for x, y, alpha in temperature_train_loader:
                x, y, alpha  = x.cuda(), y.cuda(), alpha.cuda()
                
                T_optimizer.zero_grad()
                T_output = T_model.forward(x).reshape(batch_size,train_nx,train_nt)
                print('---------Tem Tem Tem Tem---------')
                print("Iteration:{0}, Temperature__Epoch:{1}".format(i,eps))
                
                T_Ini_loss = initial_loss(T_output)*W_Temperature_ini
                T_train_ini += T_Ini_loss.item()
                print('Temperature_Initial_loss:',T_Ini_loss.data.cpu().numpy())
                    
                T_BC_loss_up = up_boundary_loss(T_output,x)*W_Temperature_upbc
                T_train_bc_up += T_BC_loss_up.item()
                print('Temperature_upBC_loss_up:',T_BC_loss_up.data.cpu().numpy())
    
                T_BC_loss_down = down_boundary_loss(T_output,x)*W_Temperature_downbc
                T_train_bc_down += T_BC_loss_down.item()
                print('Temperature_downBC_loss:',T_BC_loss_down.data.cpu().numpy())
                    
                alpha_output = alpha.reshape(batch_size,train_nx,train_nt)
                T_PDE_loss = pde_loss(T_output,alpha_output,train_nx,train_dx,train_nt,train_dt)*W_Temperature_pde
                T_train_pde += T_PDE_loss.item()
                print('Tempreature_PDE_loss:',T_PDE_loss.data.cpu().numpy())
                                    
                print('\n')

                Temperature_Loss = T_Ini_loss + T_BC_loss_up +T_BC_loss_down + T_PDE_loss
                Temperature_Loss.backward()
                T_optimizer.step()
            
            T_scheduler.step()
            
            T_train_ini = T_train_ini/(ntrain/batch_size)
            T_train_bc_up = T_train_bc_up/(ntrain/batch_size)
            T_train_bc_down = T_train_bc_down/(ntrain/batch_size)
            T_train_pde = T_train_pde/(ntrain/batch_size)
            T_train_data = T_train_data/(ntrain/batch_size)
            
            T_ini_loss[i*T_epochs+eps] = T_train_ini
            T_upbc_loss[i*T_epochs+eps] = T_train_bc_up
            T_downbc_loss[i*T_epochs+eps] = T_train_bc_down
            T_pde_loss[i*T_epochs+eps] = T_train_pde
            T_data_loss[i*A_epochs+eps] = T_train_data
            
        percent = float(i*eps)/float(T_epochs*iterations)
        if(i%5==0):
            print("%.4f"%percent,'%')
        T_model.eval()
        pre_temp = torch.ones((ntrain,train_nx,train_nt,1))
        with torch.no_grad():
            for step, (x,y) in enumerate(temperature_sequential_loader):
                out = T_model.forward(x)
                pre_temp[step*batch_size:(step+1)*batch_size,:,:,:] = out    
            pre_temp = pre_temp.cuda()   
            alpha_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(alpha_x_train, alpha_y_train,pre_temp), batch_size=batch_size, shuffle=True)
    
    t2 = default_timer()
    total_time = np.round(t2-t1, 5)
    print('The total training time:',total_time)
        
    A_model.eval()
    test_alpha = torch.ones((ntest,test_nx, test_nt))
    with torch.no_grad():
        for step, (x,y) in enumerate(alpha_test_loader):
            x, y = x.cuda(), y.cuda()
            out_A = A_model(x).reshape(batch_size, test_nx, test_nt)
            test_alpha[step*batch_size:(step+1)*batch_size,:,:] = out_A
        test_alpha = test_alpha.cuda()
        alpha_error = np.abs(test_alpha.cpu().data.numpy()-test_A[:,:,:])
        
    T_model.eval()
    test_temp = torch.ones((ntest,test_nx, test_nt))
    with torch.no_grad():
        for step, (x,y) in enumerate(temperature_test_loader):
            x, y = x.cuda(), y.cuda()
            out_T = T_model(x).reshape(batch_size, test_nx, test_nt)
            test_temp[step*batch_size:(step+1)*batch_size,:,:] = out_T
        test_temp = test_temp.cuda()
        Temperature_error = np.abs(test_temp.cpu().data.numpy()-test_T[:,:,:])
            
    loss_dict = {'A_ini_loss':A_ini_loss,
                'A_ode_loss' :A_ode_loss,
                'A_data_loss':A_data_loss,
                'T_ini_loss':T_ini_loss,
                'T_upbc_loss':T_upbc_loss,
                'T_downbc_loss':T_downbc_loss,
                'T_pde_loss':T_pde_loss,
                'T_data_loss':T_data_loss,
                'Temperature_error':Temperature_error,
                'Alpha_error':alpha_error,
                'total_time':total_time}
    
    prediction_dict = {'A_output':test_alpha.cpu().data.numpy(),'T_output':test_temp.cpu().data.numpy()}
    
    return loss_dict, prediction_dict

def initial_loss(out_T):
    
    batch_size = out_T.shape[0]
    x_size = out_T.shape[1]
    initial_true= (torch.ones((batch_size,x_size))*293).cuda()
    loss = F.mse_loss(out_T[:,:,0].view(batch_size,-1), initial_true.view(batch_size,-1))      
    return loss

def down_boundary_loss(out_T,x):
    
    batch_size = out_T.shape[0]
    x_size = out_T.shape[1]
    t_size = out_T.shape[2]
    dataInput = x.reshape(batch_size,x_size,t_size)
    bc_true = dataInput[:,0,:].cuda()
    loss = F.mse_loss(out_T[:,0,:].view(batch_size,-1),bc_true.view(batch_size,-1)) 

    return loss

def up_boundary_loss(out_T,x):
    
    batch_size = out_T.shape[0]
    x_size = out_T.shape[1]
    t_size = out_T.shape[2]
    dataInput = x.reshape(batch_size,x_size,t_size)
    bc_true = dataInput[:,0,:].cuda()
    loss = F.mse_loss(out_T[:,-1,:].view(batch_size,-1),bc_true.view(batch_size,-1))

    return loss
    
def pde_loss(out_T, out_alpha, nx, dx, nt, dt):

    #Material properties for AS4 fibre, 8552 epoxy and Invar tool
    rho_c = 1581.26
    Cp_c = 1080.225
    k_c = 0.6386  # composites thermal conductivity (W/(m K)) 
    
    rho_r = 1.300e3
    H_r = 5.4e5
    Vr = 0.426

    rho_t = 8150
    Cp_t = 510
    k_t = 13  # tool thermal conductivity (W/m K)
    
    batch_size = out_T.shape[0]
    x_size = out_T.shape[1]
    t_size = out_T.shape[2]
    T_output = out_T.reshape(batch_size, x_size, t_size)
        
    a = torch.cat((torch.ones(int(0.4*nx),nt)*rho_t*Cp_t, torch.ones(nx-int(0.4*nx),nt)*rho_c*Cp_c))
    b = torch.cat((torch.ones(int(0.4*nx),nt)*k_t, torch.ones(nx-int(0.4*nx),nt)*k_c))
    c = torch.cat((torch.ones(int(0.4*nx),nt)*0, torch.ones(nx-int(0.4*nx),nt)*Vr*H_r*rho_r))
    a = a.cuda()
    b = b.cuda()
    c = c.cuda()
      
    T_t = (T_output[:,1:-1,1:]-T_output[:,1:-1,0:-1])/dt    
    T_xx = ((b[1:-1,1:]+b[2:,1:])*(T_output[:,2:,1:] - T_output[:,1:-1,1:]) - (b[1:-1,1:]+b[0:-2,1:])*(T_output[:,1:-1,1:] - T_output[:,0:-2,1:]))/2/dx/dx
    dadt = (c[1:-1,1:]+c[2:,1:])/2*(out_alpha[:,1:-1,1:] - out_alpha[:,1:-1,0:-1])/dt
    
    equation_left = T_t*a[1:-1,1:]
    equation_right = T_xx + dadt
    loss = F.mse_loss(equation_left, equation_right)

    return loss

def alpha_initial_loss(out_A):

    batch_size = out_A.shape[0]
    x_size = out_A.shape[1]
    initial_true = (torch.ones((batch_size,x_size))*float(0.05000)).cuda()
    loss = F.mse_loss(out_A[:,:,0].view(batch_size,-1), initial_true.view(batch_size,-1))

    return loss

def alpha_ode_loss(out_A, out_T, dt):
  
    batch_size = out_A.shape[0]   
    dalpha_dt = ((out_A[:,:,1:] - out_A[:,:,0:-1])/dt)
    alpha_kinetics = alpha_RHSTOT_func(out_T, out_A)
    alpha_kinetics = torch.from_numpy(alpha_kinetics).cuda()
    loss = F.mse_loss(dalpha_dt.reshape(batch_size,-1),alpha_kinetics[:,:,1:].reshape(batch_size,-1))
  
    return loss
    
def alpha_RHSTOT_func(out_T, out_alpha):
    
    # cure kinetics properties, 8552 epoxy resin
    A = 1.528e5      # (1/s) RAVEN Material Model
    dE = 6.650e4     # (J/mol) RAVEN Material Model
    M = 0.8129       # RAVEN Model
    N = 2.7360       # RAVEN Model
    C = 43.09        # RAVEN Model
    ALCT = 5.475e-3  # (1/K) RAVEN Model
    ALC = -1.6840    # RAVEN Model
    R = 8.314        # (J/(mol K))
    
    out_T = out_T.detach().cpu().numpy()
    out_alpha = out_alpha.detach().cpu().numpy()
    
    K = A * np.exp(-dE/(R*out_T))    
    alpha_RHSTOT = (K * (out_alpha)**M) * ((1.0-out_alpha)**N) / (1+np.exp(C*(out_alpha-ALC-ALCT*out_T)))

    return alpha_RHSTOT    
     
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = np.mean(x)
        self.std = np.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()    
    
