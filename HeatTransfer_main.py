# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 17:13:25 2021

@author: Mxm
"""

import torch 
import numpy as np
import random
import matplotlib.pyplot as plt
from HeatTransfer_util import train_PINO
import seaborn as sns
import scipy.io as sio
import os
        
if __name__ == '__main__':
    
    Train_Data = sio.loadmat('data/PINO_twohold_1.mat')
    Test_Data = sio.loadmat('data/PINO_twohold_1.mat')
    curing_time = 350*60
    thickness = 0.05
    #---------------------------------------
    train_dt = 4
    interation = 30
    A_epochs = 40
    T_epochs = 60
    Batch_size = 1
    
    A_lr = 0.001
    T_lr = 0.001
    A_step_size = 40
    T_step_size = 60
    A_gamma = 0.95
    T_gamma = 0.95
    
    alpha_ini = 1e2
    alpha_ode = 1e8
    
    Temperature_ini = 1e-1
    Temperature_upbc = 1e-1
    Temperature_downbc = 1e-1
    Temperature_pde = 1e-8

    #---------------------------------------
    hyper_paras = {'interation':interation,
                   'train_dt':train_dt,
                'A_epochs' :A_epochs,
                'T_epochs':T_epochs,
                'A_lr':A_lr,
                'T_lr':T_lr,
                'Batch_size':Batch_size,
                'A_step_size':A_step_size,
                'A_gamma':A_gamma,
                'T_step_size':T_step_size,
                'T_gamma':T_gamma,
                'alpha_ini' :alpha_ini,
                'alpha_ode':alpha_ode,
                'Temperature_ini':Temperature_ini,
                'Temperature_upbc':Temperature_upbc,
                'Temperature_downbc':Temperature_downbc,
                'Temperature_pde':Temperature_pde}
    for i in range(1):
        ntrain = 1
        If_Res = 1
        loss_dict, prediction_dict = train_PINO(If_Res, ntrain, Train_Data, Test_Data, hyper_paras)
        
        doc_name = 'Results/PINO_'+'ntrain_'+str(ntrain)+'_interation_'+str(interation)+'_dt_'+str(train_dt)
        os.makedirs(doc_name,exist_ok=True)
            
        sio.savemat(doc_name+'/PINO_results.mat', mdict={'loss_dict': loss_dict, 'prediction_dict':prediction_dict,'hyper_paras':hyper_paras})

# In[]        
        Temperature_error = loss_dict['Temperature_error']
        Alpha_error = loss_dict['Alpha_error']
        
        A_ini_loss = loss_dict['A_ini_loss']
        A_ode_loss = loss_dict['A_ode_loss']
        A_data_loss = loss_dict['A_data_loss']
        T_ini_loss = loss_dict['T_ini_loss']
        T_downbc_loss = loss_dict['T_downbc_loss']
        T_upbc_loss = loss_dict['T_upbc_loss']
        T_pde_loss = loss_dict['T_pde_loss']
        T_data_loss = loss_dict['T_data_loss']
        Temperature_error = loss_dict['Temperature_error']
        Alpha_error = loss_dict['Alpha_error']
        
        test_A_pre = prediction_dict['A_output']
        test_T_pre = prediction_dict['T_output']
        
        dataA = Test_Data['dataA']
        dataT = Test_Data['dataT']
        dataTair = Test_Data['dataTair']
        test_A = dataA[:,:,:]
        test_T = dataT[:,:,:]
     
        test_nx = dataA.shape[1]
        test_nt = dataA.shape[2]
        dataInput = np.repeat((dataTair.reshape(dataTair.shape[0],1,dataTair.shape[1])), dataA.shape[1], axis=1)
        test_input = dataInput[:,:,:]


        if 1:
            
            T_maximum = np.max(np.abs(Temperature_error[:,:,0:int(0.95*test_nt)]),axis=(1,2))
            T_mean = np.mean(np.abs(Temperature_error[:,:,:]),axis=(1,2))
                        
            print('The maximum absolute error of T:',np.round(np.mean(T_maximum[:]),3),'K')
            print('The mean absolute error of T:',np.round(np.mean(T_mean[:]),3),'K')
            print('The relative percentage error of T:',np.round(np.mean(Temperature_error[:,:,0:int(0.95*test_nt)]/test_T[:,:,0:int(0.95*test_nt)]),5)*100,"%")
            print('--------------------------------------')
                   
            A_maximum = np.max(np.abs(Alpha_error[:,int(0.4*test_nx):,:]),axis=(1,2))
            A_mean = np.mean(np.abs(Alpha_error[:,int(0.4*test_nx):,:]),axis=(1,2))
            
            print('The maximum absolute error of α:',np.round(np.mean(A_maximum),3))
            print('The mean absolute error of α:',np.round(np.mean(A_mean),3))
            print('---------------------------')
            
        
        index = 0
        if 1:
            
            plt.figure(figsize=(6,8))
            
            plt.subplots_adjust(left=0.1, right=0.95,bottom=0.1,top=0.9,wspace=0.15)
            plt.subplot(211)
            # plt.plot(test_input[0,int(0.7*nx),:],label='Train cycle',ls='--',c='black')
            plt.plot(test_T_pre[index,int(0.7*test_nx),:],label='PINO at 35mm',c='r')
            plt.plot(test_T[index,int(0.7*test_nx),:],label='FEM at 35mm',c='blue')
            plt.plot(test_input[index,int(0.7*test_nx),:],label='Test cycle',c='g',ls='--')
            plt.title('Temperature')
            plt.legend()
            plt.subplot(212)
            plt.plot(test_A_pre[index,int(0.7*test_nx),:],label='PINO at 35mm',c='r')
            plt.plot(test_A[index,int(0.7*test_nx),:],label='FEM at 35mm',c='blue')
            plt.title('Degree of cure')
            plt.legend()
            plt.show()
            
        if 1: 
            
            #The spatiotemporal field predictions (temperature and DoC) of PINO and FEM under the two-dwell cure cycle.

            index = 0
            thickness = 0.05
            curing_time = 350*60
            
            plt.figure(figsize=(12,10)) 
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1,top=0.95,wspace = 0.15, hspace = 0.5)
            size = 14
            fontfamily = 'arial'
            font = {'family':fontfamily,
                        'size': 14,
                        'weight':25}
            
            #-----321
            plt.subplot(321)
            levels = np.arange(290,460,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(0, 0.05*1000, test_nx))
            cs = plt.contourf(T_grid, X_grid, test_T_pre[index,:,:], levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)

            plt.xlim(0, 350)
            plt.ylim(0, 50)
            plt.colorbar(cs)
            plt.title('PINO: Temperature',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size) 
            
            #-------323
            plt.subplot(323)
            levels = np.arange(290,460,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(0, 0.05*1000, test_nx))
            cs = plt.contourf(T_grid, X_grid, test_T[index,:,:], levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)

            plt.xlim(0, 350)
            plt.ylim(0, 50)
            plt.colorbar(cs)
            plt.title('FEM: Temperature',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size) 
            
            #-------325
            plt.subplot(325)
            levels = np.arange(290,460,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(0, 0.05*1000, test_nx))
            cs = plt.contourf(T_grid, X_grid, abs(test_T_pre[index,:,:]-test_T[index,:,:]), levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)

            plt.xlim(0, 350)
            plt.ylim(0, 50)
            plt.colorbar(cs)
            plt.title('Absolute error: Temperature',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size) 
            
            #-----322
            plt.subplot(322)
            levels = np.arange(290,500,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(20, 0.05*1000, int(test_nx*0.6)+1))
            cs = plt.contourf(T_grid, X_grid, test_A_pre[index,int(0.4*test_nx):,:], levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)

            plt.xlim(0, 350)
            plt.ylim(20, 50)
            plt.colorbar(cs)
            plt.title('PINO: DoC',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size) 
            
            #-----324
            plt.subplot(324)
            levels = np.arange(290,500,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(20, 0.05*1000, int(test_nx*0.6)+1))
            cs = plt.contourf(T_grid, X_grid, test_A[index,int(0.4*test_nx):,:], levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)

            plt.xlim(0, 350)
            plt.ylim(20, 50)
            plt.colorbar(cs)
            plt.title('FEM: DoC',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size)
            
            #-----326
            plt.subplot(326)
            levels = np.arange(290,500,1)
            T_grid, X_grid = np.meshgrid(np.linspace(0, 350, test_nt), np.linspace(20, 0.05*1000, int(test_nx*0.6)+1))
            cs = plt.contourf(T_grid, X_grid, abs(test_A_pre[index,int(0.4*test_nx):,:]-test_A[index,int(0.4*test_nx):,:]), levels=100, origin='lower',cmap='coolwarm')
            for c in cs.collections:
                c.set_rasterized(True)
                
            plt.xlim(0, 350)
            plt.ylim(20, 50)
            plt.colorbar(cs)
            plt.title('Absolute error: DoC',fontsize=16)
            plt.xlabel('time [min]', fontsize=16)
            plt.ylabel('thickness [mm]', fontsize=16)
            plt.yticks(size = size) 
            plt.xticks(size = size)
             
            plt.show()
        

        
        
        
        
        
    
    
    