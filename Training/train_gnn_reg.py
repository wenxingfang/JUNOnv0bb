#import shutil
import numpy as np
#import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
#import torch.nn.utils.rnn
import h5py
#from torch.autograd import Variable
import time
#from tqdm import tqdm
import sys
import math 
#import platform
import argparse
#import healpy as hp
import ast
import json
import random
#import torch.distributions as td
#from matplotlib.lines import Line2D
#from particle_net import get_model_decay, get_model_TwoBodys, get_model_TwoBodysV2, get_model_TwoBodysV3
from gravnet import PhotonNet
from torch import linalg as LA
#torch.autograd.set_detect_anomaly(True)
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss,self).__init__()
        return
    def forward(self, pred, truth):
        #tmp_sum_pred = torch.FloatTensor([0.])
        #tmp_sum_truth = torch.FloatTensor([0.])
        #tmp_sum_corss = torch.FloatTensor([0.])
        if pred.dim() == 3:
            tmp_sum_pred = torch.sum(pred*pred,(1,2))
            tmp_sum_truth = torch.sum(truth*truth,(1,2))
            tmp_sum_cross = torch.sum(pred*truth,(1,2))
            tmp_loss = (tmp_sum_pred+tmp_sum_truth)/(2*tmp_sum_cross) -1 
            tmp_loss = torch.sum(tmp_loss)/pred.size(0)
            return tmp_loss
        if pred.dim() == 4:
            tmp_sum_pred = torch.sum(pred*pred,(2,3))
            tmp_sum_truth = torch.sum(truth*truth,(2,3))
            tmp_sum_cross = torch.sum(pred*truth,(2,3))
            tmp_loss = (tmp_sum_pred+tmp_sum_truth)/(2*tmp_sum_cross) -1 
            tmp_loss = torch.sum(tmp_loss)/pred.size(0)
            return tmp_loss

        #tmp_loss = (torch.sum(pred*pred) + torch.sum(truth*truth))/(2*torch.sum(pred*truth))
        #tmp_loss0 = (torch.sum(pred[0]*pred[0]) + torch.sum(truth[0]*truth[0]))/(2*torch.sum(pred[0]*truth[0]))
        #print('tmp_loss0=',tmp_loss0-1,'abs sum pred=',torch.sum(torch.abs(pred[0])),',abs sum real=',torch.sum(torch.abs(truth[0])) )
        #return (tmp_loss - 1)/pred.size(0)

def dice_cost(pred_y, label_y):##https://agenda.infn.it/event/28874/contributions/169211/attachments/94397/130957/20220708_AEsforSUEP_ICHEP2022_schhibra.pdf
    tmp_loss = (torch.sum(pred_y*pred_y) + torch.sum(label_y*label_y))/(2*torch.sum(pred_y*label_y))
    return (tmp_loss - 1)/pred_y.size(0)


class link_loss(nn.Module):
    def __init__(self):
        super(link_loss,self).__init__()
        return
    def forward(self,pred_y, label_y, mask):
        tmp_loss = torch.sum( torch.abs((pred_y-label_y)*mask) )/torch.sum(mask)
        return tmp_loss

class loss_2body(nn.Module):
    def __init__(self):
        super(loss_2body,self).__init__()
        return
    def forward(self,pred_y, label_y, mask):##N,P,P N,P,P, N,P,P
        tmp_loss = torch.sum( nn.functional.binary_cross_entropy(input=pred_y, target=label_y,reduction='none')*mask )/torch.sum(mask)
        return tmp_loss


class l1_loss_w(nn.Module):
    def __init__(self):
        super(l1_loss_w,self).__init__()
        return
    def forward(self,pred_y, label_y, weight):
        tmp_loss = torch.sum( torch.abs((pred_y-label_y)*weight) )/pred_y.size(0)
        return tmp_loss

class ce_loss_w(nn.Module):
    def __init__(self):
        super(ce_loss_w,self).__init__()
        self.Loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,pred_y, label_y, weight):
        tmp_loss = torch.sum(self.Loss(pred_y, label_y)*weight)/pred_y.size(0)
        return tmp_loss





from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

def build_graph(filenamelist, device=''):
    graphs = []
    for file in filenamelist:
        f = h5py.File(file, 'r')
        df = f['data'][:]
        df_label = f['label'][:]
        for i in range(df.shape[0]):
            df_i = df[i]##11*P for sim, 12*P for calib
            label_i = df_label[i]##16
            Np = 0
            for j in range(df_i.shape[1]):
                if df_i[0,j]==0 and df_i[1,j]==0:
                    Np = j
                    break
            if Np <=0: continue
            dfi = df_i[:,0:Np]
            dfi[1:4,0:Np] /= 17700. ## r scale
            dfi[4,0:Np] /= 100. ## time scale
            dfi[6:9,0:Np] /= 17700. ## r scale
            dfi = np.transpose(dfi,(1,0))##F,P --> P,F
            tmp_x = torch.tensor(dfi.astype('float32'))

            #df_y = label_i[8:11]
            #df_y = np.reshape(df_y,(1,3))
            df_y = np.reshape(label_i,(1,-1))
   
            tmp_y = torch.tensor(df_y.astype('float32'))
            tb_graph = GraphData(x=tmp_x,y=tmp_y)
            graphs.append(tb_graph)
        f.close()
    #graphs = GraphBatch.from_data_list(graphs).to(device)
    return graphs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenamelist, Np_min, in_features):
        super(Dataset, self).__init__()
        print("Reading Dataset")
        self.data = None
        self.data_label = None
        self.data_train = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            f_json = file.replace('.h5','.json')
            f_io = open(f_json,'r')
            link_dict = json.load(f_io)
            df = f['feature'][:]
            df_train = f['feature_train'][:,in_features,:,:]
            #print('link len()=',len(link_dict),',df shape=',df.shape,',df_train=',df_train.shape)         
            assert df.shape[0]==len(link_dict)
            n_points = df.shape[2]
            tmp_links = np.full((df.shape[0],n_points,n_points),0,np.float32)
            #tmp_links = np.full((df.shape[0],n_points,n_points),-1,np.float32)
            for i in range(len(link_dict)):
                if len(link_dict[i])<=0:continue
                for lk in link_dict[i]:
                    tmp_links[i,lk[0],lk[1]] = 1 
                    tmp_links[i,lk[1],lk[0]] = 1 
                    #pdg_0 = df[i,5,lk[0] ]
                    #pdg_1 = df[i,5,lk[1] ]
                    #pdg_mo_0 = df[i,6,lk[0] ]
                    #pdg_mo_1 = df[i,6,lk[1] ]
                    #assert pdg_mo_0 == pdg_mo_1
                    #if pdg_mo_0 == 111:
                    #    print('found pi0,pdg0=',pdg_0,',pdg1=',pdg_1)
                    
            for i in range(tmp_links.shape[1]):
                tmp_links[:,i,i] = 1
       
            tmp_label = torch.tensor(tmp_links.astype('float32'))
            self.data_label = tmp_label if self.data_label is None else torch.cat((self.data_label,tmp_label),0)
            tmp_tensor = torch.tensor(df.astype('float32'))
            self.data = tmp_tensor if self.data is None else torch.cat((self.data,tmp_tensor),0)
            tmp_tensor_train = torch.tensor(df_train.astype('float32'))
            self.data_train = tmp_tensor_train if self.data_train is None else torch.cat((self.data_train,tmp_tensor_train),0)

            f_io.close()
            f.close()
        if Np_min !=0:
            del_list = []
            for i in range(self.data.shape[0]):
                Np = 0
                for j in range(self.data.shape[2]):
                    if self.data[i,3,j] != 0: Np=j+1##px,py,pz,e,m
                    else: break
                if Np < Np_min: del_list.append(i)
            self.data_label = np.delete(self.data_label, del_list, 0)
            self.data       = np.delete(self.data      , del_list, 0)
            self.data_train = np.delete(self.data_train, del_list, 0)
        self.data_dist = None
        #self.data_dist = np.full((self.data.shape[0],2,self.data.shape[2]),0,np.float32)
        #for i in range(self.data.shape[0]):
        #    for j in range(self.data.shape[2]):
        #        if self.data[i,3,j] <= 0 : break
        #        px = self.data[i,0,j]
        #        py = self.data[i,1,j]
        #        pz = self.data[i,2,j]
        #        pt = math.sqrt(px*px + py*py)
        #        p  = math.sqrt(px*px + py*py + pz*pz)
        #        costheta = pz/p ## -1 to 1
        #        theta = math.acos(costheta)##0-pi
        #        cosphi = px/pt
        #        phi = math.acos(cosphi)
        #        if py < 0: phi = 2*math.pi - phi 
        #        self.data_dist[i,0,j] = theta
        #        self.data_dist[i,1,j] = phi

        #if self.data_1D != None:
        #    self.data_1D[:,0,:] = self.data_1D[:,0,:]/(1.0*scale_1d)
        #    self.data_1D[:,1,:] = self.data_1D[:,1,:]/(1.0*scale_1d_tcor)
                                    
    def __getitem__(self, index):
        da = self.data[index,] if self.data != None else torch.tensor([0])
        da_wise = self.data_train[index,] if self.data_train != None else torch.tensor([0])
        da_label = self.data_label[index,] if self.data_label != None else torch.tensor([0])
        da_dist = self.data_dist[index,] if self.data_dist is not None else torch.tensor([0])
        return (da, da_wise, da_label.long(), da_dist)

    def __len__(self):
        return self.data_label.size()[0]

def count_training_evts(filenamelist):
    tot_n = 0 
    for file in filenamelist:
        f = h5py.File(file, 'r')
        label = f['label']
        tot_n += label.shape[0]
        #tmp_index = label[:,9]<=0 ##no c14 hit
        #tmp_index1 = label[:,9]>nhit_c14 # c14 hit
        #tot_n_pu += int( np.sum(tmp_index1) )
        f.close()
    return tot_n

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, npe_scale, time_scale, do_log_scale=False):
        super(DatasetTest, self).__init__()
        print("Reading DatasetTest...")
        self.T0 = None
        self.T1 = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            label = f['label'][:]
            f.close()
            tmp_tensor0 = torch.tensor(label.astype('float32'))
            tmp_tensor1 = torch.tensor(df   .astype('float32'))
            self.T0 = tmp_tensor0 if self.T0 is None else torch.cat((self.T0,tmp_tensor0),0)
            self.T1 = tmp_tensor1 if self.T1 is None else torch.cat((self.T1,tmp_tensor1),0)
        self.T1[:,:,:,0] = self.T1[:,:,:,0]/(1.0*npe_scale)
        self.T1[:,:,:,1] = self.T1[:,:,:,1]/(1.0*time_scale)
        if do_log_scale:
            tmp_t = self.T1[:,:,:,1]
            idx = tmp_t > 0
            tmp_t[idx] = torch.log(tmp_t[idx])
            self.T1[:,:,:,1] = tmp_t
        if channel == 0:#npe
            self.T1 = self.T1[:,:,:,0:1]
        if channel == 1:#ftime
            self.T1 = self.T1[:,:,:,1:2]
        
        self.n = self.T1.size()[0]        
                                    
    def __getitem__(self, index):
        T0 = self.T0[index,]
        T1 = self.T1[index,]
        return (T0,T1)

    def __len__(self):
        return self.n


class DatasetTestTrain(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, npe_scale, time_scale, frac_ep, nhit_c14, do_log_scale=False):
        super(DatasetTestTrain, self).__init__()
        print("Reading DatasetTestTrain")
        self.T = None
        self.Y = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            df = f['data'][:]
            label = f['label'][:]
            f.close()
            tmp_ep_index = label[:,9]<=0 ##no c14 hit
            tmp_pu_index = label[:,9]>nhit_c14 # c14 hit

            tmp_tensor = torch.tensor(df[tmp_ep_index,:,:,:].astype('float32'))
            tmp_tensor = tmp_tensor[0:int(tmp_tensor.size(0)*frac_ep)]
            tmp_label = torch.tensor(label[tmp_ep_index].astype('float32'))
            tmp_label = tmp_label[0:int(tmp_label.size(0)*frac_ep)]

            tmp_tensor_pu = torch.tensor(df[tmp_pu_index,:,:,:].astype('float32'))
            tmp_label_pu = torch.tensor(label[tmp_pu_index].astype('float32'))

            tmp_tensor =  torch.cat((tmp_tensor,tmp_tensor_pu),0)
            tmp_label =  torch.cat((tmp_label,tmp_label_pu),0)

            self.T = tmp_tensor if self.T is None else torch.cat((self.T,tmp_tensor),0)
            self.Y = tmp_label if self.Y is None else torch.cat((self.Y,tmp_label),0)
        self.T[:,:,:,0] = self.T[:,:,:,0]/(1.0*npe_scale)
        self.T[:,:,:,1] = self.T[:,:,:,1]/(1.0*time_scale)
        if do_log_scale:
            tmp_t = self.T[:,:,:,1]
            idx = tmp_t > 0
            tmp_t[idx] = torch.log(tmp_t[idx])
            self.T[:,:,:,1] = tmp_t
        if channel == 0:#npe
            self.T = self.T[:,:,:,0:1]
        if channel == 1:#ftime
            self.T = self.T[:,:,:,1:2]

        self.n = self.T.size()[0]        
                                    
    def __getitem__(self, index):
        T1 = self.T[index,]
        T0 = self.Y[index,]
        return (T0,T1)

    def __len__(self):
        return self.n


#def comb_file_block(files_txt,size):
#    files = files_txt.split(';')
#    for file in files:
#        file_block(file,size)
class Angle_cost(nn.Module):
    def __inti__(self):
        super(Angle_cost,self).__init__()
        return
    def forward(self, pred, truth):
        eps = 1e-6
        tmp_dot = torch.sum(pred*truth,axis=1)
        tmp_norm = torch.sqrt(torch.sum(pred*pred,axis=1))*torch.sqrt(torch.sum(truth*truth,axis=1))
        tmp_cosangle = tmp_dot/(tmp_norm+eps)
        tmp_cosangle[tmp_cosangle> 1] =  1
        tmp_cosangle[tmp_cosangle<-1] = -1
        tmp_angle = torch.acos(tmp_cosangle)
        return torch.mean(tmp_angle)



def file_block(files_txt,size):
    blocks = {}
    blocks[0]=[]
    index = 0
    lines = []
    print('files_txt=',files_txt)
    files = files_txt.split(':')
    for file in files:
        print('file=',file)
        if '.txt' not in file:continue
        with open(file,'r') as f:
            tmp_lines = f.readlines()
            for line in tmp_lines:
                if '#' in line:continue
                line = line.replace('\n','')
                line = line.replace(' ','')
                lines.append(line)
    random.shuffle (lines)
    for line in lines:
        if index == size:
            blocks[len(blocks)]=[]
            index = 0
            blocks[int(len(blocks)-1)].append(line)
            index += 1
        else:
            blocks[int(len(blocks)-1)].append(line)
            index += 1
    return blocks

class L1_cost(nn.Module):
    def __inti__(self):
        super(L1_cost,self).__init__()
        return
    def forward(self, pred, truth):
        #print('pred size=',pred.size(),',truth =', truth.size())
        tmp_norm = LA.norm(pred, dim=1, keepdim=True)
        #print('norm size=',tmp_norm.size())
        pred = pred/tmp_norm
        tmp_loss = torch.sum(torch.abs(pred-truth))/truth.size(0)
        return tmp_loss


class NN(object):
    def __init__(self, batch_size=64, gpu=0, smooth=0.05, parsed={}):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size        
        self.parsed = parsed
        self.train_file_block = file_block(parsed['train_file'],parsed['train_file_bsize'])
        self.valid_file_block = file_block(parsed['valid_file'],parsed['valid_file_bsize'])
        self.test_file_block  = file_block(parsed['test_file' ],parsed['test_file_bsize'])
        #print(f'train file blocks={self.train_file_block}')
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        
        input_dim = parsed['Fdim']
        self.model = PhotonNet(input_dim = input_dim, output_dim = 3, grav_dim = 128, hidden_dim = 256, n_gravnet_blocks = 3, n_postgn_dense_blocks = 4, dropout = 0.1).to(self.device)
        #self.loss = nn.CrossEntropyLoss(reduction='none')##FIXME input after RELU?
        self.loss = L1_cost()
        if parsed['loss'] == 'Angle':
            print('loss=',parsed['loss'])
            self.loss = Angle_cost()
 
        #self.loss = loss_2body()
        #self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()
        #self.loss = link_loss()
        #if parsed['loss']=='ce_we':
        #    self.loss = ce_loss_w()
        #    print('Use CrossEntropyLoss Loss with weight')
        #if parsed['loss']=='knn':
        #    self.loss = statistics_diff.SmoothKNNStatistic(self.batch_size, self.batch_size, True, 1, compute_t_stat=True)    
        #    print('Use KNN Loss')
        #if parsed['loss']=='dice':
        #    self.loss = Dice_loss()
        #    print('Use Dice Loss')
    
        if parsed['Restore']:
            print('restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.model.load_state_dict(checkpoint['state_dict'])
    def optimize(self, epochs, lr=3e-4):        

        parsed = self.parsed
        print(f'doing optimizing:')
        best_loss = float('inf')

        self.lr = lr
        self.n_epochs = epochs        

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = parsed['out_name']

        print(f'Model: {self.out_name}')
        print(" Number of params : ", sum(x.numel() for x in self.model.parameters()))


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if parsed['Restore']:
            print('opt. restored from ',parsed['restore_file'])
            checkpoint = torch.load(parsed['restore_file'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        if parsed['scheduler']=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7,patience=2,threshold=0.001,threshold_mode='rel')
        elif parsed['scheduler']=='OneCycleLR':
            total_N = 0
            idx = np.arange(len(self.train_file_block))
            for i in idx:
                tmp_n = count_training_evts(self.train_file_block[i] )
                total_N += tmp_n
            print('tot traning =',total_N)
            total_steps = int(1.0*(total_N)/self.batch_size)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=total_steps, epochs=epochs)

        for epoch in range(1, epochs + 1):
            t_loss, t_tot = self.train(epoch)
            v_loss, v_tot = self.validate()
            train_loss = 1.0*t_loss/t_tot
            valid_loss = 1.0*v_loss/v_tot
            current_lr = 0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print(f'epoch{epoch},train_loss={train_loss},valid_loss={valid_loss}, lr={current_lr}')
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            output_name = self.out_name
            output_name = output_name.replace('.pth','_epoch%d.pth'%epoch)

            if  True:
                best_loss = valid_loss
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'optimizer': self.optimizer.state_dict(),
                }
                print("Saving model...")
                torch.save(checkpoint, f'{output_name}')

            if parsed['scheduler']=='Plateau':
                self.scheduler.step(train_loss)
            elif parsed['scheduler']=='StepLR':
                self.scheduler.step()

    def train(self, epoch):
        self.model.train()
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"training Epoch {epoch}/{self.n_epochs}    - t={current_time}")
        idx = np.arange(len(self.train_file_block))
        np.random.shuffle(idx)
        total_loss = 0
        n_total = 0
        for i in idx:
            dataset = build_graph(filenamelist=self.train_file_block[i])
            train_loader = GraphDataLoader(dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)
            for grs in train_loader:
                #print('grs.x0 =',grs.x)                      
                grs = grs.to(self.device)   
                self.optimizer.zero_grad()
                z = self.model(grs.x, grs.batch)
                Y = grs.y[:,8:11].to(self.device)
                loss = self.loss(z, Y)
                loss.backward()
                if self.parsed['clip_grad'] != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.parsed['clip_grad'])
                self.optimizer.step()
                total_loss += loss.item()*Y.size(0)
                n_total += Y.size(0)
                if parsed['scheduler']=='OneCycleLR': self.scheduler.step()
        return (total_loss, n_total)

    def validate(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"validing - t={current_time}")
        self.model.eval()
        total_loss = 0
        n_total = 0
        for i in self.valid_file_block:
            dataset = build_graph(filenamelist=self.valid_file_block[i])
            validation_loader = GraphDataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            with torch.no_grad():
                for grs in validation_loader:
                    grs = grs.to(self.device)   
                    z = self.model(grs.x, grs.batch)
                    Y = grs.y[:,8:11].to(self.device)
                    loss = self.loss(z, Y)
                    total_loss += loss.item()*Y.size(0)
                    n_total += Y.size(0)
        return (total_loss, n_total)


    def test(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"testing - t={current_time}")
        self.model.eval()
        data_out = None
        for i in self.test_file_block:
            dataset = build_graph(filenamelist=self.test_file_block[i])
            test_loader = GraphDataLoader(dataset, batch_size=self.batch_size, shuffle=False, **self.kwargs)
            with torch.no_grad():
                for grs in test_loader:
                    grs = grs.to(self.device)
                    z   = self.model(grs.x, grs.batch)
                    y_pred = z.cpu().detach().numpy()
                    Y      = grs.y.cpu().detach().numpy()
                    out    =  np.concatenate((Y,y_pred), axis=1)
                    data_out = out if data_out is None else np.concatenate((data_out, out), axis=0)
        outFile1 = self.parsed['outFile'].replace('.h5','_0.h5')
        hf = h5py.File(outFile1, 'w')
        hf.create_dataset('label' , data=data_out)
        hf.close()
        print('Saved produced data %s'%outFile1)
    def save_script(self):
        with torch.no_grad():
            device = torch.device("cpu")
            self.model.to(device)
            self.model.eval()
            x, h = torch.rand(26, 8), torch.zeros(26, dtype=torch.long)
            h[25:26] = 1
            print('z0=',self.model(x, h))
            h[15:26] = 1
            print('z1=',self.model(x, h))
            h[0 :25] = 0
            h[25:26] = 1
            ##scripted = torch.jit.script(self.model)
            print('start tracing:')
            scripted = torch.jit.trace(self.model, (x, h))
            print(scripted.code)
            scripted.save(self.parsed['out_ScriptName'])
            print('test loading:')
            loaded = torch.jit.load(self.parsed['out_ScriptName'])
            print('test0:',loaded(x, h))
            h[15:26] = 1
            print('test1:',loaded(x, h))
            return 0           
    #def saveONNX(self):
    #    device = torch.device("cpu")
    #    net.to(device)
    #    net.eval()
    #    tmp_x = torch.randn(1, 6, requires_grad=False)    
    #    torch_out = net(tmp_x)
    #    print('tmp_x=',tmp_x,',torch_out=',torch_out)
    #    netONNX.to(device)
    #    netONNX.eval()
    #    torch_out = netONNX(tmp_x)
    #    print('tmp_x=',tmp_x,',onnx torch_out=',torch_out)
    #    # Export the model
    #    torch.onnx.export(netONNX,               # model being run
    #              tmp_x,                         # model input (or a tuple for multiple inputs)
    #              onnx_file_path,   # where to save the model (can be a file or file-like object)
    #              export_params=True,        # store the trained parameter weights inside the model file
    #              do_constant_folding=True,  # whether to execute constant folding for optimization
    #              input_names = ['input'],   # the model's input names
    #              output_names = ['output'], # the model's output names
    #              dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                            'output': {0 : 'batch_size'}}
    #    )

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--gpu', '--gpu', default=0, type=int, metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float, metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=200, type=int, metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--batch', '--batch', default=128, type=int, metavar='BATCH', help='Batch size')
    parser.add_argument('--train_file', default='', type=str, help='')
    parser.add_argument('--valid_file', default='', type=str, help='')
    parser.add_argument('--test_file' , default='', type=str, help='')
    parser.add_argument('--train_file_bsize', default=200, type=int, help='')
    parser.add_argument('--valid_file_bsize', default=200, type=int, help='')
    parser.add_argument('--test_file_bsize' , default=200, type=int, help='')
    parser.add_argument('--out_name' , default='', type=str, help='')
    parser.add_argument('--channel'  , default=0, type=int, help='0 for npe, 1 for first hit time')
    parser.add_argument('--npe_scale', default=5, type=float, help='')
    parser.add_argument('--time_scale', default=100, type=float, help='')
    parser.add_argument('--do_log_scale', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--scale_1d', default=10., type=float, help='')
    parser.add_argument('--scale_1d_tcor', default=50., type=float, help='')
    parser.add_argument('--E_scale', default=1., type=float, help='')
    parser.add_argument('--R_scale', default=17700., type=float, help='')
    parser.add_argument('--L_scale', default=40000., type=float, help='')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--restore_file' , default='', type=str, help='')
    parser.add_argument('--outFile' , default='', type=str, help='')
    parser.add_argument('--cfg' , default='', type=str, help='')
    parser.add_argument('--BatchNorm', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Dropout', default=0, type=float, help='')
    parser.add_argument('--DoTest', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SaveScript', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--out_ScriptName' , default='', type=str, help='')
    parser.add_argument('--DoOptimization', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--clip_grad', default=0, type=float, help='')
    parser.add_argument('--scheduler' , default='StepLR', type=str, help='')
    parser.add_argument('--loss' , default='', type=str, help='')
    parser.add_argument('--ps_features', default=32, type=int, help='')
    parser.add_argument('--ps_input_dropout', default=0.0, type=float, help='')
    parser.add_argument('--for_inference', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--fcs', nargs='+', type=int, help='')
    parser.add_argument('--Np_min', default=2, type=int, help='')
    parser.add_argument('--Fdim', default=11, type=int, help='')
    parser.add_argument('--activation' , default='relu', type=str, help='')
    parser.add_argument('--weight', default=1., type=float, help='')
    parser.add_argument('--notime', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--onlytime', action='store', type=ast.literal_eval, default=False, help='')
    
    parsed = vars(parser.parse_args())

    network = NN(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], parsed=parsed)
    if parsed['DoOptimization']:
        network.optimize(parsed['epochs'], lr=parsed['lr'])
    if parsed['DoTest']:
        network.test()
    if parsed['SaveScript']:
        network.save_script()
        #print('self_loss=',self_loss,',l1_loss=',l1_loss)

    #if parsed['saveONNX']:
    #    network.saveONNX()
