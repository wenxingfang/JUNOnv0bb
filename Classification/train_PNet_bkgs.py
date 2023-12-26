import random
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
import os 
from modules import particle_net
#import platform
import argparse
#import healpy as hp
import ast
from torch import linalg as LA
from sklearn.utils import shuffle
import logging

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)



try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


class MAPE_cost(nn.Module):
    def __inti__(self):
        super(MAPE_cost,self).__init__()
        return
    def forward(self, pred, truth):
        tmp_loss = torch.sum(torch.abs((pred-truth)/truth))/truth.size(0)
        return tmp_loss

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



def read_file(filename):##For sim
    f = h5py.File(filename, 'r')
    df = f['data'][:]##N*14*P for sim
    idx = np.sum(np.abs(df),axis=(1,2))>0
    assert np.sum(idx)>0
    df = df[idx]
    df_label = f['label'][idx]##N*16
    df[:,1:4,:] /= 17700. ## r scale
    if parsed['T0_shift']:
        t0 = np.random.uniform(-parsed['T0_shift_val'],parsed['T0_shift_val'],(df.shape[0],1))
        df[:,4  ,:] += t0
        df[:,5  ,:] += t0
    df[:,4  ,:] /= 100. ## time scale
    if parsed['rm_tori']:df[:,4  ,:] = 0 ##remove original time info.
    if parsed['rm_direction']:df[:,11:14,:] = 0 ##remove direction info.
    df[:,6:9,:] /= 17700. ## r scale
    #df, df_label = shuffle(df, df_label)
    #tmp_x = torch.tensor(df.astype('float32')).to(device) 
    #tmp_y = torch.tensor(df_label.astype('float32')).to(device) 
    f.close()
    #return tmp_x[:,6:9,:], tmp_x, tmp_y[:,8:11], df_label
    return df[:,6:9,:], df, df_label


def read_files(filenames_sig, filenames_bkg, device):##For sim
    assert len(filenames_sig) == len(filenames_bkg)
    out_pos = None
    out_data = None
    out_meta = None
    out_label = None
    for i in range(len(filenames_sig)):
        sig_pos, sig_data, sig_meta = read_file(filenames_sig[i]) 
        bkg_pos, bkg_data, bkg_meta = read_file(filenames_bkg[i]) 
        if sig_pos.shape[2] > bkg_pos.shape[2]:
            diff = sig_pos.shape[2] - bkg_pos.shape[2]
            padding_pos = np.zeros((bkg_pos.shape[0],bkg_pos.shape[1],diff), dtype=int)
            padding_data = np.zeros((bkg_data.shape[0],bkg_data.shape[1],diff), dtype=int)
            bkg_pos = np.concatenate((bkg_pos,padding_pos),axis=2)
            bkg_data = np.concatenate((bkg_data,padding_data),axis=2)
        elif sig_pos.shape[2] < bkg_pos.shape[2]:
             diff = bkg_pos.shape[2] - sig_pos.shape[2]
             padding_pos = np.zeros((sig_pos.shape[0],sig_pos.shape[1],diff), dtype=int)
             padding_data = np.zeros((sig_data.shape[0],sig_data.shape[1],diff), dtype=int)
             sig_pos = np.concatenate((sig_pos,padding_pos),axis=2)
             sig_data = np.concatenate((sig_data,padding_data),axis=2)

        all_pos = np.concatenate((sig_pos,bkg_pos),axis=0)
        all_data = np.concatenate((sig_data,bkg_data),axis=0)
        if out_pos is None:
            out_pos = all_pos
            out_data = all_data
        else:
            if out_pos.shape[2] > all_pos.shape[2]:
                padding_pos  = np.zeros((all_pos .shape[0],all_pos .shape[1],out_pos.shape[2]-all_pos.shape[2]), dtype=int)
                padding_data = np.zeros((all_data.shape[0],all_data.shape[1],out_pos.shape[2]-all_pos.shape[2]), dtype=int)
                all_pos  = np.concatenate((all_pos ,padding_pos ),axis=2)
                all_data = np.concatenate((all_data,padding_data),axis=2)
            else:
                padding_pos  = np.zeros((out_pos .shape[0],out_pos .shape[1],-out_pos.shape[2]+all_pos.shape[2]), dtype=int)
                padding_data = np.zeros((out_data.shape[0],out_data.shape[1],-out_pos.shape[2]+all_pos.shape[2]), dtype=int)
                out_pos  = np.concatenate((out_pos ,padding_pos ),axis=2)
                out_data = np.concatenate((out_data,padding_data),axis=2)
            out_pos  =  np.concatenate((out_pos ,all_pos ),axis=0)
            out_data =  np.concatenate((out_data,all_data),axis=0)
 
        sig_label = np.full((sig_data.shape[0],1),0,np.int32)
        bkg_label = np.full((bkg_data.shape[0],1),1,np.int32)
        all_label = np.concatenate((sig_label,bkg_label),axis=0)
        out_label = all_label if out_label is None else np.concatenate((out_label,all_label),axis=0)
        all_meta = np.concatenate((sig_meta,bkg_meta),axis=0)
        all_meta = np.concatenate((all_meta,all_label),axis=1)##cat 0 or 1 for sig and bkg
        ##########
        ext_sig_meta = np.full((sig_data.shape[0],1),0,np.int32)
        bkg_id = -1
        if 'e-' in filenames_bkg[i]: bkg_id = 1 
        elif 'C10' in filenames_bkg[i]: bkg_id = 2
        elif 'Bi214' in filenames_bkg[i]: bkg_id = 3
        else: print('Error: Unknow type bkgs!!!')
        ext_bkg_meta = np.full((bkg_data.shape[0],1),bkg_id,np.int32)
        ext_meta = np.concatenate((ext_sig_meta,ext_bkg_meta),axis=0)
        all_meta = np.concatenate((all_meta,ext_meta),axis=1)
        ###########
        out_meta = all_meta if out_meta is None else np.concatenate((out_meta,all_meta),axis=0)

    out_pos, out_data, out_meta, out_label = shuffle(out_pos, out_data, out_meta, out_label)
    out_pos = torch.tensor(out_pos.astype('float32')).to(device)
    out_data = torch.tensor(out_data.astype('float32')).to(device)
    #out_meta = torch.tensor(out_meta.astype('float32')).to(device)
    out_label = torch.tensor(out_label).long().to(device)
    return out_pos, out_data, out_label, out_meta

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filenamelist, channel, npe_scale, time_scale, E_scale, R_scale, L_scale, use_2D=True, use_ext=True, use_ori_label=False):
        super(Dataset, self).__init__()
        print("Reading Dataset")
        self.data_2D = None
        self.data_ext = None
        self.data_label = None
        self.data_ori_label = None
        for file in filenamelist:
            f = h5py.File(file, 'r')
            label = f['label'][:]
            tmp_label = torch.tensor( label[:,8:11].astype('float32'))
            self.data_label = tmp_label if self.data_label is None else torch.cat((self.data_label,tmp_label),0)
            if use_2D:
                df = f['data2D'][:]
                tmp_tensor = torch.tensor(df.astype('float32'))
                self.data_2D = tmp_tensor if self.data_2D is None else torch.cat((self.data_2D,tmp_tensor),0)
            if use_ext:
                df = f['label'][:,11:15]##FIXME, should be adjusted according to training data, QTEn,recx,recy,recz
                tmp_tensor = torch.tensor(df.astype('float32'))
                self.data_ext = tmp_tensor if self.data_ext is None else torch.cat((self.data_ext,tmp_tensor),0)
            if use_ori_label:
                df = f['label'][:]
                tmp_tensor = torch.tensor(df.astype('float32'))
                self.data_ori_label = tmp_tensor if self.data_ori_label is None else torch.cat((self.data_ori_label,tmp_tensor),0)
            f.close()
        if self.data_2D != None:
            self.data_2D[:,:,:,0] = self.data_2D[:,:,:,0]/(1.0*npe_scale)
            self.data_2D[:,:,:,1] = self.data_2D[:,:,:,1]/(1.0*time_scale)
            if channel == 0:#npe
                self.data_2D = self.data_2D[:,:,:,0:1]
            if channel == 1:#ftime
                self.data_2D = self.data_2D[:,:,:,1:2]
        if self.data_ext != None:
            self.data_ext[:,0] = self.data_ext[:,0]/(1.0*E_scale)
            self.data_ext[:,1:4] = self.data_ext[:,1:4]/(1.0*R_scale)
                                    
    def __getitem__(self, index):
        da_2D = self.data_2D[index,] if self.data_2D != None else torch.tensor([0])
        da_ext = self.data_ext[index,] if self.data_ext != None else torch.tensor([0])
        da_label = self.data_label[index,] if self.data_label != None else torch.tensor([0])
        da_ori_label = self.data_ori_label[index,] if self.data_ori_label != None else torch.tensor([0])
        return (da_2D, da_ext, da_label, da_ori_label)

    def __len__(self):
        return self.data_label.size()[0]

def count_training_evts(filenamelist):
    tot_n = 0 
    for file in filenamelist:
        f = h5py.File(file, 'r')
        tot_n += f['label'].shape[0]
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

def file_block(files_txt,size):
    blocks = {}
    blocks[0]=[]
    index = 0
    with open(files_txt,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            if '#' in line:continue
            line = line.replace('\n','')
            line = line.replace(' ','')
            if index == size:
                blocks[len(blocks)]=[]
                index = 0
                blocks[int(len(blocks)-1)].append(line)
                index += 1
            else:
                blocks[int(len(blocks)-1)].append(line)
                index += 1
    return blocks


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
        self.train_file_block_sig = file_block(self.parsed['train_file_sig'],self.parsed['train_file_bsize'])
        self.valid_file_block_sig = file_block(self.parsed['valid_file_sig'],self.parsed['valid_file_bsize'])
        self.test_file_block_sig  = file_block(self.parsed['test_file_sig' ],self.parsed['test_file_bsize'])
        self.train_file_block_bkg = file_block(self.parsed['train_file_bkg'],self.parsed['train_file_bsize'])
        self.valid_file_block_bkg = file_block(self.parsed['valid_file_bkg'],self.parsed['valid_file_bsize'])
        self.test_file_block_bkg  = file_block(self.parsed['test_file_bkg' ],self.parsed['test_file_bsize'])
        #print(f'train file blocks={self.train_file_block}')
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        #print('fcs=',parsed['fcs'])
        hyperparameters = {
            'knn':parsed['knn'],
            'ps_features':parsed['ps_features'],
        }
        self.model = particle_net.get_model(**hyperparameters).to(self.device)
        version_str = torch.__version__ 
        version_tuple = tuple(map(int, version_str.split('.')[:3]))
        if version_tuple > (2,0,0):
            self.model = torch.compile(self.model)
            print('compiled model !')
 
        #self.loss = L1_cost()
        self.loss = nn.CrossEntropyLoss()
        #if parsed['loss'] == 'Angle':
        #    print('loss=',parsed['loss'])
        #    self.loss = Angle_cost()
 
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
            total_n_sig = 0
            idx = np.arange(len(self.train_file_block_sig))
            for i in idx:
                total_n_sig += count_training_evts(self.train_file_block_sig[i])
            total_n_bkg = 0
            idx = np.arange(len(self.train_file_block_bkg))
            for i in idx:
                total_n_bkg += count_training_evts(self.train_file_block_bkg[i])
            print('tot traning sig =',total_n_sig,',bkg=',total_n_bkg)
            total_steps = int(1.0*(total_n_sig+total_n_bkg)/self.batch_size)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=total_steps, epochs=epochs)

        for epoch in range(1, epochs + 1):
            t_loss, t_corr, t_tot = self.train(epoch)
            v_loss, v_corr, v_tot = self.validate()
            train_loss = 1.0*t_loss/t_tot
            valid_loss = 1.0*v_loss/v_tot
            current_lr = 0
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            print(f'epoch{epoch},train_loss={train_loss},valid_loss={valid_loss}, lr={current_lr}')
            logger.info('')
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            output_name = self.out_name
            output_name = output_name.replace('.pth','_epoch%d.pth'%epoch)

            #if  (valid_loss < best_loss):
            #if  (train_loss < best_loss):
            if  True:
                best_loss = valid_loss

                #hyperparameters = self.model.hyperparameters

                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    #'hyperparameters': hyperparameters,
                    'optimizer': self.optimizer.state_dict(),
                }
                print("Saving model...")
                torch.save(checkpoint, f'{output_name}')

            if parsed['scheduler']=='Plateau':
                self.scheduler.step(train_loss)
            elif parsed['scheduler']=='StepLR':
                self.scheduler.step()

    def train(self, epoch):
        self.train_file_block_sig = file_block(self.parsed['train_file_sig'],self.parsed['train_file_bsize'])
        self.train_file_block_bkg = file_block(self.parsed['train_file_bkg'],self.parsed['train_file_bsize'])
        self.model.train()
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"training Epoch {epoch}/{self.n_epochs}    - t={current_time}")
        idx_sig = np.arange(len(self.train_file_block_sig))
        np.random.shuffle(idx_sig)
        idx_bkg = np.arange(len(self.train_file_block_bkg))
        np.random.shuffle(idx_bkg)
        total_loss = 0
        total_corr = 0
        n_total = 0
        for i in range(len(idx_sig)):
            sig_files = self.train_file_block_sig[idx_sig[i]]
            bkg_files = self.train_file_block_bkg[idx_bkg[i%len(idx_bkg)]]
            if len(sig_files) != len(bkg_files): continue
            df_cord, df_fs, df_y, _ = read_files(sig_files, bkg_files, self.device)
            for ib in range(0, df_cord.size(0), self.batch_size):
                x_cord = df_cord[ib:ib+self.batch_size]                       
                x_fs   = df_fs  [ib:ib+self.batch_size]                       
                Y      = df_y   [ib:ib+self.batch_size]                       
                if x_cord.size(0) != self.batch_size:continue
                Npos = (x_fs.abs().sum(dim=(0,1))>0).sum().item()
                if Npos > 1000: 
                    Npos = 1000
                    x_fs   = x_fs  [:,:,0:Npos]
                    x_cord = x_cord[:,:,0:Npos]
                            
                self.optimizer.zero_grad()
                z = self.model(x_cord, x_fs, None)
                #print('z=',z.size(),',y=',Y.size())
                loss = self.loss(z, Y.squeeze())
                
                loss.backward()
                if self.parsed['clip_grad'] != 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.parsed['clip_grad'])
                self.optimizer.step()
                #total_corr += torch.sum( (torch.argmax(z, dim=1)==Y) ).item()
                total_loss += loss.item()*z.size(0)
                n_total += z.size(0)
                if parsed['scheduler']=='OneCycleLR': self.scheduler.step()
            
        return (total_loss, total_corr, n_total)

    def validate(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"validing - t={current_time}")
        self.model.eval()
        total_corr = 0
        total_loss = 0
        n_total = 0
        with torch.no_grad():
            for i in self.valid_file_block_sig:
                sig_files = self.valid_file_block_sig[i] 
                bkg_files = self.valid_file_block_bkg[i%len(self.valid_file_block_bkg)] 
                if len(sig_files) != len(bkg_files): continue
                df_cord, df_fs, df_y, _ = read_files(sig_files, bkg_files, self.device)
                for ib in range(0, df_cord.size(0), self.batch_size):
                    x_cord = df_cord[ib:ib+self.batch_size]                       
                    x_fs   = df_fs  [ib:ib+self.batch_size]                       
                    Y      = df_y   [ib:ib+self.batch_size]                       
                    if x_cord.size(0) != self.batch_size:continue
                    Npos = (x_fs.abs().sum(dim=(0,1))>0).sum().item()
                    if Npos > 1000: 
                        Npos = 1000
                        x_fs   = x_fs  [:,:,0:Npos]
                        x_cord = x_cord[:,:,0:Npos]
                    z = self.model(x_cord, x_fs, None)
                    loss = self.loss(z, Y.squeeze())
                    total_loss += loss.item()*z.size(0)
                    n_total += z.size(0)
        return (total_loss, total_corr, n_total)


    def test(self):
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"testing - t={current_time}")
        self.model.eval()
        data_out = None
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i in self.test_file_block_sig:
                sig_files = self.test_file_block_sig[i]
                bkg_files = self.test_file_block_bkg[i%len(self.test_file_block_bkg)] 
                if len(sig_files) != len(bkg_files): continue
                df_cord, df_fs, df_y, df_y0 = read_files(sig_files, bkg_files, self.device)
                for ib in range(0, df_cord.size(0), self.batch_size):
                    x_cord = df_cord[ib:ib+self.batch_size]                       
                    x_fs   = df_fs  [ib:ib+self.batch_size]                       
                    Y0     = df_y0  [ib:ib+self.batch_size]                       
                    out = self.model(x_cord, x_fs, None)
                    out=softmax(out)
                    y_pred = out.cpu()
                    y_pred = y_pred.detach().numpy()
                    y_pred =  np.concatenate((Y0,y_pred), axis=1)
                    data_out = y_pred if data_out is None else np.concatenate((data_out, y_pred), axis=0)
        outFile1 = self.parsed['outFile'].replace('.h5','_0.h5')
        hf = h5py.File(outFile1, 'w')
        hf.create_dataset('label' , data=data_out)
        hf.close()
        print('Saved produced data %s'%outFile1)
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
    parser.add_argument('--train_file_sig', default='', type=str, help='')
    parser.add_argument('--valid_file_sig', default='', type=str, help='')
    parser.add_argument('--test_file_sig' , default='', type=str, help='')
    parser.add_argument('--train_file_bkg', default='', type=str, help='')
    parser.add_argument('--valid_file_bkg', default='', type=str, help='')
    parser.add_argument('--test_file_bkg' , default='', type=str, help='')
    parser.add_argument('--train_file_bsize', default=20, type=int, help='')
    parser.add_argument('--valid_file_bsize', default=20, type=int, help='')
    parser.add_argument('--test_file_bsize' , default=20, type=int, help='')
    parser.add_argument('--out_name' , default='', type=str, help='')
    parser.add_argument('--channel'  , default=2, type=int, help='0 for npe, 1 for first hit time')
    parser.add_argument('--npe_scale', default=5, type=float, help='')
    parser.add_argument('--time_scale', default=50, type=float, help='')
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
    parser.add_argument('--DoOptimization', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--clip_grad', default=0, type=float, help='')
    parser.add_argument('--scheduler' , default='StepLR', type=str, help='')
    parser.add_argument('--loss' , default='', type=str, help='')
    parser.add_argument('--frac_ep', default=0.2, type=float, help='')
    parser.add_argument('--frac_pu', default=1.0, type=float, help='')
    parser.add_argument('--nhit_c14', default=50, type=float, help='')
    parser.add_argument('--check_train', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--dim', default=1, type=int, help='')
    parser.add_argument('--n_ext', default=4, type=int, help='')
    parser.add_argument('--useRes', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--use_2D', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--use_1D', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--fcs', nargs='+', type=int, help='')
    parser.add_argument('--knn',default=7, type=int, help='')
    parser.add_argument('--ps_features',default=11, type=int, help='')
    parser.add_argument('--rm_tori', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--rm_direction', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--T0_shift', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--T0_shift_val',default=2, type=float, help='')
    
    parsed = vars(parser.parse_args())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )

    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    logger.info('Constructure NN')

    network = NN(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'], parsed=parsed)
    if parsed['DoOptimization']:
        network.optimize(parsed['epochs'], lr=parsed['lr'])
    if parsed['DoTest']:
        network.test()
        #print('self_loss=',self_loss,',l1_loss=',l1_loss)

    #if parsed['saveONNX']:
    #    network.saveONNX()
