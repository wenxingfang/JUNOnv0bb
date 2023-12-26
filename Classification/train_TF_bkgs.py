import os
import numpy as np
#import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
#import torch.nn.utils.rnn
import h5py
#from torch.autograd import Variable
import time
import random
#from tqdm import tqdm
import sys
from modules import model_ae as model
#import platform
import argparse
#import healpy as hp
import ast
from torch import linalg as LA
from sklearn.utils import shuffle
import math
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


def read_file(filename):##For sim
    f = h5py.File(filename, 'r')
    df = f['data'][:]##N*14*P for sim
    idx = np.sum(np.abs(df),axis=(1,2))>0
    assert np.sum(idx)>0
    df = df[idx]
    df_label = f['label'][idx]##N*16
    df[:,1:4,:] /= 17700. ## r scale
    #if parsed['T0_shift']:
    #    t0 = np.random.uniform(-parsed['T0_shift_val'],parsed['T0_shift_val'],(df.shape[0],1))
    #    df[:,4  ,:] += t0
    #    df[:,5  ,:] += t0
    df[:,4  ,:] /= 100. ## time scale
    #if parsed['rm_tori']:df[:,4  ,:] = 0 ##remove original time info.
    #if parsed['rm_direction']:df[:,11:14,:] = 0 ##remove direction info.
    df[:,6:9,:] /= 17700. ## r scale
    f.close()
    return df[:,6:9,:], df, df_label

def read_file_sort(filename, device, sidx):##For sim
    f = h5py.File(filename, 'r')
    df = f['data'][:]##N*11*P for sim, N*12*P for calib
    idx = np.sum(np.abs(df),axis=(1,2))>0
    assert np.sum(idx)>0
    df = df[idx]
    df_label = f['label'][idx]##N*16
    df, df_label = shuffle(df, df_label)
    df       = torch.tensor(df      .astype('float32')).to(device) 
    df_label = torch.tensor(df_label.astype('float32')).to(device) 
    if sidx>=0:
        mask = df.abs().sum(dim=1)>0##N,P
        df[:,sidx,:] += 9999*(~mask)
        sorted, indices = torch.sort(df[:,sidx,:],dim=1)
        df[:,sidx,:] -= 9999*(~mask)##back
        for i in range(df.size(0)):
            #print('before:',df[i,:,:])
            df[i,:,:] = df[i,:,indices[i]]
            #print('after :',df[i,:,:])
    df[:,1:4,:] /= 17700. ## r scale
    df[:,4  ,:] /= 100. ## time scale
    df[:,6:9,:] /= 17700. ## r scale
    f.close()
    #return tmp_x[:,6:9,:], tmp_x[:,[0,1,2,3,4,5,9,10],:], tmp_y[:,8:11], df_label
    #return df[:,6:9,:], df, df_label[:,8:11], df_label
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
        ########## 
        out_meta = all_meta if out_meta is None else np.concatenate((out_meta,all_meta),axis=0)

    out_pos, out_data, out_meta, out_label = shuffle(out_pos, out_data, out_meta, out_label)
    out_pos = torch.tensor(out_pos.astype('float32')).to(device)
    out_data = torch.tensor(out_data.astype('float32')).to(device)
    #out_meta = torch.tensor(out_meta.astype('float32')).to(device)
    out_label = torch.tensor(out_label).long().to(device)
    return out_pos, out_data, out_label, out_meta




def count_training_evts(filenamelist):
    tot_n = 0 
    for file in filenamelist:
        f = h5py.File(file, 'r')
        tot_n += f['label'].shape[0]
        f.close()
    return tot_n


def file_block(files_txt,size):
    blocks = {}
    blocks[0]=[]
    index = 0
    with open(files_txt,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)##Add
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
        self.train_file_block_sig = file_block(parsed['train_file_sig'],parsed['train_file_bsize'])
        self.valid_file_block_sig = file_block(parsed['valid_file_sig'],parsed['valid_file_bsize'])
        self.test_file_block_sig  = file_block(parsed['test_file_sig' ],parsed['test_file_bsize'])
        self.train_file_block_bkg = file_block(parsed['train_file_bkg'],parsed['train_file_bsize'])
        self.valid_file_block_bkg = file_block(parsed['valid_file_bkg'],parsed['valid_file_bsize'])
        self.test_file_block_bkg  = file_block(parsed['test_file_bkg' ],parsed['test_file_bsize'])
 
        #print(f'train file blocks={self.train_file_block}')
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        #print('fcs=',parsed['fcs'])

        hyperparameters = {
            #'features_cfg': cfg[parsed['cfg'] ],
            #'fcs_cfg':[1024, 128, 2],
            'in_channels': parsed['ps_features'],
            'fcs_cfg':parsed['fcs'],
            'dropout':parsed['Dropout'],
            'emb_dim':parsed['emb_dim'],
            'psencoding':parsed['psencoding'],
            'last_act':parsed['last_act'],
            'nlayers':parsed['nlayers'],
            'nhead':parsed['nhead'],
            'nhid':parsed['nhid'],
            'en_dropout':parsed['en_dropout']
        }
 


        self.model = model.TFENet(hyperparameters).to(self.device)
        version_str = torch.__version__ 
        version_tuple = tuple(map(int, version_str.split('.')[:3]))
        if version_tuple > (2,0,0):
            self.model = torch.compile(self.model)
            print('compiled model !')
 
        self.loss = nn.CrossEntropyLoss()
   
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
        self.train_file_block_sig = file_block(self.parsed['train_file_sig'],self.parsed['train_file_bsize'])#Add
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
                x_fs   = df_fs  [ib:ib+self.batch_size]                       
                Y      = df_y   [ib:ib+self.batch_size]                       
                Npos = (x_fs.abs().sum(dim=(0,1))>0).sum().item()
                if Npos > 1000: Npos = 1000
                #print('Npos=',Npos,',size0=',x_fs.size(2))
                x_fs = x_fs[:,:,0:Npos]
                if x_fs.size(0) != self.batch_size: continue
                #mask = x_fs.abs().sum(dim=1).squeeze()  # (N, bin)                        
                #if torch.any(mask.sum(dim=1)<=0):
                #    print('fname=',fname,',size=',df_cord.size(0),',ib=',ib,',ib+self.batch_size=',ib+self.batch_size,',x_fs size=',x_fs.size() )
                self.optimizer.zero_grad()
                z = self.model(x_fs)
                #print('z=',z.size(),',Y=',Y.size())
                #print('z=',z,',Y=',Y)
                loss = self.loss(z, Y.squeeze())
                #print('loss nan=',torch.any(torch.isnan(loss)),',loss=',loss )
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
                    x_fs   = df_fs  [ib:ib+self.batch_size]                       
                    Y      = df_y   [ib:ib+self.batch_size]                       
                    Npos = (x_fs.abs().sum(dim=(0,1))>0).sum().item()
                    if Npos > 1000: Npos = 1000
                    x_fs = x_fs[:,:,0:Npos]
                    if x_fs.size(0) != self.batch_size: continue
                    z = self.model(x_fs)
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
                    x_fs   = df_fs  [ib:ib+self.batch_size]                       
                    Y0     = df_y0  [ib:ib+self.batch_size]                       
                    Npos = (x_fs.abs().sum(dim=(0,1))>0).sum().item()
                    if Npos > 1000: Npos = 1000
                    x_fs = x_fs[:,:,0:Npos]
                    if x_fs.size(0) != self.batch_size: continue
                    out = self.model(x_fs)
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
    parser.add_argument('--train_file_bsize', default=2, type=int, help='')
    parser.add_argument('--valid_file_bsize', default=2, type=int, help='')
    parser.add_argument('--test_file_bsize' , default=2, type=int, help='')
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
    parser.add_argument('--emb_dim', default=32, type=int, help='')
    parser.add_argument('--nlayers', default=2, type=int, help='')
    parser.add_argument('--nhead', default=8, type=int, help='')
    parser.add_argument('--nhid', default=2048, type=int, help='')
    parser.add_argument('--en_dropout', default=0.1, type=float, help='')
    parser.add_argument('--sort_idx', default=-1, type=int, help='4 is hittime, 5 is hittime_cor')
    parser.add_argument('--psencoding', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--last_act' , default='', type=str, help='')
 
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
