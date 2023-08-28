import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

#class ResidualBlock(nn.Module):
#    def __init__(self, NSIDE, channels_in=16, channels_out=16, dropout=0.0):
#        """
#        Define all layers of a residual conv block
#        """
#        super(ResidualBlock, self).__init__()
#                        
#        activation = nn.LeakyReLU(0.2)
#
#        self.layers = nn.ModuleList([])
#
#        # self.layers.append(nn.BatchNorm1d(channels_in))
#        self.layers.append(activation)
#        self.layers.append(spherical.sphericalConv(NSIDE, channels_in, channels_out, nest=True))
#
#        # self.layers.append(nn.BatchNorm1d(channels_out))
#        self.layers.append(activation)
#        # self.layers.append(nn.Dropout(dropout))
#        self.layers.append(spherical.sphericalConv(NSIDE, channels_out, channels_out, nest=True))
#
#        self.residual = nn.Conv1d(channels_in, channels_out, kernel_size=1)
#                
#    def forward(self, x):
#
#        tmp = x
#        
#        for layer in self.layers:
#            x = layer(x)
#        
#        return x + self.residual(tmp)
#
#    def weights_init(self):        
#        for module in self.modules():
#            kaiming_init(module)

class ResConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, dim, batch_norm):
        """
        Define all layers of a residual conv block
        """
        super(ResConvBlock, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1) if dim == 1 else nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1) )
        self.layers.append(nn.BatchNorm1d(channels_out) if dim == 1 else nn.BatchNorm2d(channels_out))
        self.layers.append(nn.ReLU(inplace=True))
        self.residual = nn.Conv1d(channels_in, channels_out, kernel_size=1, padding=0) if dim == 1 else nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0)
    def forward(self, x):
        tmp = x
        for layer in self.layers:
            x = layer(x)
        return x + self.residual(tmp)
    def weights_init(self):        
        for module in self.modules():
            kaiming_init(module)

def make_layers_dim(cfg, in_channels, batch_norm=False, useRes=False, dim=2, bn_after=False, bn_input=False):
    layers = []
    if bn_input:
        layers += [nn.BatchNorm2d(in_channels) if dim==2 else nn.BatchNorm1d(in_channels)]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2) if dim==2 else nn.MaxPool1d(kernel_size=2, stride=2) ]
        else:
            if useRes:
                conv = ResConvBlock(in_channels, v, dim, batch_norm)
                layers += [conv]
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3) if dim==2 else nn.Conv1d(in_channels, v, kernel_size=3)
                if batch_norm:
                    layers += [conv, nn.BatchNorm2d(v) if dim==2 else nn.BatchNorm1d(v), nn.ReLU(inplace=True)] if bn_after==False else [conv, nn.ReLU(inplace=True), nn.BatchNorm2d(v) if dim==2 else nn.BatchNorm1d(v)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

def make_layers(cfg, in_channels, batch_norm=False, useRes=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class CNN1D2D(nn.Module):
    def __init__(self, hyperparameters):
        super(CNN1D2D, self).__init__()
        self.hyperparameters = hyperparameters
        in_channels = hyperparameters['in_channels']
        features_cfg = hyperparameters['features_cfg']
        fcs_cfg = hyperparameters['fcs_cfg']
        Batch_Norm = hyperparameters['Batch_Norm']
        n_input = hyperparameters['n_input']
        dropout = hyperparameters['dropout']
        n_ext = hyperparameters['n_ext']
        useRes = hyperparameters['useRes']
        bn_after = hyperparameters['bn_after']
        bn_input = hyperparameters['bn_input']
        mod_list_1d = []
        mod_list_1d += make_layers_dim(features_cfg, in_channels, batch_norm=Batch_Norm, useRes=useRes, dim=1, bn_after=bn_after, bn_input=bn_input )
        self.features_1d = nn.ModuleList(mod_list_1d)
        mod_list_2d = []
        mod_list_2d += make_layers_dim(features_cfg, in_channels, batch_norm=Batch_Norm, useRes=useRes, dim=2, bn_after=bn_after, bn_input=bn_input )
        self.features_2d = nn.ModuleList(mod_list_2d)
        fcs_list = []
        for i in range(len(fcs_cfg)):
            if i == 0:
                fcs_list.append(  nn.Linear(n_input+n_ext, fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
            elif i== (len(fcs_cfg)-1):
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
            else:
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
        self.fcs = nn.ModuleList(fcs_list)

    def forward(self, x_1d, x_2d, x_add):
        for i in self.features_1d:
            x_1d = i(x_1d)
        x_2d = x_2d.transpose(1,3)
        x_2d = x_2d.transpose(2,3)##FIXME
        for i in self.features_2d:
            x_2d = i(x_2d)
        x_1d = torch.reshape(x_1d,(x_1d.size(0), -1))
        #print('x_1d size=',x_1d.size())
        x_2d = torch.reshape(x_2d,(x_2d.size(0), -1))
        #print('x_2d size=',x_2d.size())
        x_1d = torch.cat((x_1d,x_2d),1)## add
        #print('y size=',y.size(),',x=',x_add)
        if torch.all(x_add==0) == False:
            x_1d = torch.cat((x_1d,x_add),1)## add
        #print('y size=',y.size())
        for i in self.fcs:
            x_1d = i(x_1d)
        return x_1d


class CNN1D(nn.Module):
    def __init__(self, hyperparameters):
        super(CNN1D, self).__init__()
        self.hyperparameters = hyperparameters
        in_channels = hyperparameters['in_channels']
        features_cfg = hyperparameters['features_cfg']
        fcs_cfg = hyperparameters['fcs_cfg']
        Batch_Norm = hyperparameters['Batch_Norm']
        n_input = hyperparameters['n_input']
        dropout = hyperparameters['dropout']
        dim = hyperparameters['dim']
        n_ext = hyperparameters['n_ext']
        useRes = hyperparameters['useRes']
        bn_after = hyperparameters['bn_after']
        bn_input = hyperparameters['bn_input']
        mod_list = []
        #mod_list += make_layers(features_cfg, in_channels, batch_norm=Batch_Norm, dim=dim)
        mod_list += make_layers_dim(features_cfg, in_channels, batch_norm=Batch_Norm, useRes=useRes, dim=dim, bn_after=bn_after, bn_input=bn_input )
        self.features = nn.ModuleList(mod_list)
        fcs_list = []
        for i in range(len(fcs_cfg)):
            if i == 0:
                fcs_list.append(  nn.Linear(n_input+n_ext, fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
            elif i== (len(fcs_cfg)-1):
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
            else:
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
        self.fcs = nn.ModuleList(fcs_list)

    def forward(self, x, x_add):
        y = x
        for i in self.features:
            y = i(y)
        y = torch.reshape(y,(y.size(0), -1))
        #print('y size=',y.size(),',x=',x_add)
        if torch.all(x_add==0) == False:
            y = torch.cat((y,x_add),1)## add
        #print('y size=',y.size())
        for i in self.fcs:
            y = i(y)
        return y

#class Vgg(torch.jit.ScriptModule):
class Vgg(nn.Module):
    #__constants__ = ['features','fcs']
    #def __init__(self, in_channels, features_cfg, fcs_cfg, Batch_Norm, n_input, dropout):
    def __init__(self, hyperparameters):
        super(Vgg, self).__init__()
        self.hyperparameters = hyperparameters
        # Hyperparameters
        in_channels = hyperparameters['in_channels']
        features_cfg = hyperparameters['features_cfg']
        fcs_cfg = hyperparameters['fcs_cfg']
        Batch_Norm = hyperparameters['Batch_Norm']
        n_input = hyperparameters['n_input']
        #print('n_input=',n_input)
        dropout = hyperparameters['dropout']
        useRes = hyperparameters['useRes']
        bn_after = hyperparameters['bn_after']
        bn_input = hyperparameters['bn_input']
        mod_list = []
        #mod_list += make_layers(features_cfg, in_channels, batch_norm=Batch_Norm)
        mod_list += make_layers_dim(features_cfg, in_channels, batch_norm=Batch_Norm, useRes=useRes, dim=2, bn_after=bn_after, bn_input=bn_input )
        self.features = nn.ModuleList(mod_list)
        fcs_list = []
        for i in range(len(fcs_cfg)):
            if i == 0:
                #fcs_list.append(  nn.Linear(n_input + 1 + 3 + 4 + 4, fcs_cfg[i]) )##cfgC ##add tot pe, charge center (x,y,z), fastest hit (time,pmt_x_y_z), max npe(n,pmt_x_y_z)
                fcs_list.append(  nn.Linear(n_input, fcs_cfg[i]) )
                #fcs_list.append( nn.LeakyReLU(0.01) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
                #if(Batch_Norm): fcs_list.append( nn.BatchNorm1d(fcs_cfg[i]) )##FIXME,added at 20230227
            elif i== (len(fcs_cfg)-1):
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                #fcs_list.append( nn.ReLU(True) )
            else:
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
                #if(Batch_Norm): fcs_list.append( nn.BatchNorm1d(fcs_cfg[i]) )
        self.fcs = nn.ModuleList(fcs_list)


    #@torch.jit.script_method
    def forward(self, x, x_add):
        #print('x size=',x.size(),',x_add=',x_add.size() ) #(N, 124, 231, 2), (N, 4)
        #tot_pe = torch.sum(x[:,:,:,0:1], dim=(1,2), keepdim=False )/1000  
        x = x.transpose(1,3)##N,X,Y,F -> N,F,Y,X
        x = x.transpose(2,3)##N,F,Y,X -> N,F,X,Y
        #print('x tranposed size=',x.size()) #(N, 2, 124, 231)
        # the first two values in the pad input correspond to the last dimension
        #x = F.pad(input=x, pad=(0, 1, 1, 1), mode='constant', value=0)
        for i in self.features:
            x = i(x)
        #print('x size0=',x.size())
        #y = y.view(y.size(0), -1)
        x = torch.reshape(x,(x.size(0), -1))
        if torch.all(x_add==0) == False:
            x = torch.cat((x,x_add),1)## add
        #print('x size=',x.size())
        #y = torch.cat((y,tot_pe),1)## add tot pe
        for i in self.fcs:
            x = i(x)
            #print('y=',y)
        return x


class Model(nn.Module):
    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.hyperparameters = hyperparameters

        # Hyperparameters
        self.in_channels = hyperparameters['in_channels']
        
        # Encoder
        #self.encoder = nn.Sequential(
        #    nn.Conv2d(self.in_channels, 128, 3, stride=1),  # 
        #    nn.BatchNorm2d(128),
        #    nn.PReLU(128),
        #    nn.MaxPool2d(2),##N, C, 62, 115
        #    nn.Conv2d(128, 64, 3, stride=1),
        #    nn.BatchNorm2d(64),
        #    nn.PReLU(64),
        #    nn.MaxPool2d(2),##N, C, 31, 57
        #    nn.Conv2d(64, 32, 3, stride=1),
        #    nn.BatchNorm2d(32),
        #    nn.PReLU(32),
        #    nn.MaxPool2d(2),##N, C, 15, 29
        #    nn.Conv2d(32, 16, 3, stride=1),
        #    nn.BatchNorm2d(16),
        #    nn.PReLU(16),
        #    nn.MaxPool2d(2),##N, 16, 8, 15
        #    nn.Conv2d(16, 8, 3, stride=1),
        #    nn.BatchNorm2d(8),
        #    nn.PReLU(8),
        #    nn.MaxPool2d(2)##N, 8, 4, 8
        #)

        #self.decoder = nn.Sequential(
        #    nn.Upsample(scale_factor=2, mode='nearest'),
        #    nn.Conv2d(8, 16, 3, stride=1),
        #    nn.BatchNorm2d(16),
        #    nn.PReLU(16),
        #    nn.Upsample(scale_factor=2, mode='nearest'),
        #    nn.Conv2d(16, 32, 3, stride=1),
        #    nn.BatchNorm2d(32),
        #    nn.PReLU(32),
        #    nn.Upsample(scale_factor=2, mode='nearest'),
        #    nn.Conv2d(32, 64, 3, stride=1),
        #    nn.BatchNorm2d(64),
        #    nn.PReLU(64),
        #    nn.Upsample(scale_factor=2, mode='nearest'),
        #    nn.Conv2d(64, 128, 3, stride=1),
        #    nn.BatchNorm2d(128),
        #    nn.PReLU(128),
        #    nn.Upsample(scale_factor=2, mode='nearest'),
        #    nn.Conv2d(128, self.in_channels, 3, stride=1),
        #    nn.BatchNorm2d(self.in_channels),
        #    nn.ReLU()
        #)

        #B, C, 124, 230
        self.encoder_layers = nn.ModuleList([])
        #self.encoder_layers.append( nn.Conv2d(self.in_channels, 128, 3, stride=(3,2), padding=(10,5)) )## 128, 48, 120  
        self.encoder_layers.append( nn.Conv2d(self.in_channels, 128, 3, stride=(3,2)) )## 128, 48, 120  
        self.encoder_layers.append( nn.BatchNorm2d(128) )
        self.encoder_layers.append( nn.PReLU(128) )
        #self.encoder_layers.append( nn.Conv2d(128, 64, 3, stride=2) )### 64,24,60
        self.encoder_layers.append( nn.Conv2d(128, 64, 3, stride=2, padding=(1,1)) )### 64,24,60
        self.encoder_layers.append( nn.BatchNorm2d(64) )
        self.encoder_layers.append( nn.PReLU(64) )
        #self.encoder_layers.append( nn.Conv2d(64, 32, 3, stride=2) )## 32, 12, 30
        self.encoder_layers.append( nn.Conv2d(64, 32, 3, stride=2, padding=(1,1)) )## 32, 12, 30
        self.encoder_layers.append( nn.BatchNorm2d(32) )
        self.encoder_layers.append( nn.PReLU(32) )
        #self.encoder_layers.append( nn.Conv2d(32, 16, 3, stride=2) )## 16, 6, 15
        self.encoder_layers.append( nn.Conv2d(32, 16, 3, stride=2, padding=(1,1)) )## 16, 6, 15
        self.encoder_layers.append( nn.BatchNorm2d(16) )
        self.encoder_layers.append( nn.PReLU(16) )
        self.encoder_layers.append( nn.Conv2d(16, 8, 3, stride=(2,3), padding=(1,0)) )##8, 3, 5
        self.encoder_layers.append( nn.BatchNorm2d(8) )
        self.encoder_layers.append( nn.PReLU(8) )

        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.append( nn.ConvTranspose2d(8, 16, 3, stride=(2,3), padding=(1,0), output_padding=(1,0)) )
        self.decoder_layers.append( nn.BatchNorm2d(16) )
        self.decoder_layers.append( nn.PReLU(16) )
        self.decoder_layers.append( nn.ConvTranspose2d(16, 32, 3, stride=2, padding=(1,1), output_padding=(1,1)) ) 
        self.decoder_layers.append( nn.BatchNorm2d(32) )
        self.decoder_layers.append( nn.PReLU(32) )
        self.decoder_layers.append( nn.ConvTranspose2d(32, 64, 3, stride=2, padding=(1,1), output_padding=(1,1))  ) 
        self.decoder_layers.append( nn.BatchNorm2d(64) )
        self.decoder_layers.append( nn.PReLU(64) )
        self.decoder_layers.append( nn.ConvTranspose2d(64, 128, 3, stride=2, padding=(1,1), output_padding=(1,1))  )
        self.decoder_layers.append( nn.BatchNorm2d(128) )
        self.decoder_layers.append( nn.PReLU(128) )
        self.decoder_layers.append( nn.ConvTranspose2d(128, self.in_channels, 3, stride=(3,2), output_padding=(0,1))  )
        self.decoder_layers.append( nn.BatchNorm2d(self.in_channels) )
        self.decoder_layers.append( nn.ReLU())
    def forward(self, x):
        #print('input size=',x.size()) 
        for layer in self.encoder_layers:
            x = layer(x)
        #    print('en size=',x.size()) 
            
        for layer in self.decoder_layers:
            x = layer(x)
        #    print('de size=',x.size())
        #print('out size=',x.size())
        return x

class PositionalEncoding(nn.Module):
    "Implement the PE function. https://blog.51cto.com/u_11466419/5530949"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)




class TFENet(nn.Module):
    def __init__(self, hyperparameters):
        super(TFENet, self).__init__()
        self.hyperparameters = hyperparameters
        in_channels = hyperparameters['in_channels']
        emb_dim = hyperparameters['emb_dim']
        fcs_cfg = hyperparameters['fcs_cfg']
        dropout = hyperparameters['dropout']
        self.usePSEN = hyperparameters['psencoding']
        nlayers = 2
        nhead = 8
        nhid = 2048#default
        en_dropout = 0.1#default
        self.emb = nn.Conv1d(in_channels, emb_dim, kernel_size=1)
        pre_list = [self.emb, nn.BatchNorm1d(emb_dim)]
        self.pre_layer = nn.ModuleList(pre_list)
        #self.transformer = nn.Transformer(d_model=emb_dim, num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=512, batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, nhead, nhid, en_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, nlayers)
        #self.transformer = nn.Transformer(d_model=emb_dim, num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=512)
        self.positional_encoding = PositionalEncoding(emb_dim, dropout=0, max_len=5000)
        #self.predictor = nn.Linear(emb_dim, 2)
        fcs_list = []
        for i in range(len(fcs_cfg)):
            if i == 0:
                fcs_list.append(  nn.Linear(emb_dim, fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
            elif i== (len(fcs_cfg)-1):
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
            else:
                fcs_list.append(  nn.Linear(fcs_cfg[i-1], fcs_cfg[i]) )
                fcs_list.append( nn.ReLU(True) )
                if(dropout != 0): fcs_list.append( nn.Dropout(p=dropout) )
        self.fcs = nn.ModuleList(fcs_list)



    def forward(self, src):#N,F,bin
        if torch.any(torch.isnan(src)):
            print('src_in find nan')
        mask = (src.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, bin)
        mask1 = mask.permute(0, 2, 1)#(N, 1, bin)->(N, bin, 1)
        src_padding_mask = (mask==False).squeeze()##N,bin, False means not padding, True means padding
        for i in self.pre_layer:
            src = i(src)*mask
        
        if torch.any(torch.isnan(src)):
            print('src0 find nan')
        src = src.permute(0, 2, 1)##N,emb,bin -> N,bin,emb
        if self.usePSEN:
            src = self.positional_encoding(src)*mask1
        #if torch.any(torch.isnan(src)):
        #    print('src1 find nan')
        src = src.permute(1, 0, 2)##N,bin,emb -> bin,N,emb
        if torch.any(torch.isnan(src_padding_mask)):
            print('src mask find nan')
        #print('src=',src)
        #print('mask=',src_padding_mask)
        out = self.transformer(src=src, src_key_padding_mask=src_padding_mask) ## bin,N,emb
        if torch.any(torch.isnan(out)):
            print('out0 find nan')
        #print('out0=',out)
        out = out.permute(1, 0, 2)##bin,N,emb -> N,bin,emb
        count = (out.abs().sum(dim=2, keepdim=True) != 0)  # (N, bin, 1)
        count = torch.sum(count,1)#N,1
        if torch.any(count<=0):
            print('find zero count')
        out = torch.sum(out, 1)/count ## N,bin,emb -> N,emb
        #print('out1 nan=',torch.any(torch.isnan(out)) )
        for i in self.fcs:
            out = i(out)
        #print('out2 nan=',torch.any(torch.isnan(out)) )
        return out





