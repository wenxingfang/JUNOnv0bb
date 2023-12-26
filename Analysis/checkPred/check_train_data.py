import ROOT as rt
import tempfile
import os
import numpy as np
import h5py
import math
import sys
import argparse
import gc
import ast
from numpy import linalg as LA
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);

# For Ge68 use track size == 1 event
# add info with first hit time and max npe of pmt and charge center
def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO study. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch_size', action='store', type=int, default=3000,
                        help='Number of event for each batch.')
    parser.add_argument('--m_max_evt', action='store', type=int, default=-1,
                        help='max_evt Number')
    parser.add_argument('--input', nargs='+', default=[],
                        help='input root file.')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output hdf5 file.')
    parser.add_argument('--r_scale', action='store', type=float, default=17700,
                        help='r normalization')
    parser.add_argument('--theta_scale', action='store', type=float, default=180,
                        help='theta scale.')
    parser.add_argument('--phi_scale', action='store', type=float, default=180,
                        help='dphi scale.')
    parser.add_argument('--isGe68', action='store', type=ast.literal_eval, default=True,
                        help='isGe68.')
    parser.add_argument('--add_DN', action='store', type=ast.literal_eval, default=False,
                        help='add_DN.')
    parser.add_argument('--add_tts', action='store', type=ast.literal_eval, default=False,
                        help='add_tts.')

    return parser

def plot_grs(hs_dict, out_name, title, rangeX, rangeY, x_sep):#hs_dict={'leg_name':[hist,color,drawoption]}
    for i in hs_dict:
        hs_dict[i][0].SetLineWidth(2)
        hs_dict[i][0].SetLineColor(hs_dict[i][1])
        hs_dict[i][0].SetMarkerColor(hs_dict[i][1])
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    y_min = rangeY[0]
    y_max = rangeY[1]
    if 'logy' in out_name: 
        canvas.SetLogy()
        y_min = tmp_min_y
        y_min = int( math.log10(y_min) )
        y_min = math.pow(10,y_min-1)
        y_max = tmp_max_y
        y_max = int( math.log10(y_max) )
        y_max = math.pow(10,y_max+1)
    dummy = rt.TH2D("dummy","",1, x_min, x_max, 1, y_min, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("")
    for i in hs_dict:
        if 'training' in i: hs_dict[i][0].SetMarkerStyle(24)
        if 'testing' in i: 
            #hs_dict[i][0].SetMarkerStyle(20)
            strs = i.split(':')[1]
            strs = strs.split('->')
            str_from = strs[0]
            str_to   = strs[1]
            str_from = str_from.replace(' ','')
            str_to   = str_to  .replace(' ','')
            markers = [20,24]
            if 'e^{-}' in str_from:
                markers = [21,25]
            elif 'gamma' in str_from:
                markers = [22,26]
            hs_dict[i][0].SetMarkerStyle(markers[0])
            if str_to != str_from:
                #hs_dict[i][0].SetMarkerStyle(24)
                hs_dict[i][0].SetMarkerStyle(markers[1])
        hs_dict[i][0].Draw("%s"%(hs_dict[i][2]))
    dummy.Draw("AXISSAME")
    x_l = 0.15
    y_h = 0.85
    y_dh = 0.3
    x_dl = 0.65
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.SetNColumns(3)
    for i in hs_dict:
        legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'pe' if ('ep' in hs_dict[i][2] or 'pe' in hs_dict[i][2]) else 'pl')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)
    legend.Draw()
    #N_int = h1.Integral(1,h1.GetNbinsX())
    #N_sumw2 = h1.GetSumOfWeights()
    #tline_list = []
    #for sep in x_sep:
    #    cen = 0
    #    for j in range(h1.GetNbinsX(),2,-1):
    #        if h1.Integral(1,j)>N_int*sep and h1.Integral(1,j-1)<N_int*sep:
    #            cen = 0.5*h1.GetXaxis().GetBinCenter(j) + 0.5*h1.GetXaxis().GetBinCenter(j-1)
    #            break
    #    tline = rt.TLine(cen,0,cen,0.5*y_max)
    #    tline.SetLineColor(rt.kBlack)
    #    tline.SetLineWidth(2)
    #    tline.Draw()
    #    tline_list.append(tline)
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()




def plot_hs(hs_dict, norm, out_name, title, rangeX, x_sep, sep_list=[]):#hs_dict={'leg_name':[hist,color,drawoption]}
    tmp_max_y = 0
    tmp_min_y = 999
    if norm:
        for i in hs_dict:
            hs_dict[i][0].Scale(1.0/hs_dict[i][0].GetSumOfWeights())
    for i in hs_dict:
        hs_dict[i][0].SetLineWidth(2)
        hs_dict[i][0].SetMarkerStyle(4)
        hs_dict[i][0].SetMarkerColor(hs_dict[i][1])
        hs_dict[i][0].SetLineColor(hs_dict[i][1])
        if hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMaximumBin()) > tmp_max_y: tmp_max_y = hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMaximumBin())
        if hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMinimumBin()) < tmp_min_y: tmp_min_y = hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMinimumBin())
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    y_min = 0
    y_max = 1.5*tmp_max_y
    if 'logy' in out_name: 
        canvas.SetLogy()
        #y_min = 1e-5 
        y_min = tmp_min_y if tmp_min_y > 0 else 1e-1
        #print('y_min=',y_min)
        y_min = int( math.log10(y_min) )
        y_min = math.pow(10,y_min-1)
        y_min = 1e-5 
        y_max = tmp_max_y
        y_max = int( math.log10(y_max) )
        y_max = math.pow(10,y_max+1)
    dummy = rt.TH2D("dummy","",1, x_min, x_max, 1, y_min, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("")
    sep_lines = {}
    for i in hs_dict:
        #hs_dict[i][0].Draw("same:pe")
        hs_dict[i][0].Draw("same:%s"%(hs_dict[i][2]))
        tmp_sumw2 = hs_dict[i][0].GetSumOfWeights()
        if tmp_sumw2 <=0:continue
        for sep in sep_list:
            for xbin in range(1, hs_dict[i][0].GetNbinsX()+1):
                if hs_dict[i][0].Integral(1,xbin)/tmp_sumw2 < sep and hs_dict[i][0].Integral(1,xbin+1)/tmp_sumw2 > sep:
                    sep_lines['%s_%s'%(i,str(sep))]=[hs_dict[i][0].GetXaxis().GetBinCenter(xbin+1),hs_dict[i][1]]
                    break
    tlines = []
    for i in sep_lines:
        tmp_line = rt.TLine(sep_lines[i][0],0,sep_lines[i][0],0.01)
        tmp_line.SetLineColor(sep_lines[i][1])
        tmp_line.SetLineWidth(2)
        tmp_line.Draw('same')
        tlines.append(tmp_line)
        
    x_l = 0.15
    y_h = 0.85
    y_dh = 0.3
    x_dl = 0.65
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.SetNColumns(3)
    for i in hs_dict:
        #legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'pel' if ('ep' in hs_dict[i][2] or 'pe' in hs_dict[i][2]) else 'l')
        legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'pe' if ('ep' in hs_dict[i][2] or 'pe' in hs_dict[i][2]) else 'l')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)
    legend.Draw()
    #N_int = h1.Integral(1,h1.GetNbinsX())
    #N_sumw2 = h1.GetSumOfWeights()
    #tline_list = []
    #for sep in x_sep:
    #    cen = 0
    #    for j in range(h1.GetNbinsX(),2,-1):
    #        if h1.Integral(1,j)>N_int*sep and h1.Integral(1,j-1)<N_int*sep:
    #            cen = 0.5*h1.GetXaxis().GetBinCenter(j) + 0.5*h1.GetXaxis().GetBinCenter(j-1)
    #            break
    #    tline = rt.TLine(cen,0,cen,0.5*y_max)
    #    tline.SetLineColor(rt.kBlack)
    #    tline.SetLineWidth(2)
    #    tline.Draw()
    #    tline_list.append(tline)
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()


def root2hdf5 (batch_size, tree, start_event, out_name, id_dict, x_max, y_max, ID_x_y_z_dict):

    df = np.full((batch_size, y_max+1, x_max+1, 2), 0, np.float32)
    df_true = np.full((batch_size, 8+9), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        edep      = getattr(tree, "edep")
        edepX     = getattr(tree, "edepX")
        edepY     = getattr(tree, "edepY")
        edepZ     = getattr(tree, "edepZ")
        C14_edep  = getattr(tree, "C14_edep")
        C14_edepX = getattr(tree, "C14_edepX")
        C14_edepY = getattr(tree, "C14_edepY")
        C14_edepZ = getattr(tree, "C14_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recx")
        recy      = getattr(tree, "recy")
        recz      = getattr(tree, "recz")

        nC14      = getattr(tree, "nhitC14")
        nIBD      = getattr(tree, "nhitIBD")
        time_C14  = getattr(tree, "time_C14")
        time_ep   = getattr(tree, "time_ep")
        time_readout   = getattr(tree, "time_trigger")##trigger time - 100 ns

        pmtID     = getattr(tree, "PMTIDs")
        hittime   = getattr(tree, "times")
        charges   = getattr(tree, "charges")

        pmtID_new = []
        hittime_new = []
        charges_new = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            pmtID_new.append(pmtID[ih])
            hittime_new.append(hittime[ih])
            charges_new.append(charges[ih])
        pmtID = pmtID_new
        hittime = hittime_new 
        charges = charges_new
        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = C14_edep
        df_true[ie-start_event, 5] = C14_edepX
        df_true[ie-start_event, 6] = C14_edepY
        df_true[ie-start_event, 7] = C14_edepZ
        df_true[ie-start_event, 8] = nIBD
        df_true[ie-start_event, 9] = nC14
        df_true[ie-start_event, 10] = time_ep -time_readout
        df_true[ie-start_event, 11] = time_C14-time_readout
        df_true[ie-start_event, 12] = QTEn
        df_true[ie-start_event, 13] = recx
        df_true[ie-start_event, 14] = recy
        df_true[ie-start_event, 15] = recz
        df_true[ie-start_event, 16] = QTL
        for i in range(0, len(pmtID)):
            ID     = pmtID[i]
            if ID not in id_dict:continue
            if ID not in tmp_dict:
                tmp_dict[ID] = charges[i]
            else:
                tmp_dict[ID] += charges[i]
            if ID not in tmp_firstHitTime_dict:
                tmp_firstHitTime_dict[ID] = hittime[i]
            else:
                tmp_firstHitTime_dict[ID] = hittime[i] if hittime[i] < tmp_firstHitTime_dict[ID] else tmp_firstHitTime_dict[ID]
        for iD in id_dict:
            ix     = id_dict[iD][0]
            iy     = id_dict[iD][1]
            tmp_npe = 0
            tmp_firstHitTime = 0
            if iD in tmp_dict:
                tmp_npe          = tmp_dict[iD]
                tmp_firstHitTime = tmp_firstHitTime_dict[iD]
            df[ie-start_event,iy,ix,0] = tmp_npe
            df[ie-start_event,iy,ix,1] = tmp_firstHitTime

    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)

    print('data shape=',df.shape,', label=', df_true.shape)

    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data' , data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)

    if Draw_data:
        for i in range(10):
            tmp_df_0 = df[i,:,:,0]
            tmp_df_1 = df[i,:,:,1]
            tmp_nIBD = df_true[i,8] 
            tmp_nC14 = df_true[i,9] 
            draw_data(outname='npe_nIBD_%d_nC14_%d'%(tmp_nIBD, tmp_nC14)  , x_max=x_max, y_max=y_max, df=tmp_df_0)
            draw_data(outname='Ftime_nIBD_%d_nC14_%d'%(tmp_nIBD, tmp_nC14), x_max=x_max, y_max=y_max, df=tmp_df_1)



def get_pmt_theta_phi(file_pos, sep, i_id, i_theta, i_phi):
    id_dict = {}
    theta_list = []
    phi_list = []
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        #phi   = int(phi) ## otherwise it will be too much
        if theta not in theta_list:
            theta_list.append(theta)
        if phi not in phi_list:
            phi_list.append(phi)
        if ID not in id_dict:
            id_dict[ID]=[theta, phi]
    return (id_dict, theta_list, phi_list)

def get_pmt_x_y_z_theta_phi(file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi):
    id_dict = {}
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        x     = float(items[i_x])
        y     = float(items[i_y])
        z     = float(items[i_z])
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        if ID not in id_dict:
            id_dict[ID]=[x, y, z, theta, phi]
    return id_dict


def do_plot2d(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.Draw("COLZ")
    canvas.SaveAs("%s.png"%(out_name))
    del canvas
    gc.collect()

def pmt_type(file_in):
    f = open(file_in,'r')
    lines = f.readlines()
    id_type_dict = {}
    tmp_n_Ham = 0
    tmp_n_NNVT = 0
    for line in lines:
        items = line.split()
        ID    = int(items[0])
        tmp_type    = 0 if items[1]=='Hamamatsu' else 1
        id_type_dict[ID] = tmp_type
        if tmp_type==0: tmp_n_Ham +=1
        else: tmp_n_NNVT += 1
    print('Hamamatsu=',tmp_n_Ham,',NNVT=',tmp_n_NNVT)
    return id_type_dict

def id_ipix_map(file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi):
    f = open(file_pos,'r')
    lines = f.readlines()
    id_dict = {}
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        phi   = float(items[i_phi])
        theta = float(items[i_theta])
        ipix = hp.ang2pix(NSIDE, math.pi*theta/180, math.pi*phi/180, NEST)
        id_dict[ID] = ipix
    return id_dict     
      
def vertexRec_map(file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi):
    max_width = 230
    x_shift = 115
    f = open(file_pos,'r')
    lines = f.readlines()
    pmt_zi = 0
    z_index = 0
    First = True
    lx_ly_id_dict = {}
    id_lx_ly_dict = {}
    id_x_y_z_dict = {}
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        x     = float(items[i_x])
        y     = float(items[i_y])
        z     = float(items[i_z])
        id_x_y_z_dict[ID] = [x,y,z]
        phi   = float(items[i_phi])
        if phi > 180: phi = phi-360
        local_r = math.sqrt(x*x + y*y)
        r       = math.sqrt(x*x + y*y + z*z)
        if First :
            pmt_zi = z
            First = False

        # set the index of z axis
        if int(z) == int(pmt_zi):
            pass
        else:
            z_index += 1
            pmt_zi = z

        #shift the first and last 21 pmts to avoid overlap
        if ID == 7 or ID == 17606: 
            z_index += 1

        #lx = int(np.floor((phi * (local_r / r) / (np.pi * 2.0)) * 230)) + 150
        #print('phi=',phi,',local_r=',local_r,',r=',r)
        #lx = int( max_width* phi * local_r / (r * 360) )
        lx = round( max_width* phi * local_r / (r * 360) ) + x_shift
        ly = z_index
        if lx not in lx_ly_id_dict:
            lx_ly_id_dict[lx] = {}
        if ly not in lx_ly_id_dict[lx]:
            lx_ly_id_dict[lx][ly] = ID
        else:
            print('overlap:ID=',ID,',ori ID=',lx_ly_id_dict[lx][ly],',lx=',lx,',ly=',ly,',x=',x,'y,=',y,'z=',z)
        id_lx_ly_dict[ID]=[lx,ly] 
    return id_lx_ly_dict, id_x_y_z_dict

def draw_map(id_dict):
    x_min = 999
    x_max = -1
    y_min = 999
    y_max = -1
    for i in id_dict:
        x = id_dict[i][0]
        y = id_dict[i][1]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    print('x_min=',x_min,',x_max=',x_max,',y_min=',y_min,',y_max=',y_max)
    h_map = rt.TH2F('map','',x_max+2, -1, x_max+1, y_max+2, -1, y_max+1)
    for i in id_dict:
        x = id_dict[i][0]
        y = id_dict[i][1]
        h_map.SetBinContent(x+1, y+1, i) 
    do_plot2d(hist=h_map, out_name='%s/id_map'%plots_path, title={'X':'x(phi)','Y':'y(z)'})
    return (x_min, x_max, y_min, y_max)

def draw_data(outname, x_max, y_max, df):
    h_map = rt.TH2F('map_%s'%outname,'',x_max+2, -1, x_max+1, y_max+2, -1, y_max+1)
    for iy in range(df.shape[0]):
        for ix in range(df.shape[1]):
            h_map.SetBinContent(ix+1, iy+1, df[iy][ix]) 
    do_plot2d(hist=h_map, out_name='%s/map_%s'%(plots_path, outname), title={'X':'x(phi)','Y':'y(z)'})

def doAna(df_data, tag='train') :
    #global m_em_n_all 
    #global m_em_n_corr
    #global m_ep_n_all 
    #global m_ep_n_corr
    #global m_gm_n_all 
    #global m_gm_n_corr
    #h_pred_prob_tmp = rt.TH1F('h_pred_prob_tmp_%s'%tag, '', 100, 0, 1)
    #h_tmp_edep = rt.TH1F('h_tmp_edep_%s'%tag, '', 12, 0, 12)
    #h_tmp_r    = rt.TH1F('h_tmp_r_%s'%tag, '', 18, 0, 18)
    #h_pred_list = []
    #h_edep_list = []
    #h_r_list = []
    #for i in pdgid:
    #   for j in pdgid:
    #       h_pred_list.append(h_pred_prob_tmp.Clone('h_pred__%s__%s__%s'%(i,j,tag)) )
    #       h_edep_list.append(h_tmp_edep     .Clone('h_edep__%s__%s__%s'%(i,j,tag)) )
    #       h_r_list   .append(h_tmp_r        .Clone('h_r__%s__%s__%s'   %(i,j,tag)) )

    h_hittime_cor = rt.TH1F('h_hittime_cor_%s'%tag, '', 50, -10, 40)
    #h_pred_sig = rt.TH1F('h_pred_sig_%s'%tag, '', 100, 0, 1)
    #h_pred_bkg = rt.TH1F('h_pred_bkg_%s'%tag, '', 100, 0, 1)
    #h_E_sig = rt.TH1F('h_E_sig_%s'%tag, 'edep (MeV)', 100, 2, 3)
    #h_E_bkg = rt.TH1F('h_E_bkg_%s'%tag, 'edep (MeV)', 100, 2, 3)
    #h_R_sig = rt.TH1F('h_R_sig_%s'%tag, 'R (m)', 32, 0, 16)
    #h_R_bkg = rt.TH1F('h_R_bkg_%s'%tag, 'R (m)', 32, 0, 16)
    #h_diff_px = rt.TH1F('h_diff_px_%s'%tag, '', 200, -2, 2)
    #h_diff_py = rt.TH1F('h_diff_py_%s'%tag, '', 200, -2, 2)
    #h_diff_pz = rt.TH1F('h_diff_pz_%s'%tag, '', 200, -2, 2)
    #h_diff_theta = rt.TH1F('h_diff_theta_%s'%tag, '', 180, -180, 180)
    #h_diff_phi   = rt.TH1F('h_diff_phi_%s'  %tag, '', 180, -180, 180)
    #h_dAngle   = rt.TH1F('h_dAngle_%s'  %tag, '', 180, 0, 180)
    #h_dAngle_r0   = rt.TH1F('h_dAngle_r0_%s'  %tag, 'R<10m', 180, 0, 180)
    #h_dAngle_r1   = rt.TH1F('h_dAngle_r1_%s'  %tag, 'R>10m', 180, 0, 180)
    #h_dAngle_E0   = rt.TH1F('h_dAngle_E0_%s'  %tag, '1-2MeV', 180, 0, 180)
    #h_dAngle_E1   = rt.TH1F('h_dAngle_E1_%s'  %tag, '2-3MeV', 180, 0, 180)
    #h_dAngle_rand   = rt.TH1F('h_dAngle_rand_%s'  %tag, '', 180, 0, 180)
    #h_cos_dAngle_rand   = rt.TH1F('h_cos_dAngle_rand_%s'  %tag, '', 200, -1, 1)
    #h_cos_dAngle_E1   = rt.TH1F('h_cos_dAngle_E1_%s'  %tag, '2-3MeV', 200, -1, 1)
    #h_pred_link_true  = rt.TH1F('h_pred_link_true_tmp_%s'%tag, '', 120, -0.1, 1.1)
    #h_pred_link_false = rt.TH1F('h_pred_link_false_tmp_%s'%tag, '', 120, -0.1, 1.1)
    #h2_pred_link_true_mass  = rt.TH2F('h2_pred_link_true_mass_tmp_%s'%tag, '', 60, -0.1, 1.1, 300,0,3000)
    #h2_pred_link_true_Np    = rt.TH2F('h2_pred_link_true_Np_tmp_%s'%tag, '', 60, -0.1, 1.1, 20,0,10)
    #h_Np  = rt.TH1F('h_Np_tmp_%s'%tag, '', 100, 0, 10)
    #h2_Np_Nl  = rt.TH2F('h_Np_Nl_tmp_%s'%tag, '', 10, 0, 10, 6, 0, 6)
    #h_mass  = rt.TH1F('h_mass_tmp_%s'%tag, '', 320, 0, 3200)
    #h2_real_pred  = rt.TH2F('h2_real_pred_%s'%tag, '', 100, 0, 1, 100, 0, 1)
    #h2_real_pred_px   = rt.TH2F('h2_real_pred_px_%s'%tag, '', 220, -1.1, 1.1, 220, -1.1, 1.1)
    #h2_real_pred_py   = rt.TH2F('h2_real_pred_py_%s'%tag, '', 220, -1.1, 1.1, 220, -1.1, 1.1)
    #h2_real_pred_pz   = rt.TH2F('h2_real_pred_pz_%s'%tag, '', 220, -1.1, 1.1, 220, -1.1, 1.1)
    #h2_real_pred_theta = rt.TH2F('h2_real_pred_theta_%s'%tag, '', 180, 0, 180, 180, 0, 180)
    #h2_real_pred_phi   = rt.TH2F('h2_real_pred_phi_%s'  %tag, '', 180, -180, 180, 180, -180, 180)
    #t_scale = 40
    #dict_mo_pid_count ={}
    #dict_mo_pid ={}
    #dict_mo_pid_not_found =[]
    #y_true = []
    #y_pred = []
    print('df_data shape=',df_data.shape)
    for i in range(df_data.shape[0]):
        for j in range(df_data.shape[1]):
            h_hittime_cor.SetBinContent(j, df_data[i,j]+h_hittime_cor.GetBinContent(j))
        #########################
    return [h_hittime_cor]

def draw_roc(y_true, y_pred, leg_name='example estimator', out_name='roc'):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=leg_name, pos_label=1)
    display.plot()
    #plt.savefig("%s/%s.png"%(plots_path,out_name), dpi=DPI)
    plt.savefig("%s/%s.png"%(plots_path,out_name))
    plt.close()


def doAna_iso(df, tag='train', doPlot=False, nPlot=0):
    start_index = 18
    diff=0.02
    prob_cut =0.25#0.2#0.38#0.66 
    prob_cut_list=[0.25, 0.4, 0.48, 0.52, 0.55, 0.63]
    iso_list=[1.022        ,  2.022        , 3.022       ,  4.022        ,  5.022        , 6.022]
    bin_list=[[700,0.7,1.4], [1000,1.7,2.7], [1300,2.7,4], [1400,3.8,5.2], [1500,5. ,6.5], [1600,6.1,7.7]]
    h_recE_ep_list = []
    h_recE_ep_pu_list = []
    h_recE_ep_skim_list = []
    h_recE_ep_pu_skim_list = []
    h_pred_ep_list =[]
    h_pred_pu_list =[]
    h_recE_ep_pu0_0MeV = rt.TH1F('h_ep_pu0_0MeV_%s'%(tag), '', bin_list[0][0], bin_list[0][1], bin_list[0][2])
    h_c14_nhit = rt.TH1F('h_c14_nhit_%s'%(tag), '', 300, 0, 300)
    for i in range(len(iso_list)):
        h_recE_ep_list.append( rt.TH1F('h_ep_bin_%d_%s'%(i,tag), '', bin_list[i][0], bin_list[i][1], bin_list[i][2]) )
        h_recE_ep_pu_list.append( rt.TH1F('h_ep_pu_bin_%d_%s'%(i,tag), '', bin_list[i][0], bin_list[i][1], bin_list[i][2]) )
        h_recE_ep_skim_list.append( rt.TH1F('h_ep_skim_bin_%d_%s'%(i,tag), '', bin_list[i][0], bin_list[i][1], bin_list[i][2]) )
        h_recE_ep_pu_skim_list.append( rt.TH1F('h_ep_pu_skim_bin_%d_%s'%(i,tag), '', bin_list[i][0], bin_list[i][1], bin_list[i][2]) )
        h_pred_ep_list.append( rt.TH1F('h_pred_ep_bin_%d_%s'%(i,tag), '', 100, 0, 1)  )
        h_pred_pu_list.append( rt.TH1F('h_pred_pu_bin_%d_%s'%(i,tag), '', 100, 0, 1)  )
    h2_dt_edepC14_0MeV      = rt.TH2F('h2_dt_edepC14_0MeV_%s'%tag ,'',20,-1000,1000,4,0,0.2)
    h2_dt_edepC14_pass_0MeV = h2_dt_edepC14_0MeV.Clone('%s_pass'%h2_dt_edepC14_0MeV.GetName())
    print('df.shape[0]=',df.shape[0])
    np.sum(df[:,0])
    for i in range(len(iso_list)):
        n_pass = np.sum( np.logical_and(df[:,0]>iso_list[i]-diff, df[:,0]<iso_list[i]+diff) )
        print('bin=',i,',n_pass=',n_pass)
    N_tot_pass = 0
    Not_find = 0
    for i in range(df.shape[0]):
        label = df[i,0:start_index]
        pred  = df[i,start_index:]##class prob
        nhitC14 = label[9]
        nhitIBD = label[8] 
        C14_edep = label[4]
        R_C14 = math.sqrt(label[5]*label[5]+label[6]*label[6]+label[7]*label[7])
        dtime_C14 = label[11] #c14_time - trigger_time
        rec_x = label[13]
        rec_y = label[14]
        rec_z = label[15]
        R = math.pow(rec_x*rec_x+rec_y*rec_y+rec_z*rec_z,0.5)
        edep = label[0]
        QTEn = label[12]
        if R > 17200:continue
        if nhitIBD <=0:continue
        ie_index = -1
        for ie in range(len(iso_list)):
            #print(edep,iso_list[ie]-diff,iso_list[ie]+diff)
            if iso_list[ie]-diff < edep and edep < iso_list[ie]+diff:
                ie_index = ie
                break
        if ie_index == -1:
            Not_find += 1
            continue
        if nhitC14>0: h_c14_nhit.Fill(nhitC14 if nhitC14<300 else 299.9)
        N_tot_pass += 1
        h_recE_ep_pu_list[ie_index].Fill(QTEn) 
        prob_cut=prob_cut_list[ie_index]
        if ie_index==0 and nhitC14 <=50:
            h_recE_ep_pu0_0MeV.Fill(QTEn)
        if nhitC14 <=0:
            h_recE_ep_list[ie_index].Fill(QTEn)
            h_pred_ep_list[ie_index].Fill(pred[1] )
        else:
            h_pred_pu_list[ie_index].Fill(pred[1] )
            if ie_index==0:
                tmp_dt =  dtime_C14#time_C14-time_ep
                h2_dt_edepC14_0MeV.Fill(tmp_dt,C14_edep if C14_edep<0.2 else 0.1999)
                if pred[1] > prob_cut:
                    h2_dt_edepC14_pass_0MeV.Fill(tmp_dt,C14_edep if C14_edep<0.2 else 0.1999)
        if pred[1] > prob_cut:
            pass
        else:
            h_recE_ep_pu_skim_list[ie_index].Fill(QTEn)
            if nhitC14 <=0:h_recE_ep_skim_list[ie_index].Fill(QTEn)
    print('N_tot_pass=',N_tot_pass,',not find=',Not_find)
    print('ep bin0=',h_recE_ep_list[0].GetSumOfWeights())
    return h_recE_ep_list, h_recE_ep_pu_list, h_recE_ep_skim_list, h_recE_ep_pu_skim_list, [h2_dt_edepC14_0MeV,h2_dt_edepC14_pass_0MeV], h_pred_ep_list, h_pred_pu_list, h_recE_ep_pu0_0MeV, h_c14_nhit


def plot_h(h1, h2, str_leg, out_name,title, rangeX):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    y_min = 0
    y_max = 1.5*h1.GetBinContent(h1.GetMaximumBin())
    
    dummy = rt.TH2D("dummy","",1, x_min, x_max, 1, y_min, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("")
    h1.SetLineColor(2)
    h1.SetMarkerColor(2)
    h1.SetLineWidth(2)
    h1.Draw("same:hist")
    if h2 is not None:
        h2.SetLineColor(4)
        h2.SetMarkerColor(4)
        h2.SetLineWidth(4)
        h2.Draw("same:hist")
    x_l = 0.6
    y_h = 0.8
    y_dh = 0.1
    x_dl = 0.1
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    if h1 is not None:
        legend.AddEntry(h1 ,'%s'%str_leg['h1']  ,'pl')
    if h2 is not None:
        legend.AddEntry(h2 ,'%s'%str_leg['h2']  ,'pl')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()


def plot_h_v1(h1, h2, str_leg, out_name,title, rangeX, x_sep, norm=False):
    if norm:
        h1.Scale(1.0/h1.GetSumOfWeights())
        if h2 is not None: h2.Scale(1.0/h2.GetSumOfWeights())
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    y_min = 0
    y_max1 = h1.GetBinContent(h1.GetMaximumBin())
    y_max2 = 0 if h2 is None else h2.GetBinContent(h2.GetMaximumBin())
    y_min1 = h1.GetBinContent(h1.GetMinimumBin())
    y_min2 = 0 if h2 is None else h2.GetBinContent(h2.GetMinimumBin())
    
    y_max = 1.5*y_max1 if y_max1 > y_max2 else 1.5*y_max2
    if 'logy' in out_name:
        canvas.SetLogy()
        #y_min = 1e-5 
        y_min = y_min1 if y_min2 > y_min1 else y_min2
        y_min = int( math.log10(y_min) )
        y_min = math.pow(10,y_min-1)
        y_max = y_max1 if y_max2 < y_max1 else y_max2
        y_max = int( math.log10(y_max) )
        y_max = math.pow(10,y_max+1)
        
    dummy = rt.TH2D("dummy","",1, x_min, x_max, 1, y_min, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("")
    h1.SetLineColor(2)
    h1.SetMarkerColor(2)
    h1.SetLineWidth(2)
    h1.Draw("same:hist")
    if h2 is not None:
        h2.SetLineColor(4)
        h2.SetMarkerColor(4)
        h2.SetLineWidth(4)
        h2.Draw("same:hist")
    x_l = 0.6
    y_h = 0.8
    y_dh = 0.1
    x_dl = 0.1
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    if h1 is not None and str_leg['h1'] != '':
        legend.AddEntry(h1 ,'%s'%str_leg['h1']  ,'pl')
    if h2 is not None and str_leg['h2'] != '':
        legend.AddEntry(h2 ,'%s'%str_leg['h2']  ,'pl')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.05)
    legend.SetTextFont(42)
    legend.Draw()
    N_int = h1.Integral(1,h1.GetNbinsX())
    N_sumw2 = h1.GetSumOfWeights()
    #print(f'N_int={N_int},N_sumw2={N_sumw2},ratio={1.0*N_int/N_sumw2}')
    tline_list = []
    for sep in x_sep:
        cen = 0
        sep_bin = 0
        for j in range(h1.GetNbinsX(),2,-1):
            if h1.Integral(1,j)>N_int*sep and h1.Integral(1,j-1)<N_int*sep:
                cen = 0.5*h1.GetXaxis().GetBinCenter(j) + 0.5*h1.GetXaxis().GetBinCenter(j-1)
                sep_bin = j
                break
        tline = rt.TLine(cen,0,cen,0.5*y_max)
        tline.SetLineColor(rt.kBlack)
        tline.SetLineWidth(2)
        tline.Draw()
        tline_list.append(tline)
        if N_int>0: print('sep=',sep,',bin=',sep_bin,',ratio h1=',h1.Integral(1,sep_bin)/N_int,',h2=',h2.Integral(1,sep_bin)/h2.Integral(1,h2.GetNbinsX()) if h2 is not None else -1 )

    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()

def readfiles(in_file):
    filelist=[]
    with open(in_file,'r') as f:
        for line in f.readlines():
            if '#' in line:continue
            line = line.replace('\n','')
            filelist.append(line)
    return filelist

def readdf(files):
    df_data = None
    for f in files:
        hf = h5py.File(f, 'r')
        tmp_df_data = hf['data_1D'][:]
        hf.close()
        df_data = tmp_df_data if df_data is None else np.concatenate((df_data, tmp_df_data), axis=0) 
    return np.squeeze(df_data)

def do_plot2D(hist,out_name,title, draw_opt='',decimal='.0f'):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #h_corr.Draw("COLZ")
    #h_corr.LabelsDeflate("X")
    #h_corr.LabelsDeflate("Y")
    #h_corr.LabelsOption("v")
    hist.SetStats(rt.kFALSE)
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitleOffset(1.2)
    rt.gStyle.SetPaintTextFormat('%s'%decimal)
    hist.Draw("COLZ %s"%draw_opt)
    canvas.SaveAs("%s/%s.png"%(plots_path, out_name))
    del canvas
    gc.collect()

def do_fit(hist, fitrange):
    if hist.GetSumOfWeights() < 100 : return hist, [-1,0,0,0,0]
    out_para = []
    mean = hist.GetMean()
    rms  = hist.GetRMS()
    lower  = mean - 2*rms if fitrange is None else fitrange[0]
    higher = mean + 2*rms if fitrange is None else fitrange[1]
    f1 = rt.TF1("f1", "gaus", lower, higher)
    f1.SetParameters(0.0, 5.0)
    #result = hist.Fit('f1','RLS0')
    result = hist.Fit('f1','QRLS0')
    status = -1
    try:
        status = result.Status()
    except:
        status = -1
    else:
        status = 0
    if status == 0:
        par0   = result.Parameter(0)
        err0   = result.ParError(0)
        par1   = result.Parameter(1)
        err1   = result.ParError(1)
        par2   = result.Parameter(2)
        err2   = result.ParError (2)
        ######### do it again ###### 
        mean = par1
        rms  = par2
        lower  = mean - 4*rms if fitrange is None else fitrange[0]
        higher = mean + 4*rms if fitrange is None else fitrange[1]
        f2 = rt.TF1("f2", "gaus", lower, higher)
        f2.SetParameters(par1, par2)
        #f2.SetLineColor(hist.GetLineColor())
        f2.SetLineColor(rt.kRed)
        #result = hist.Fit('f2','RLS')
        #result = hist.Fit('f2','QRLS')##L means likelihood, default chi_2
        result = hist.Fit('gaus','QS') if sigma4Fit==False else hist.Fit('f2','QRS')
        chi2oNdf = hist.GetFunction("f2").GetChisquare()/hist.GetFunction("f2").GetNDF() if (hist.GetFunction("f2") and hist.GetFunction("f2").GetNDF()!=0) else 0
        print('fit 1 for %s with entry %f mean=%f, rms=%f, lower=%f, higher=%f'%(hist.GetName(), hist.GetSumOfWeights(), hist.GetMean(), hist.GetRMS(), lower, higher))
        status = -1
        try:
            status = result.Status()
        except:
            status = -1
        else:
            status = 0
        if status == 0:
            par0   = result.Parameter(0)
            err0   = result.ParError(0)
            par1   = result.Parameter(1)
            err1   = result.ParError(1)
            par2   = result.Parameter(2)
            err2   = result.ParError (2)
            out_para.append(par1)
            out_para.append(err1)
            out_para.append(par2)
            out_para.append(err2)
            out_para.append(chi2oNdf)
        else:
            out_para.append(-1)
            out_para.append(0)
            out_para.append(0)
            out_para.append(0)
            out_para.append(0)
            print('failed fit 1 for %s with entry %f, lower=%f, higher=%f'%(hist.GetName(), hist.GetSumOfWeights(), lower, higher))
    else:
        out_para.append(-1)
        out_para.append(0)
        out_para.append(0)
        out_para.append(0)
        out_para.append(0)
        print('failed fit 0 for %s with entry %f'%(hist.GetName(), hist.GetSumOfWeights()))
    return hist, out_para

def plot_h1(hist,out_name,title, doFit, fitfun, fitrange=None, clines=None, draw_opt='ep', ep_info=None, C14_info=None, decimal='0.f'):
    rt.gStyle.SetPaintTextFormat('%s'%decimal)
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    #canvas.SetGridy()
    #canvas.SetGridx()
    x_axis = hist.GetXaxis()
    x_min = x_axis.GetXmin() 
    x_max = x_axis.GetXmax() 
    y_max = 1.5*hist.GetBinContent(hist.GetMaximumBin())
    dummy = rt.TH2D("dummy_%s"%out_name,"",1, x_min, x_max, 1, 0, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetTitleOffset(1.)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("")
    results = [-1,0,0,0]
    if doFit == False:
        #hist.Draw("same:hist")
        hist.Draw("same:%s"%draw_opt)
    else:
        hist1 = None
        out_para = None
        if fitfun=='gaus' : hist1, out_para = do_fit(hist, fitrange) 
        hist1.SetStats(rt.kFALSE)
        hist1.GetYaxis().SetTitle(title['Y'])
        hist1.GetXaxis().SetTitle(title['X'])
        hist1.GetYaxis().SetTitleSize(0.04)
        hist1.GetXaxis().SetTitleSize(0.04)
        hist1.GetYaxis().SetLabelSize(0.04)
        hist1.GetXaxis().SetLabelSize(0.04)
        hist1.GetYaxis().SetTitleOffset(1.7)
        hist1.GetXaxis().SetTitleOffset(1.1)
        hist1.GetXaxis().SetMoreLogLabels()
        hist1.GetXaxis().SetTitleFont(42)
        hist1.GetXaxis().SetLabelFont(42)
        hist1.GetXaxis().SetNdivisions(405)
        hist1.GetYaxis().SetTitleFont(42)
        hist1.GetYaxis().SetLabelFont(42)

        hist1.Draw("same:pe")
        label1 = rt.TLatex(0.8 , 0.9, "mean=%.4f#pm%.4f,#sigma=%.5f#pm%.5f"%(out_para[0], out_para[1], out_para[2], out_para[3]))
        #print("mean=%.4f#pm%.4f,#sigma=%.4f#pm%.4f"%(out_para[0], out_para[1], out_para[2], out_para[3]) )
        label1.SetTextAlign(32)
        label1.SetTextSize(0.035)
        label1.SetNDC(rt.kTRUE)
        label1.Draw() 
        #label2 = rt.TLatex(0.8 , 0.8, "#chi^{2}/NDF=%.4f"%(out_para[4]))
        #label2 = rt.TLatex(0.8 , 0.8, "Events=%d, #chi^{2}/NDF=%.4f"%(hist.GetSumOfWeights(),out_para[4]) ) 
        label2 = rt.TLatex(0.8 , 0.7, "Events=%d, #chi^{2}/NDF=%.4f"%(hist.GetSumOfWeights(),out_para[4]) ) 
        label2.SetTextAlign(32)
        label2.SetTextSize(0.035)
        label2.SetNDC(rt.kTRUE)
        label2.Draw() 
        results[0] = out_para[0]
        results[1] = out_para[1]
        results[2] = out_para[2]
        results[3] = out_para[3]
    #Tlines = []
    #line_thre = rt.TLine(x_min,cls_threshold,x_max,cls_threshold)
    #line_thre.SetLineColor(1)
    #if clines != None:
    #    for i in range(len(clines[0])):
    #        lx1 = clines[0][i] 
    #        lx2 = clines[0][i]
    #        ly1 = 0 
    #        ly2 = y_max  
    #        tmp_line = rt.TLine(lx1,ly1,lx2,ly2)
    #        tmp_line.SetLineColor(2 if clines[1][i]=='up' else 8)
    #        tmp_line.Draw()
    #        Tlines.append(tmp_line)
    #    line_thre.Draw()
    label_ep = None
    if ep_info != None:
        tmp_r = math.sqrt( math.pow(ep_info[0],2) + pow(ep_info[1],2) + pow(ep_info[2],2) )
        label_ep = rt.TLatex(0.8 , 0.9, "e^{+}: edep (x:%.1f,y:%.1f,z:%.1f,r=%.1f) (m), %.2f (MeV), t_{mix} %d (ns)"%(ep_info[0]/1000., ep_info[1]/1000., ep_info[2]/1000., tmp_r/1000., ep_info[4], ep_info[3]))
        label_ep.SetTextAlign(32)
        label_ep.SetTextSize(0.02)
        label_ep.SetNDC(rt.kTRUE)
        label_ep.Draw()
    label_C14 = None
    if C14_info != None:
        tmp_r = math.sqrt( math.pow(C14_info[0],2) + pow(C14_info[1],2) + pow(C14_info[2],2) )
        label_C14 = rt.TLatex(0.8 , 0.95, "C14: edep (x:%.1f,y:%.1f,z:%.1f,r=%.1f) (m), %.2f (MeV), t_{mix} %d (ns)"%(C14_info[0]/1000., C14_info[1]/1000., C14_info[2]/1000., tmp_r/1000., C14_info[4], C14_info[3]))
        label_C14.SetTextAlign(32)
        label_C14.SetTextSize(0.02)
        label_C14.SetNDC(rt.kTRUE)
        label_C14.Draw()

    canvas.SaveAs("%s/%s.png"%(plots_path,out_name) )
    del dummy
    del canvas
    gc.collect()
    return results

def get_hs_dict(h_list, ext_str='training',draw_opt='hist'):
    hs_dict={}
    for h in h_list:
        hname = h.GetName()
        pid0 = hname.split('__')[1]
        pid1 = hname.split('__')[2]
        leg_name = '%s:%s->%s'%(ext_str,pid_dict[pid0], pid_dict[pid1])
        hs_dict[leg_name] = [h, color_dict['%s__%s'%(pid0,pid1)], draw_opt]
    return hs_dict
 
def get_grs_dict(h_dict, ext_str='training',draw_opt='hist'):
    hs_dict={}
    for hname in h_dict:
        h = h_dict[hname]
        pid0 = hname.split('__')[1]
        pid1 = hname.split('__')[2]
        leg_name = '%s:%s->%s'%(ext_str,pid_dict[pid0], pid_dict[pid1])
        hs_dict[leg_name] = [h, color_dict['%s__%s'%(pid0,pid1)], draw_opt]
    return hs_dict
 


def get_eff(hlist):
    h_all={}
    for h in hlist:
        hname = h.GetName()
        pid0 = hname.split('__')[1]
        pid1 = hname.split('__')[2]
        if pid0 != pid1:continue
        h_all[pid0] = h.Clone('%s_all_%s'%(h.GetName(),pid0))
        h_all[pid0].Scale(.0)
    for h in hlist:
        hname = h.GetName()
        pid0 = hname.split('__')[1]
        pid1 = hname.split('__')[2]
        h_all[pid0].Add(h)
    h_eff_dict={}
    for ha in h_all:
        for h in hlist:
            hname = h.GetName()
            pid0 = hname.split('__')[1]
            pid1 = hname.split('__')[2]
            if pid0 != ha: continue
            gr_eff = rt.TGraphAsymmErrors()
            gr_eff.Divide(h, h_all[ha], "cl=0.683 b(1,1) mode")
            gr_eff.SetName(hname)
            h_eff_dict[hname] = gr_eff
    return h_eff_dict

def read_pdg_table(file):
    dict_out = {}
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'add  p' not in line: continue
            items = line.split()
            name  = items[3]
            pdgid  = int(float(items[4]) )
            mass   = 1000*float(items[5])
            width  = 1000*float(items[6])
            charge = float(items[8])/3.
            #print(name,pdgid)
            dict_out[pdgid]=[name,mass,width,charge]
    return dict_out

if __name__ == '__main__':
    print('Starting ...')
    pwd_path = os.getcwd()
    plots_path = '%s/plots_train_data/'%pwd_path
    if os.path.exists(plots_path) == False: os.makedirs(plots_path) 
    parser = get_parser()
    parse_args = parser.parse_args()
    ftrain = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/dataset/detsim_m10ns_earlys0p05_points_add_1d/train_bb0n.txt'
    train_files = readfiles(ftrain)
    ftest = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/dataset/detsim_m10ns_earlys0p05_points_add_1d/train_e-.txt'
    test_files = readfiles(ftest)
    df_data_train = readdf(train_files)
    df_data_test  = readdf(test_files )

    h1_real_pred_train  =  doAna(df_data = df_data_train, tag='sig')
    h1_real_pred_test   =  doAna(df_data = df_data_test , tag='bkg' )
    str_sig = '#nu0bb'
    str_bkg = 'e^{-}'
    leg_train = 'hist'
    leg_test  = 'pel'
    str_pred = 'hittime_cor'
    hs_dict_pred = {}
    hs_dict_pred[str_sig  ]=[h1_real_pred_train[0] ,2,leg_train]
    hs_dict_pred[str_bkg  ]=[h1_real_pred_test [0] ,4,leg_test ]
    plot_hs(hs_dict=hs_dict_pred, norm=False, out_name='comp_%s'%h1_real_pred_test[0].GetName(), title={'X':str_pred,'Y':'N hit'}, rangeX=[-10,40], x_sep=[])
    plot_hs(hs_dict=hs_dict_pred, norm=True, out_name='norm_comp_%s'%h1_real_pred_test[0].GetName(), title={'X':str_pred,'Y':'N hit'}, rangeX=[-10,40], x_sep=[])
    print('Done')
