import ROOT as rt
import numpy as np
import h5py
import math
import sys
import argparse
import gc
import ast
import random
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);
rt.TH1.AddDirectory(0)
# For Ge68 use track size == 1 event
# add info with first hit time and max npe of pmt and charge center
def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO study. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch_size', action='store', type=int, default=5000,
                        help='Number of event for each batch.')
    parser.add_argument('--m_max_evt', action='store', type=int, default=-1,
                        help='max_evt Number')
    parser.add_argument('--input', action='store', type=str, default='',
                        help='input root file.')
    parser.add_argument('--doReWeight', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--doEcut', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--E_min' , action='store', type=float, default=2.3, help='')
    parser.add_argument('--E_max' , action='store', type=float, default=2.4, help='')
    parser.add_argument('--reweight_input_0', action='store', type=str, default='',
                        help='input TH1F for reweighting.')
    parser.add_argument('--reweight_input_1', action='store', type=str, default='',
                        help='input TH1F for reweighting.')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output hdf5 file.')
    parser.add_argument('--out_root', action='store', type=str, default='hist.root',
                        help='output root for histograms')
    parser.add_argument('--addTTS', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--addDN', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--addEnergySmear', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--addVertexSmear', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--addT0Smear', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--T0_sigma' , action='store', type=float, default=2., help='')
    parser.add_argument('--TTS_realistic', action='store', type=ast.literal_eval, default=True, help='')
    parser.add_argument('--DoPlot', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--draw_xy_ck', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--gen_r_cut' , action='store', type=float, default=1., help='')
    parser.add_argument('--DoAnaCalib', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SaveH5', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Save2D', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Save2D_detsim', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SavePoints_detsim', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SavePoints_detsim_v2', action='store', type=ast.literal_eval, default=False, help='saving some early p.e.')
    parser.add_argument('--SavePoints_detsim_v3', action='store', type=ast.literal_eval, default=False, help='add 1D hist')
    parser.add_argument('--SavePoints_calib', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--SavePoints_calib_v2', action='store', type=ast.literal_eval, default=False, help='saving some early p.e.')
    parser.add_argument('--save_dir_random', action='store', type=ast.literal_eval, default=False, help='saving direction info. randomly')
    parser.add_argument('--save_dir_gen'   , action='store', type=ast.literal_eval, default=False, help='saving direction info. from gen')
    parser.add_argument('--smear_gen'   , action='store', type=ast.literal_eval, default=False, help='smearing direction  from gen')
    parser.add_argument('--direction_smear_input', action='store', type=str, default='.root', help='')
    parser.add_argument('--use_dir_cut'   , action='store', type=ast.literal_eval, default=False, help='use direction info. cut hits')
    parser.add_argument('--cosAngle_min' , action='store', type=float, default=0., help='')
    parser.add_argument('--sim_ratio' , action='store', type=float, default=0.1, help='')
    parser.add_argument('--calib_ratio' , action='store', type=float, default=0.1, help='')
    parser.add_argument('--Save_only_ck', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Draw_data', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--Draw_CK', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--only_Hama', action='store', type=ast.literal_eval, default=False, help='')
    parser.add_argument('--time_cut', action='store', type=float, default=1e6, help='')
    parser.add_argument('--time_cut_down', action='store', type=float, default=-1e6, help='')
    parser.add_argument('--calibtime_cut_low' , action='store', type=float, default=-1e6, help='')
    parser.add_argument('--calibtime_cut_high', action='store', type=float, default=1e6, help='')
    parser.add_argument('--time_1D_low' , action='store', type=int, default=-10, help='')
    parser.add_argument('--time_1D_high', action='store', type=int, default= 40, help='')

    return parser


def plot_hs(hs_dict, norm, out_name, title, rangeX, x_sep):#hs_dict={'leg_name':[hist,color,drawoption]}
    tmp_max_y = 0
    tmp_min_y = 999
    if norm:
        for i in hs_dict:
            hs_dict[i][0].Scale(1.0/hs_dict[i][0].GetSumOfWeights())
    for i in hs_dict:
        hs_dict[i][0].SetLineWidth(2)
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
        #hs_dict[i][0].Draw("same:pe")
        hs_dict[i][0].Draw("same:%s"%(hs_dict[i][2]))
    x_l = 0.18
    y_h = 0.85
    y_dh = 0.2
    x_dl = 0.1
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    for i in hs_dict:
        if i == '':continue
        legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'pel' if ('ep' in hs_dict[i][2] or 'pe' in hs_dict[i][2]) else 'pl')
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

def fun_rot_phi(ori_x, ori_y, rot):
    tmp_r = math.sqrt(ori_x*ori_x + ori_y*ori_y)
    if tmp_r==0:
        return (ori_x,ori_y)
    tmp_phi = math.asin(ori_y/tmp_r)*180/math.pi if ori_x > 0 else 180-math.asin(ori_y/tmp_r)*180/math.pi
    tmp_phi_new = tmp_phi + rot
    if tmp_phi_new < 0:
        tmp_phi_new = tmp_phi_new+360
    elif tmp_phi_new > 360:
        tmp_phi_new = tmp_phi_new-360
    tmp_new_x = tmp_r*math.cos(tmp_phi_new*math.pi/180)
    tmp_new_y = tmp_r*math.sin(tmp_phi_new*math.pi/180)
    return (tmp_new_x,tmp_new_y)

def fun_rot_theta(ori_x, ori_z, rot):
    tmp_r = math.sqrt(ori_x*ori_x + ori_z*ori_z)
    tmp_theta = math.acos(ori_z/tmp_r)*180/math.pi
    tmp_theta_new = tmp_theta + rot
    if ori_x > 0:
        if 0<=tmp_theta_new and tmp_theta_new <= 180:
            new_x = tmp_r*math.sin(tmp_theta_new*math.pi/180)
            new_z = tmp_r*math.cos(tmp_theta_new*math.pi/180)
        elif tmp_theta_new < 0:
            new_z = tmp_r*math.cos(tmp_theta_new*math.pi/180)
            new_x = tmp_r*math.sin(tmp_theta_new*math.pi/180)
        elif tmp_theta_new > 180:
            tmp_theta_new = -( 180 - (tmp_theta_new - 180) )
            new_z = tmp_r*math.cos(tmp_theta_new*math.pi/180)
            new_x = tmp_r*math.sin(tmp_theta_new*math.pi/180)
    else:
        tmp_theta_new = -tmp_theta + rot
        if tmp_theta_new < -180:
            tmp_theta_new = 180-(-tmp_theta_new - 180)
            new_z = tmp_r*math.cos(tmp_theta_new*math.pi/180)
            new_x = tmp_r*math.sin(tmp_theta_new*math.pi/180)
        else:
            new_z = tmp_r*math.cos(tmp_theta_new*math.pi/180)
            new_x = tmp_r*math.sin(tmp_theta_new*math.pi/180)
            
   
    return (new_x, new_z)

def ID_type_map(file_pos, i_id):
    with open(file_pos,'r') as f:
        id_type_dict = {}
        lines = f.readlines()
        for line in lines:
            items = line.split()
            ID    = float(items[i_id])
            ID    = int(ID)
            str_type  = items[1]
            id_type_dict[ID] = str_type
        return id_type_dict


def AnaCalib (tree, max_evt):
    print('Perform AnaCalib')
    #h_sim_npeByPMT     = rt.TH1F('h_sim_npeByPMT','N p.e. by PMT',20,0,10)
    h_calib_npeByPMT   = rt.TH1F('h_calib_npeByPMT','N p.e. by PMT',20,0,10)
    h_calib_tcor      = rt.TH1F('h_calib_tcor'     ,'hittime_{cor} (ns)',300,200,500)
    h_calib_tcor_ck_ori  = rt.TH1F('h_calib_tcor_ck_ori','hittime_{cor} (ns)',100,-50,50)
    h_calib_tcor_ck_ori_Hama  = rt.TH1F('h_calib_tcor_ck_ori_Hama','hittime_{cor} (ns)',100,-50,50)
    h_calib_tcor_ck_ori_NNVT  = rt.TH1F('h_calib_tcor_ck_ori_NNVT','hittime_{cor} (ns)',100,-50,50)

    #h_resE = rt.TH1F('h_resE','(rec_{E}-mc_{E})/mc_{E} (%)',300,-10,20)
    #h_calib_tcor_Hama = rt.TH1F('h_calib_tcor_Hama','hittime_{cor} (ns)',300,200,500)
    #h_calib_tcor_Hama_r3_1000_d10m = rt.TH1F('h_calib_tcor_Hama_r3_1000_d10m','hittime^{r^{3}<1000,d=10m}_{cor} (ns)',300,200,500)
    #h_calib_tcor_NNVT_r3_1000_d10m = rt.TH1F('h_calib_tcor_NNVT_r3_1000_d10m','hittime^{r^{3}<1000,d=10m}_{cor} (ns)',300,200,500)
    #h_calib_tcor_NNVT = rt.TH1F('h_calib_tcor_NNVT','hittime_{cor} (ns)',300,200,500)
    #h_calib_tcor_Hama_one = rt.TH1F('h_calib_tcor_Hama_one','hittime_{cor} (ns)',150,200,500)
    #h_calib_tcor_ck_ori = rt.TH1F('h_calib_tcor_ck_ori','hittime_{cor} (ns)',300,200,500)
    #h_sim_tcor        = rt.TH1F('h_sim_tcor'       ,'hittime_{cor} (ns)',50,-10,40)
    #h_sim_tcor_CK     = rt.TH1F('h_sim_tcor_CK'    ,'hittime_{cor} (ns)',50,-10,40)
    #h_sim_tcor_CK_ori = rt.TH1F('h_sim_tcor_CK_ori','hittime_{cor} (ns)',50,-10,40)
    #h_sim_tcor_Hama_r3_1000_d10m = rt.TH1F('h_sim_tcor_Hama_r3_1000_d10m','hittime^{r^{3}<1000,d=10m}_{cor} (ns)',300,200,500)
    #h_sim_tcor_NNVT_r3_1000_d10m = rt.TH1F('h_sim_tcor_NNVT_r3_1000_d10m','hittime^{r^{3}<1000,d=10m}_{cor} (ns)',300,200,500)
    #h_sim_tcor_Hama_one = rt.TH1F('h_sim_tcor_Hama_one','hittime_{cor} (ns)',150,200,500)
    #h_nhit = rt.TH1F('h_nhit','nhit(t_{cor} %d-%d ns)'%(m_tcor_low,m_tcor_high),100,0,4000)
    #h_ith_tcor_CK_ori      = rt.TH1F('h_ith_tcor_CK_ori','i^{th} hit(t_{cor})',400,0,4000)
    #h_ith_tcor_non_CK_ori  = rt.TH1F('h_ith_tcor_non_CK_ori','i^{th} hit(t_{cor})',400,0,4000)
    #h_Hama_ith_tcor_CK_ori      = rt.TH1F('h_Hama_ith_tcor_CK_ori','i^{th} hit(t_{cor})',400,0,4000)
    #h_Hama_ith_tcor_non_CK_ori  = rt.TH1F('h_Hama_ith_tcor_non_CK_ori','i^{th} hit(t_{cor})',400,0,4000)
    #N_evt = 0
    #N_total = 0
    #N_ck = 0
    #N_ck_ori = 0
    #N_ck_ori_Hama = 0
    #N_ck_ori_NNVT = 0
    #N_ck_ori_SMPT = 0
    #e_min = 2.
    #e_max = 3.
    for ie in range(0, max_evt):
        #print('for evt=',ie)
        tree.GetEntry(ie)
        recx = getattr(tree, "recQTx")
        recy = getattr(tree, "recQTy")
        recz = getattr(tree, "recQTz")
        rect0 = getattr(tree, "QTt0")
        recE = getattr(tree, "QTEn")
        simx = getattr(tree, "sim_QedepX")
        simy = getattr(tree, "sim_QedepY")
        simz = getattr(tree, "sim_QedepZ")
        simE = getattr(tree, "sim_Qedep")
        calib_charges = getattr(tree, "calib_charges")
        calib_times   = getattr(tree, "calib_times")
        calib_PMTIDs  = getattr(tree, "calib_PMTIDs")
        simhit_times  = getattr(tree, "simhit_times")
        simhit_PMTIDs = getattr(tree, "simhit_PMTIDs")
        simhit_isCerenkov = getattr(tree, "simhit_isCerenkov")
        simhit_isOriginalOP = getattr(tree, "simhit_isOriginalOP")
        tmp_npeByPmt = {}
        ########### FIXME, use sim info here
        recx = simx
        recy = simy
        recz = simz
        recE = simE
        ########### 

        for ih in range(len(calib_PMTIDs)):
            tmp_pmt_id = int(calib_PMTIDs[ih])
            if tmp_pmt_id > 17611: continue
            if tmp_pmt_id not in tmp_npeByPmt:
                tmp_npeByPmt[tmp_pmt_id] = 1
            else:
                tmp_npeByPmt[tmp_pmt_id] += 1

            pmt_x = m_GeoSvc.get_pmt_x(tmp_pmt_id)
            pmt_y = m_GeoSvc.get_pmt_y(tmp_pmt_id)
            pmt_z = m_GeoSvc.get_pmt_z(tmp_pmt_id)
            dist = math.sqrt(math.pow(pmt_x-recx,2)+math.pow(pmt_y-recy,2)+math.pow(pmt_z-recz,2) )
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
            timecor = calib_times[ih]-tof 
            h_calib_tcor.Fill(timecor)
        max_bin = h_calib_tcor.GetMaximumBin() 
        tmp_peak = h_calib_tcor.GetBinCenter(max_bin)
        for i in range(17612):
            if i in tmp_npeByPmt:
                h_calib_npeByPMT.Fill(tmp_npeByPmt[i])
            else:
                h_calib_npeByPMT.Fill(0)
        ############### sim ############################# 
        tmp_ori_ck = {}
        for ih in range(len(simhit_PMTIDs)):
            tmp_pmt_id = int(simhit_PMTIDs[ih])
            if tmp_pmt_id > 17611: continue
            pmt_x = m_GeoSvc.get_pmt_x(tmp_pmt_id)
            pmt_y = m_GeoSvc.get_pmt_y(tmp_pmt_id)
            pmt_z = m_GeoSvc.get_pmt_z(tmp_pmt_id)
            dist = math.sqrt(math.pow(pmt_x-simx,2)+math.pow(pmt_y-simy,2)+math.pow(pmt_z-simz,2) )
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=simx, evty=simy, evtz=simz)
            timecor = simhit_times[ih]-tof 
            #h_sim_tcor.Fill(timecor)
            if simhit_isCerenkov[ih] == 1 and  simhit_isOriginalOP[ih] == 1:
                if tmp_pmt_id not in tmp_ori_ck: tmp_ori_ck[tmp_pmt_id] = []
                tmp_ori_ck[tmp_pmt_id].append(timecor+tmp_peak)
 
        for ih in range(len(calib_PMTIDs)):
            tmp_pmt_id = int(calib_PMTIDs[ih])
            if tmp_pmt_id in tmp_ori_ck:
                pmt_x = m_GeoSvc.get_pmt_x(tmp_pmt_id)
                pmt_y = m_GeoSvc.get_pmt_y(tmp_pmt_id)
                pmt_z = m_GeoSvc.get_pmt_z(tmp_pmt_id)
                dist = math.sqrt(math.pow(pmt_x-recx,2)+math.pow(pmt_y-recy,2)+math.pow(pmt_z-recz,2) )
                tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
                timecor = calib_times[ih]-tof 
                h_calib_tcor_ck_ori.Fill(timecor-tmp_peak)
                pmt_type = m_Id_type_dict[tmp_pmt_id]
                if pmt_type == 'Hamamatsu': h_calib_tcor_ck_ori_Hama.Fill(timecor-tmp_peak)
                else                      : h_calib_tcor_ck_ori_NNVT.Fill(timecor-tmp_peak)

    return [h_calib_npeByPMT, h_calib_tcor_ck_ori, h_calib_tcor_ck_ori_Hama, h_calib_tcor_ck_ori_NNVT] 

def draw_sim (tree, max_evt):
    print('Perform drawing')
    h_dict = {}
    h_dict['h_sim_n_ck_ori_early'   ] = rt.TH1F('h_sim_n_ck_ori_early' ,'n ck_ori',20,0,20)
    h_dict['h_sim_ck_ori_early_time'] = rt.TH1F('h_sim_ck_ori_early_time' ,'hittime_{cor} (ns)',40,-20,20)
    h_dict['h_sim_s2b_early'        ] = rt.TH1F('h_sim_s2b_early' ,'n ck_ori/npe',10,0,0.1)
    h_dict['h_sim_tcor'             ] = rt.TH1F('h_sim_tcor'       ,'hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_Hama'        ] = rt.TH1F('h_sim_tcor_Hama'  ,'hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_NNVT'        ] = rt.TH1F('h_sim_tcor_NNVT'  ,'hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_CK'          ] = rt.TH1F('h_sim_tcor_CK'    ,'hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_CK_ori'      ] = rt.TH1F('h_sim_tcor_CK_ori','hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_CK_ori_Hama' ] = rt.TH1F('h_sim_tcor_CK_ori_Hama','hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_tcor_CK_ori_NNVT' ] = rt.TH1F('h_sim_tcor_CK_ori_NNVT','hittime_{cor} (ns)',60,-20,40)
    h_dict['h_sim_CK_ori_pass_ratio'] = rt.TH1F('h_sim_CK_ori_pass_ratio','CK_ori_pass_ratio',100,0,1)
    h_dict['h_sim_npe_CK_ori_Hama'  ] = rt.TH1F('h_sim_npe_CK_ori_Hama','N p.e.',40,0,40)
    h_dict['h_sim_npe_CK_ori_NNVT'  ] = rt.TH1F('h_sim_npe_CK_ori_NNVT','N p.e.',40,0,40)
    h_dict['h_sim_totnpe'           ] = rt.TH1F('h_sim_totnpe','N p.e.',300,3000,6000)
    h_dict['h_sim_Qedep'            ] = rt.TH1F('h_sim_Qedep','Qedep (MeV)',100,2.3,2.4)
    h_dict['h_gen_N'                ] = rt.TH1F('h_gen_N','N_{gen}',20, 0 ,10)
    h_dict['h_gen_x'                ] = rt.TH1F('h_gen_x','x(m)',90,-18,18)
    h_dict['h_gen_y'                ] = rt.TH1F('h_gen_y','y(m)',90,-18,18)
    h_dict['h_gen_z'                ] = rt.TH1F('h_gen_z','z(m)',90,-18,18)
    h_dict['h_gen_r3'               ] = rt.TH1F('h_gen_r3','r^{3}(m)',100,0,5000)
    h_dict['h_gen_p'                ] = rt.TH1F('h_gen_p' ,'p (MeV)',40,0,4)
    h_dict['h_gen_px'               ] = rt.TH1F('h_gen_px','px (MeV)',30,0,3)
    h_dict['h_gen_py'               ] = rt.TH1F('h_gen_py','py (MeV)',30,0,3)
    h_dict['h_gen_pz'               ] = rt.TH1F('h_gen_pz','pz (MeV)',30,0,3)
    h_dict['h_gen_theta'            ] = rt.TH1F('h_gen_theta','#theta',90,0   ,180)
    h_dict['h_gen_phi'              ] = rt.TH1F('h_gen_phi','#phi'    ,90,-180,180)
    h_dict['h_gen_costheta'         ] = rt.TH1F('h_gen_costheta','cos#theta',100,-1,1)
    h_dict['h_gen_cosdangle'        ] = rt.TH1F('h_gen_cosdangle','cos(#DeltaAngle)',100,-1,1)
    h_dict['h_cosdangle_gen_ck'     ] = rt.TH1F('h_cosdangle_gen_ck','cos(#DeltaAngle)',100,-1,1)
    h_dict['h_dangle_gen_ck'        ] = rt.TH1F('h_dangle_gen_ck'   ,'#DeltaAngle',180,0,180)
    h_dict['h_dangle_gen_sel'        ] = rt.TH1F('h_dangle_gen_sel'   ,'#DeltaAngle',180,0,180)
    h_dict['h_cosdangle_gen_sel'     ] = rt.TH1F('h_cosdangle_gen_sel','cos(#DeltaAngle)',100,-1,1)

    h_dict['h_gen2_x'                ] = rt.TH1F('h_gen2_x','x(m)',90,-18,18)
    h_dict['h_gen2_y'                ] = rt.TH1F('h_gen2_y','y(m)',90,-18,18)
    h_dict['h_gen2_z'                ] = rt.TH1F('h_gen2_z','z(m)',90,-18,18)
    h_dict['h_gen2_r3'               ] = rt.TH1F('h_gen2_r3','r^{3}(m)',100,0,5000)
    h_dict['h_gen2_p'                ] = rt.TH1F('h_gen2_p' ,'p (MeV)',40,0,4)
    h_dict['h_gen2_px'               ] = rt.TH1F('h_gen2_px','px (MeV)',30,0,3)
    h_dict['h_gen2_py'               ] = rt.TH1F('h_gen2_py','py (MeV)',30,0,3)
    h_dict['h_gen2_pz'               ] = rt.TH1F('h_gen2_pz','pz (MeV)',30,0,3)
    h_dict['h_gen2_theta'            ] = rt.TH1F('h_gen2_theta','#theta',90,0   ,180)
    h_dict['h_gen2_phi'              ] = rt.TH1F('h_gen2_phi','#phi'    ,90,-180,180)
    h_dict['h_gen2_costheta'         ] = rt.TH1F('h_gen2_costheta','cos#theta',100,-1,1)
    h_dict['h2_gen_KE1_KE2'          ] = rt.TH2F('h2_gen_KE1_KE2','',30,0,3,30,0,3)
 
    h_dict['h_dn_time'              ] = rt.TH1F('h_dn_time','hit time (ns)',110,-100,1000)
    h_dict['h_dn_tcor'              ] = rt.TH1F('h_dn_tcor','hit time_{cor} (ns)',2000,-1000,1000)
    h_dict['gr_sim_earlys_nck_s2b'  ] = rt.TGraph()
    h_dict['gr_sim_xy_ckori'] = []
    N_gr = 10
    for i in range(N_gr):
        h_dict['gr_sim_xy_ckori'].append(rt.TGraph())
    
    h_dict['gr_sim_xy_ckori_sel'] = rt.TGraph()
    N_evt = 0
    N_total = 0
    N_ck = 0
    N_ck_ori = 0
    N_ck_ori_Hama = 0
    N_ck_ori_NNVT = 0
    N_ck_ori_SMPT = 0
    e_min = 2.
    e_max = 3.
    gr_index = 0
    for ie in range(0, max_evt):
        #print('for evt=',ie)
        tree.GetEntry(ie)
        initX                = getattr(tree, "gen_InitX")
        initY                = getattr(tree, "gen_InitY")
        initZ                = getattr(tree, "gen_InitZ")
        initPX               = getattr(tree, "gen_InitPX")
        initPY               = getattr(tree, "gen_InitPY")
        initPZ               = getattr(tree, "gen_InitPZ")
        simx                 = getattr(tree, "sim_QedepX")
        simy                 = getattr(tree, "sim_QedepY")
        simz                 = getattr(tree, "sim_QedepZ")
        simE                 = getattr(tree, "sim_Qedep")
        tmp_simE = 0
        tmp_simx = 0
        tmp_simy = 0
        tmp_simz = 0
        for i in range(len(simE)):
            tmp_simE += simE[i]
            tmp_simx += simx[i]*simE[i]
            tmp_simy += simy[i]*simE[i]
            tmp_simz += simz[i]*simE[i]
        simx = tmp_simx/tmp_simE
        simy = tmp_simy/tmp_simE
        simz = tmp_simz/tmp_simE
        simE = tmp_simE
        simhit_times         = list( getattr(tree, "simhit_times"       ) )
        simhit_PMTIDs        = list( getattr(tree, "simhit_PMTIDs"      ) )
        simhit_isCerenkov    = list( getattr(tree, "simhit_isCerenkov"  ) )
        simhit_isOriginalOP  = list( getattr(tree, "simhit_isOriginalOP") )
        ########## reweighting ########
        if parse_args.doReWeight: 
            we = get_weight(simE)
            if we > 1:
                sv = 1/we
                ran = random.uniform(0, 1)
                if ran > sv:continue
        ########## Gen info ############
        h_dict['h_gen_N'].Fill(len(initX))
        h_dict['h_gen_x'].Fill(initX[0]/1000.)
        h_dict['h_gen_y'].Fill(initY[0]/1000.)
        h_dict['h_gen_z'].Fill(initZ[0]/1000.)
        tmp_gen_r = math.sqrt(initX[0]*initX[0] + initY[0]*initY[0] + initZ[0]*initZ[0])/1000. 
        h_dict['h_gen_r3'].Fill(math.pow(tmp_gen_r,3))
        tmp_gen_p = math.sqrt(initPX[0]*initPX[0] + initPY[0]*initPY[0] + initPZ[0]*initPZ[0])
        me = 0.511 #MeV
        tmp_ke1 = math.sqrt(tmp_gen_p*tmp_gen_p+me*me)-me
        h_dict['h_gen_p' ].Fill(tmp_gen_p)
        h_dict['h_gen_px'].Fill(initPX[0])
        h_dict['h_gen_py'].Fill(initPY[0])
        h_dict['h_gen_pz'].Fill(initPZ[0])
        tmp_gen_v3 = rt.TVector3(initPX[0],initPY[0],initPZ[0])
        h_dict['h_gen_theta']   .Fill(180*tmp_gen_v3.Theta()/math.pi)
        h_dict['h_gen_costheta'].Fill(tmp_gen_v3.CosTheta())
        h_dict['h_gen_phi']     .Fill(180*tmp_gen_v3.Phi  ()/math.pi)
        if parse_args.draw_xy_ck and tmp_gen_r < parse_args.gen_r_cut and  simE > 2.35 and simE < 2.38 and gr_index < len(h_dict['gr_sim_xy_ckori']):
            for ih in range(len(simhit_PMTIDs)):
                iD = int(simhit_PMTIDs[ih])
                if iD > 17611: continue
                pmt_type = m_Id_type_dict[iD]
                if parse_args.only_Hama and pmt_type != 'Hamamatsu': continue
                if simhit_isCerenkov[ih] == 1 and simhit_isOriginalOP[ih] == 1:
                    pmt_x = m_GeoSvc.get_pmt_x(iD)
                    pmt_y = m_GeoSvc.get_pmt_y(iD)
                    pmt_z = m_GeoSvc.get_pmt_z(iD)
                    tmp_pmt_v3 = rt.TVector3(pmt_x,pmt_y,pmt_z)
                    tmp_pmt_v3.RotateZ(-tmp_gen_v3.Phi()); 
                    tmp_pmt_v3.RotateY(-tmp_gen_v3.Theta()); 
                    h_dict['gr_sim_xy_ckori'][gr_index].SetPoint(h_dict['gr_sim_xy_ckori'][gr_index].GetN(), tmp_pmt_v3.X(), tmp_pmt_v3.Y())
            gr_index += 1 
        if len(initX)==2:
            h_dict['h_gen2_x'].Fill(initX[1]/1000.)
            h_dict['h_gen2_y'].Fill(initY[1]/1000.)
            h_dict['h_gen2_z'].Fill(initZ[1]/1000.)
            tmp_gen_r = math.sqrt(initX[1]*initX[1] + initY[1]*initY[1] + initZ[1]*initZ[1])/1000. 
            h_dict['h_gen2_r3'].Fill(math.pow(tmp_gen_r,3))
            tmp_gen_p = math.sqrt(initPX[1]*initPX[1] + initPY[1]*initPY[1] + initPZ[1]*initPZ[1])
            tmp_ke2 = math.sqrt(tmp_gen_p*tmp_gen_p+me*me)-me
            h_dict['h_gen2_p' ].Fill(tmp_gen_p)
            h_dict['h_gen2_px'].Fill(initPX[1])
            h_dict['h_gen2_py'].Fill(initPY[1])
            h_dict['h_gen2_pz'].Fill(initPZ[1])
            tmp_gen0_v3 = rt.TVector3(initPX[0],initPY[0],initPZ[0])
            tmp_gen_v3 = rt.TVector3(initPX[1],initPY[1],initPZ[1])
            h_dict['h_gen2_theta']   .Fill(180*tmp_gen_v3.Theta()/math.pi)
            h_dict['h_gen2_costheta'].Fill(tmp_gen_v3.CosTheta())
            h_dict['h_gen2_phi']     .Fill(180*tmp_gen_v3.Phi  ()/math.pi)
            h_dict['h_gen_cosdangle'].Fill(math.cos(tmp_gen_v3.Angle(tmp_gen0_v3)) )
            h_dict['h2_gen_KE1_KE2'].Fill(tmp_ke1,tmp_ke2)


        ######## add different effects from elecsim and event reconstruction ##################
        if parse_args.addTTS:
            simhit_times = pmt_times_tts(simhit_PMTIDs,simhit_times,parse_args.TTS_realistic)
        if parse_args.addEnergySmear:
            simE = energy_semar(simE)
        if parse_args.addVertexSmear:
            simx, simy, simz = vertex_semar(simE, simx, simy, simz)
        t0 = 0
        if parse_args.addT0Smear:
            t0 = random.gauss(0, parse_args.T0_sigma) 
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                pmt_x = m_GeoSvc.get_pmt_x(int(iD))
                pmt_y = m_GeoSvc.get_pmt_y(int(iD))
                pmt_z = m_GeoSvc.get_pmt_z(int(iD))
                tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=simx, evty=simy, evtz=simz)
                for it in dark_noise_dict[iD]:
                    simhit_PMTIDs.append(iD)
                    simhit_times.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)
                    timecor = it-tof-t0
                    h_dict['h_dn_time'].Fill(it)
                    h_dict['h_dn_tcor'].Fill(timecor)
        ###########################
        N_tot_ck_ori = 0
        simhit_time_cor_sort = []
        simhit_PMTID_sort = []
        simhit_is_ck_ori_sort = []
        tmp_Npe_tot = 0
        tmp_Npe_CK_ori_Hama = 0
        tmp_Npe_CK_ori_NNVT = 0
        tmp_ck_ori_dict = {}

        for ih in range(len(simhit_PMTIDs)):
            if simhit_PMTIDs[ih] > 17611: continue
            pmt_type = m_Id_type_dict[int(simhit_PMTIDs[ih])]
            if parse_args.only_Hama and pmt_type != 'Hamamatsu': continue
            if simhit_isCerenkov[ih] == 1 and simhit_isOriginalOP[ih] == 1:
                N_tot_ck_ori += 1
            pmt_x = m_GeoSvc.get_pmt_x(int(simhit_PMTIDs[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(simhit_PMTIDs[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(simhit_PMTIDs[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=simx, evty=simy, evtz=simz)
            timecor = simhit_times[ih]-tof-t0
            simhit_PMTID_sort.append(simhit_PMTIDs[ih])
            simhit_time_cor_sort.append(timecor) 
            simhit_is_ck_ori_sort.append(1 if (simhit_isCerenkov[ih] == 1 and simhit_isOriginalOP[ih] == 1) else 0)
            tmp_Npe_tot += 1
            h_dict['h_sim_tcor'].Fill(timecor)
            if pmt_type == 'Hamamatsu': h_dict['h_sim_tcor_Hama'].Fill(timecor)
            else                      : h_dict['h_sim_tcor_NNVT'].Fill(timecor)
            if simhit_isCerenkov[ih] == 1:
                h_dict['h_sim_tcor_CK'].Fill(timecor)
                if simhit_isOriginalOP[ih] == 1:
                    h_dict['h_sim_tcor_CK_ori'].Fill(timecor)
                    if pmt_type == 'Hamamatsu': 
                        h_dict['h_sim_tcor_CK_ori_Hama'].Fill(timecor)
                        tmp_Npe_CK_ori_Hama += 1
                    else                      : 
                        h_dict['h_sim_tcor_CK_ori_NNVT'].Fill(timecor)
                        tmp_Npe_CK_ori_NNVT += 1
                    if int(simhit_PMTIDs[ih]) not in tmp_ck_ori_dict: tmp_ck_ori_dict[int(simhit_PMTIDs[ih])] = []
                    tmp_ck_ori_dict[int(simhit_PMTIDs[ih])].append(timecor)
        h_dict['h_sim_npe_CK_ori_Hama'].Fill(tmp_Npe_CK_ori_Hama)
        h_dict['h_sim_npe_CK_ori_NNVT'].Fill(tmp_Npe_CK_ori_NNVT)
        h_dict['h_sim_totnpe'].Fill(tmp_Npe_tot)
        h_dict['h_sim_Qedep'].Fill(simE)
        ########### select early hits ##################
        tmp_Nmax = int(len(simhit_time_cor_sort)*parse_args.sim_ratio)
        sort_index  = np.argsort(np.array(simhit_time_cor_sort))
        tmp_idex = 0
        tmp_n_early_ck_ori = 0
        for ii in range(sort_index.shape[0]):
            ih=sort_index[ii]    
            if simhit_time_cor_sort[ih] < parse_args.time_cut_down:continue
            tmp_idex += 1
            pmt_x = m_GeoSvc.get_pmt_x(int(simhit_PMTID_sort[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(simhit_PMTID_sort[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(simhit_PMTID_sort[ih]))
            tmp_PMT_v3 = rt.TVector3(pmt_x-simx, pmt_y-simy, pmt_z-simz)
            tmp_dangle = tmp_gen_v3.Angle(tmp_PMT_v3) 
            h_dict['h_dangle_gen_sel'].Fill(180*tmp_dangle/math.pi)
            h_dict['h_cosdangle_gen_sel'].Fill(math.cos(tmp_dangle))
            if simhit_is_ck_ori_sort[ih]==1:
                tmp_n_early_ck_ori += 1
                h_dict['h_sim_ck_ori_early_time'].Fill(simhit_time_cor_sort[ih])
                h_dict['h_dangle_gen_ck'].Fill(180*tmp_dangle/math.pi)
                h_dict['h_cosdangle_gen_ck'].Fill(math.cos(tmp_dangle))
            if tmp_idex >= tmp_Nmax:break
        if tmp_idex > 0 : h_dict['gr_sim_earlys_nck_s2b'].SetPoint(h_dict['gr_sim_earlys_nck_s2b'].GetN(),tmp_n_early_ck_ori,1.0*tmp_n_early_ck_ori/tmp_idex)
        h_dict['h_sim_n_ck_ori_early'].Fill(tmp_n_early_ck_ori)
        if tmp_idex > 0 :h_dict['h_sim_s2b_early'].Fill(1.0*tmp_n_early_ck_ori/tmp_idex)
        if N_tot_ck_ori > 0: h_dict['h_sim_CK_ori_pass_ratio'].Fill(1.0*tmp_n_early_ck_ori/N_tot_ck_ori)
    return h_dict

def root2hdf5_Points_calib0 (batch_size, tree, start_event, out_name):
    h_tcor = rt.TH1F('h_tcor','hittime_{cor} (ns)',300,200,500)
    Nmax = 0
    evt_peak_dict={}
    for ie in range(start_event, batch_size+start_event):
        tree.GetEntry(ie)
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        h_tcor.Scale(0.)
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )##TODO, still use sim vertex
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof
            h_tcor.Fill(hittime_cor)
        tmp_peak = h_tcor.GetBinCenter(h_tcor.GetMaximumBin())
        evt_peak_dict[ie]=tmp_peak
        tmp_nmax = 0
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )##TODO, still use sim vertex
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof
            if hittime_cor > (tmp_peak-parse_args.calibtime_cut_low) and hittime_cor < (tmp_peak+parse_args.calibtime_cut_high):
                tmp_nmax += 1 
        if tmp_nmax > Nmax: Nmax = tmp_nmax
    print('Nmax=',Nmax)
    nfeature = 12
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")##seems is a correction time
        FadcEvtT  = getattr(tree, "FadcEvtT")
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")

        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue
        tmp_idx = 0
        tmp_peak = evt_peak_dict[ie]
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)##TODO, still use sim vertex
            hittime_cor = hittime[ih] - tof
            if hittime_cor <= (tmp_peak-parse_args.calibtime_cut_low) or hittime_cor >= (tmp_peak+parse_args.calibtime_cut_high):continue
            df[ie-start_event,0,tmp_idx] = edep
            df[ie-start_event,1,tmp_idx] = edepX
            df[ie-start_event,2,tmp_idx] = edepY
            df[ie-start_event,3,tmp_idx] = edepZ
            df[ie-start_event,4,tmp_idx] = hittime[ih]-tmp_peak
            df[ie-start_event,5,tmp_idx] = hittime_cor-tmp_peak
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            df[ie-start_event,11,tmp_idx] = charges[ih]
            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = rect0
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)

def root2hdf5_Points_calib_v2 (batch_size, tree, start_event, out_name):
    Nmax = 0
    for ie in range(start_event, batch_size+start_event):
        tmp_N = 0
        tree.GetEntry(ie)
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            tmp_N += 1
        if tmp_N > Nmax: Nmax = tmp_N

    Nmax = int(Nmax*parse_args.calib_ratio)
    print('Nmax=',Nmax)
    nfeature = 12
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")##seems is a correction time
        FadcEvtT  = getattr(tree, "FadcEvtT")
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")

        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue

        pmtID_v1 = []
        hittime_v1 = []
        hittimes_cor = []
        charges_v1 = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue

            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
            hittime_cor = hittime[ih] - tof - FadcEvtT + m_t_shift
            hittimes_cor.append(hittime_cor) 
            pmtID_v1.append(pmtID[ih]) 
            hittime_v1.append(hittime[ih])
            charges_v1.append(charges[ih])
            
        pmtID = pmtID_v1
        hittime = hittime_v1 
        charges = charges_v1
        np_cor_time = np.array(hittimes_cor)
        sort_index  = np.argsort(np_cor_time) 

        tmp_Nmax = int(len(pmtID)*parse_args.sim_ratio)

        tmp_idx = 0
        for ii in range(sort_index.shape[0]):
            ih = sort_index[ii]
            if tmp_idx >= df.shape[2] or tmp_idx >= tmp_Nmax:break
            #if hittimes_cor[ih] < parse_args.calibtime_cut_low or hittime_cor > parse_args.calibtime_cut_high:continue
            if hittimes_cor[ih] < parse_args.calibtime_cut_low:continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            df[ie-start_event,0,tmp_idx] = QTEn
            df[ie-start_event,1,tmp_idx] = recx
            df[ie-start_event,2,tmp_idx] = recy
            df[ie-start_event,3,tmp_idx] = recz
            df[ie-start_event,4,tmp_idx] = hittime[ih]-FadcEvtT
            df[ie-start_event,5,tmp_idx] = hittimes_cor[ih]
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            df[ie-start_event,11,tmp_idx] = charges[ih]
            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = FadcEvtT
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)




def root2hdf5_Points_calib (batch_size, tree, start_event, out_name):
    Nmax = 0
    for ie in range(start_event, batch_size+start_event):
        tree.GetEntry(ie)
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        FadcEvtT  = getattr(tree, "FadcEvtT")
        tmp_nmax = 0
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
            #hittime_cor = hittime[ih] - tof - FadcEvtT
            hittime_cor = hittime[ih] - tof - FadcEvtT + m_t_shift
            if hittime_cor > parse_args.calibtime_cut_low and hittime_cor < parse_args.calibtime_cut_high:
                tmp_nmax += 1 
        if tmp_nmax > Nmax: Nmax = tmp_nmax
    print('Nmax=',Nmax)
    nfeature = 12
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")##seems is a correction time
        FadcEvtT  = getattr(tree, "FadcEvtT")
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")

        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue
        tmp_idx = 0
        for ih in range(len(pmtID)):
            if tmp_idx >= df.shape[2]:break
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
            hittime_cor = hittime[ih] - tof - FadcEvtT + m_t_shift
            if hittime_cor < parse_args.calibtime_cut_low or hittime_cor > parse_args.calibtime_cut_high:continue
            df[ie-start_event,0,tmp_idx] = QTEn
            df[ie-start_event,1,tmp_idx] = recx
            df[ie-start_event,2,tmp_idx] = recy
            df[ie-start_event,3,tmp_idx] = recz
            df[ie-start_event,4,tmp_idx] = hittime[ih]-FadcEvtT
            df[ie-start_event,5,tmp_idx] = hittime_cor
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            df[ie-start_event,11,tmp_idx] = charges[ih]
            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = FadcEvtT
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)


def pmt_times_tts(pmtID,hittime,realistic):
    tmp_times_list = []
    for ih in range(len(pmtID)):
        if pmtID[ih] > 17611:##SPMT,do nonthing
            tmp_times_list.append(hittime[ih])
        else:
            tmp_mean_tts = 1 ##ns,Hama
            pmt_type = m_Id_type_dict[int(pmtID[ih])]
            if pmt_type != 'Hamamatsu': tmp_mean_tts = 5 ##ns, NNVT
            tmp_times_list.append(hittime[ih]+np.random.normal(0,1)*tmp_mean_tts if realistic==False else pmt_effect.produce_tts(int(pmtID[ih]),hittime[ih]))
    return tmp_times_list
 

def get_weight(val):
    if val < m_x_min  or val > m_x_max: return 0.
    tmp_bin = m_h_axis.FindBin(val)
    return m_h_num.GetBinContent(tmp_bin)

def root2hdf5_Points_detsim_v3 (batch_size, tree, start_event, out_name):##add 1D hist of hittime_cor
    Nmax = 0
    for ie in range(start_event, batch_size+start_event):
        tmp_N = 0
        tree.GetEntry(ie)
        pmtID                = list( getattr(tree, "simhit_PMTIDs"      ) )
        hittime              = list( getattr(tree, "simhit_times"       ) )
        simhit_isCerenkov    = list( getattr(tree, "simhit_isCerenkov"  ) )
        simhit_isOriginalOP  = list( getattr(tree, "simhit_isOriginalOP") )
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                for it in dark_noise_dict[iD]:
                    pmtID.append(iD)
                    hittime.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            tmp_N += 1
 
        if tmp_N > Nmax: Nmax = tmp_N

    Nmax = int(Nmax*parse_args.sim_ratio)
    print('Nmax=',Nmax)
    nfeature = 11+3 
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_1D   = np.full((batch_size, int(parse_args.time_1D_high-parse_args.time_1D_low)), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    ref_vector = rt.TVector3(1,0,0)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")[0]
        initY      = getattr(tree, "gen_InitY")[0]
        initZ      = getattr(tree, "gen_InitZ")[0]
        initPX     = getattr(tree, "gen_InitPX")[0]
        initPY     = getattr(tree, "gen_InitPY")[0]
        initPZ     = getattr(tree, "gen_InitPZ")[0]
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_QedepX")
        edepY     = getattr(tree, "sim_QedepY")
        edepZ     = getattr(tree, "sim_QedepZ")
        #####################
        tmp_simE = 0
        tmp_simx = 0
        tmp_simy = 0
        tmp_simz = 0
        for i in range(len(edep)):
            if edep[i] <=0:continue
            tmp_simE += edep[i]
            #print('i=',i,',edepX=',edepX[i],',edep=',edep[i])
            tmp_simx += edepX[i]*edep[i]
            tmp_simy += edepY[i]*edep[i]
            tmp_simz += edepZ[i]*edep[i]
        edep = tmp_simE 
        edepX = tmp_simx/edep
        edepY = tmp_simy/edep
        edepZ = tmp_simz/edep
        #print('edepX=',edepX,',tmp_simx=',tmp_simx,',edep=',edep)
        ########## energy cut, in MeV ########
        if parse_args.doEcut: 
            if edep < parse_args.E_min or edep > parse_args.E_max:continue 
        #print('edepX=',edepX,',tmp_simx=',tmp_simx,',edep=',edep)
        ########## reweighting ########
        if parse_args.doReWeight: 
            we = get_weight(edep)
            if we > 1:
                sv = 1/we
                ran = random.uniform(0, 1)
                if ran > sv:continue
        ####################
        pmtID                = list( getattr(tree, "simhit_PMTIDs"       ))
        hittime              = list( getattr(tree, "simhit_times"        ))
        simhit_isCerenkov    = list( getattr(tree, "simhit_isCerenkov"   ))
        simhit_isOriginalOP  = list( getattr(tree, "simhit_isOriginalOP" ))

        if parse_args.addTTS:
            hittime = pmt_times_tts(pmtID,hittime,parse_args.TTS_realistic)
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                for it in dark_noise_dict[iD]:
                    pmtID.append(iD)
                    hittime.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)

        if parse_args.addEnergySmear:
            edep = energy_semar(edep)
        if parse_args.addVertexSmear:
            edepX, edepY, edepZ = vertex_semar(edep, edepX, edepY, edepZ)
        t0 = 0
        if parse_args.addT0Smear:
            t0 = random.gauss(0, parse_args.T0_sigma) 
        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue

        rand_costheta = np.random.uniform(-1,1,1)
        rand_phi      = np.random.uniform(0,2*math.pi,1)
        ran_initPZ = rand_costheta[0]
        ran_initPX = math.sqrt(1-ran_initPZ*ran_initPZ)*math.cos(rand_phi[0])
        ran_initPY = math.sqrt(1-ran_initPZ*ran_initPZ)*math.sin(rand_phi[0])
        ran_initP = math.sqrt(ran_initPX*ran_initPX + ran_initPY*ran_initPY + ran_initPZ*ran_initPZ)
 

        pmtID_v1 = []
        hittime_v1 = []
        hittimes_cor = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof - t0
            #print('ih=',ih,',hittime=',hittime[ih],',tof=',tof,',edepX=',edepX,',edepY=',edepY,',edepZ=',edepZ,',pmt_x=',pmt_x,',pmt_y=',pmt_y,',pmt_z=',pmt_z)
            pmtID_v1.append(int(pmtID[ih]))
            hittime_v1.append(hittime[ih])
            hittimes_cor.append(hittime_cor)
        tmp_Nmax = int(len(hittimes_cor)*parse_args.sim_ratio)

        pmtID = pmtID_v1
        hittime = hittime_v1 
        np_cor_time = np.array(hittimes_cor)
        sort_index  = np.argsort(np_cor_time) 
        tmp_idx = 0

        ########## save 1D hist ####################
        for ii in range(len(hittimes_cor)):
            #print('ii=',ii,',hittimes_cor=',hittimes_cor[ii])
            tmp_bin = int(hittimes_cor[ii]-parse_args.time_1D_low)
            if tmp_bin<0 or tmp_bin>= df_1D.shape[1]:continue
            df_1D[ie-start_event,tmp_bin] += 1
        ############ save hits ###################
        #for ii in range(sort_index.shape[0]):
        for ii in range(tmp_Nmax):
            #if tmp_idx >= df.shape[2] or tmp_idx > tmp_Nmax:break
            ih = sort_index[ii]
            if hittimes_cor[ih] < parse_args.time_cut_down:continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            if parse_args.use_dir_cut:##not use for default
                assert (parse_args.save_dir_gen or parse_args.save_dir_random)
                gen_v3 = rt.TVector3(ran_initPX,ran_initPY,ran_initPZ)
                if parse_args.save_dir_gen:
                    gen_v3 = rt.TVector3(initPX,initPY,initPZ)
                pmt_v3 = rt.TVector3(pmt_x-edepX,pmt_y-edepY,pmt_z-edepZ)
                cosdangle = math.cos( gen_v3.Angle(pmt_v3))
                if cosdangle < parse_args.cosAngle_min:continue 
            df[ie-start_event,0,tmp_idx] = edep
            df[ie-start_event,1,tmp_idx] = edepX
            df[ie-start_event,2,tmp_idx] = edepY
            df[ie-start_event,3,tmp_idx] = edepZ
            df[ie-start_event,4,tmp_idx] = hittime[ih]
            df[ie-start_event,5,tmp_idx] = hittimes_cor[ih]
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            df[ie-start_event,11,tmp_idx] = 0
            df[ie-start_event,12,tmp_idx] = 0
            df[ie-start_event,13,tmp_idx] = 0
            if parse_args.save_dir_random: 
                df[ie-start_event,11,tmp_idx] = ran_initPX/ran_initP
                df[ie-start_event,12,tmp_idx] = ran_initPY/ran_initP
                df[ie-start_event,13,tmp_idx] = ran_initPZ/ran_initP
            elif parse_args.save_dir_gen: 
                if parse_args.smear_gen: 
                    gen_v3     = rt.TVector3(initPX,initPY,initPZ)
                    gen_v3_bak = rt.TVector3(initPX,initPY,initPZ)
                    normal_vector = gen_v3.Cross(ref_vector)
                    smear_costh = m_h_smear.GetRandom() 
                    smear_angle = math.acos(smear_costh)
                    rand_angle  = np.random.uniform(0,2*math.pi,1)[0]
                    gen_v3.Rotate(smear_angle, normal_vector); # rotation around normal_vector
                    gen_v3.Rotate(rand_angle , gen_v3_bak   ); # rotation around gen_v3_bak
                    df[ie-start_event,11,tmp_idx] = gen_v3.X()/gen_v3.Mag() 
                    df[ie-start_event,12,tmp_idx] = gen_v3.Y()/gen_v3.Mag()
                    df[ie-start_event,13,tmp_idx] = gen_v3.Z()/gen_v3.Mag()
                else: 
                    df[ie-start_event,11,tmp_idx] = initPX/tmp_initP
                    df[ie-start_event,12,tmp_idx] = initPY/tmp_initP
                    df[ie-start_event,13,tmp_idx] = initPZ/tmp_initP

            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10]= initPZ/tmp_initP
        df_true[ie-start_event, 11] = 0
        df_true[ie-start_event, 12] = 0
        df_true[ie-start_event, 13] = 0
        df_true[ie-start_event, 14] = 0 
        df_true[ie-start_event, 15] = t0
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_1D   = np.delete(df_1D  , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('data_1D', data=df_1D)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', data 1D shape=',df_1D.shape,', label=', df_true.shape)




def root2hdf5_Points_detsim_v2 (batch_size, tree, start_event, out_name):
    Nmax = 0
    for ie in range(start_event, batch_size+start_event):
        tmp_N = 0
        tree.GetEntry(ie)
        pmtID                = list( getattr(tree, "simhit_PMTIDs"      ) )
        hittime              = list( getattr(tree, "simhit_times"       ) )
        simhit_isCerenkov    = list( getattr(tree, "simhit_isCerenkov"  ) )
        simhit_isOriginalOP  = list( getattr(tree, "simhit_isOriginalOP") )
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                for it in dark_noise_dict[iD]:
                    pmtID.append(iD)
                    hittime.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            tmp_N += 1
 
        if tmp_N > Nmax: Nmax = tmp_N

    Nmax = int(Nmax*parse_args.sim_ratio)
    print('Nmax=',Nmax)
    nfeature = 11+3 
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")[0]
        initY      = getattr(tree, "gen_InitY")[0]
        initZ      = getattr(tree, "gen_InitZ")[0]
        initPX     = getattr(tree, "gen_InitPX")[0]
        initPY     = getattr(tree, "gen_InitPY")[0]
        initPZ     = getattr(tree, "gen_InitPZ")[0]
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_QedepX")
        edepY     = getattr(tree, "sim_QedepY")
        edepZ     = getattr(tree, "sim_QedepZ")
        #####################
        tmp_simE = 0
        tmp_simx = 0
        tmp_simy = 0
        tmp_simz = 0
        for i in range(len(edep)):
            tmp_simE += edep[i]
            tmp_simx += edepX[i]*edep[i]
            tmp_simy += edepY[i]*edep[i]
            tmp_simz += edepZ[i]*edep[i]
        edep = tmp_simE 
        edepX = tmp_simx/edep
        edepY = tmp_simy/edep
        edepZ = tmp_simz/edep
        ########## energy cut, in MeV ########
        if parse_args.doEcut: 
            if edep < parse_args.E_min or edep > parse_args.E_max:continue 
        ########## reweighting ########
        if parse_args.doReWeight: 
            we = get_weight(edep)
            if we > 1:
                sv = 1/we
                ran = random.uniform(0, 1)
                if ran > sv:continue
        ####################
        pmtID                = list( getattr(tree, "simhit_PMTIDs"       ))
        hittime              = list( getattr(tree, "simhit_times"        ))
        simhit_isCerenkov    = list( getattr(tree, "simhit_isCerenkov"   ))
        simhit_isOriginalOP  = list( getattr(tree, "simhit_isOriginalOP" ))

        if parse_args.addTTS:
            hittime = pmt_times_tts(pmtID,hittime,parse_args.TTS_realistic)
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                for it in dark_noise_dict[iD]:
                    pmtID.append(iD)
                    hittime.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)

        if parse_args.addEnergySmear:
            edep = energy_semar(edep)
        if parse_args.addVertexSmear:
            edepX, edepY, edepZ = vertex_semar(edep, edepX, edepY, edepZ)
        t0 = 0
        if parse_args.addT0Smear:
            t0 = random.gauss(0, parse_args.T0_sigma) 
        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue

        rand_costheta = np.random.uniform(-1,1,1)
        rand_phi      = np.random.uniform(0,2*math.pi,1)
        ran_initPZ = rand_costheta[0]
        ran_initPX = math.sqrt(1-ran_initPZ*ran_initPZ)*math.cos(rand_phi[0])
        ran_initPY = math.sqrt(1-ran_initPZ*ran_initPZ)*math.sin(rand_phi[0])
        ran_initP = math.sqrt(ran_initPX*ran_initPX + ran_initPY*ran_initPY + ran_initPZ*ran_initPZ)
 

        pmtID_v1 = []
        hittime_v1 = []
        hittimes_cor = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof - t0
            pmtID_v1.append(int(pmtID[ih]))
            hittime_v1.append(hittime[ih])
            hittimes_cor.append(hittime_cor)
        tmp_Nmax = int(len(hittimes_cor)*parse_args.sim_ratio)

        pmtID = pmtID_v1
        hittime = hittime_v1 
        np_cor_time = np.array(hittimes_cor)
        sort_index  = np.argsort(np_cor_time) 
        tmp_idx = 0
        #for ii in range(sort_index.shape[0]):
        for ii in range(tmp_Nmax):
            #if tmp_idx >= df.shape[2] or tmp_idx > tmp_Nmax:break
            ih = sort_index[ii]
            if hittimes_cor[ih] < parse_args.time_cut_down:continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            if parse_args.use_dir_cut:
                assert (parse_args.save_dir_gen or parse_args.save_dir_random)
                gen_v3 = rt.TVector3(ran_initPX,ran_initPY,ran_initPZ)
                if parse_args.save_dir_gen:
                    gen_v3 = rt.TVector3(initPX,initPY,initPZ)
                pmt_v3 = rt.TVector3(pmt_x-edepX,pmt_y-edepY,pmt_z-edepZ)
                cosdangle = math.cos( gen_v3.Angle(pmt_v3))
                if cosdangle < parse_args.cosAngle_min:continue 
            df[ie-start_event,0,tmp_idx] = edep
            df[ie-start_event,1,tmp_idx] = edepX
            df[ie-start_event,2,tmp_idx] = edepY
            df[ie-start_event,3,tmp_idx] = edepZ
            df[ie-start_event,4,tmp_idx] = hittime[ih]
            df[ie-start_event,5,tmp_idx] = hittimes_cor[ih]
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            df[ie-start_event,11,tmp_idx] = 0
            df[ie-start_event,12,tmp_idx] = 0
            df[ie-start_event,13,tmp_idx] = 0
            if parse_args.save_dir_random: 
                df[ie-start_event,11,tmp_idx] = ran_initPX/ran_initP
                df[ie-start_event,12,tmp_idx] = ran_initPY/ran_initP
                df[ie-start_event,13,tmp_idx] = ran_initPZ/ran_initP
            elif parse_args.save_dir_gen: 
                df[ie-start_event,11,tmp_idx] = initPX/tmp_initP
                df[ie-start_event,12,tmp_idx] = initPY/tmp_initP
                df[ie-start_event,13,tmp_idx] = initPZ/tmp_initP

            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10]= initPZ/tmp_initP
        df_true[ie-start_event, 11] = 0
        df_true[ie-start_event, 12] = 0
        df_true[ie-start_event, 13] = 0
        df_true[ie-start_event, 14] = 0 
        df_true[ie-start_event, 15] = t0
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)



def root2hdf5_Points_detsim (batch_size, tree, start_event, out_name):

    Nmax = 0
    for ie in range(start_event, batch_size+start_event):
        tree.GetEntry(ie)
        pmtID     = getattr(tree, "simhit_PMTIDs")
        hittime   = getattr(tree, "simhit_times")
        simhit_isCerenkov = getattr(tree, "simhit_isCerenkov")
        simhit_isOriginalOP = getattr(tree, "simhit_isOriginalOP")
        #edepX     = getattr(tree, "sim_edepX")
        #edepY     = getattr(tree, "sim_edepY")
        #edepZ     = getattr(tree, "sim_edepZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_QedepX")
        edepY     = getattr(tree, "sim_QedepY")
        edepZ     = getattr(tree, "sim_QedepZ")
        ############################## 
        if parse_args.addTTS:
            hittime = pmt_times_tts(pmtID,hittime,parse_args.TTS_realistic)
        if parse_args.addEnergySmear:
            edep = energy_semar(edep)
        if parse_args.addVertexSmear:
            edepX, edepY, edepZ = vertex_semar(edep, edepX, edepY, edepZ)
        t0 = 0
        if parse_args.addT0Smear:
            t0 = random.gauss(0, parse_args.T0_sigma) 
        ############################## 
        tmp_nmax = 0
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            #dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof - t0
            #if hittime_cor > parse_args.time_cut:continue
            if hittime_cor > parse_args.time_cut or hittime_cor < parse_args.time_cut_down:continue
            tmp_nmax += 1 
        if tmp_nmax > Nmax: Nmax = tmp_nmax
    if parse_args.addTTS: Nmax += 100 ##just of safe
    print('Nmax=',Nmax)
    nfeature = 11 
    df      = np.full((batch_size, nfeature, Nmax), 0, np.float32)
    df_true = np.full((batch_size, 16), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        #edepX     = getattr(tree, "sim_edepX")
        #edepY     = getattr(tree, "sim_edepY")
        #edepZ     = getattr(tree, "sim_edepZ")
        edepX     = getattr(tree, "sim_QedepX")
        edepY     = getattr(tree, "sim_QedepY")
        edepZ     = getattr(tree, "sim_QedepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")

        pmtID     = getattr(tree, "simhit_PMTIDs")
        hittime   = getattr(tree, "simhit_times")
        simhit_isCerenkov = getattr(tree, "simhit_isCerenkov")
        simhit_isOriginalOP = getattr(tree, "simhit_isOriginalOP")
        pmtID = list(pmtID)
        hittime = list(hittime)
        simhit_isCerenkov = list(simhit_isCerenkov)
        simhit_isOriginalOP = list(simhit_isOriginalOP)

        if parse_args.addTTS:
            hittime = pmt_times_tts(pmtID,hittime,parse_args.TTS_realistic)
        if parse_args.addDN:
            dark_noise_dict = pmt_effect.produce_dark_noise()
            for iD in dark_noise_dict:
                for it in dark_noise_dict[iD]:
                    pmtID.append(iD)
                    hittime.append(it)
                    simhit_isCerenkov.append(0)
                    simhit_isOriginalOP.append(0)

        if parse_args.addEnergySmear:
            edep = energy_semar(edep)
        if parse_args.addVertexSmear:
            edepX, edepY, edepZ = vertex_semar(edep, edepX, edepY, edepZ)
        t0 = 0
        if parse_args.addT0Smear:
            t0 = random.gauss(0, parse_args.T0_sigma) 
 
        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue
        tmp_idx = 0
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            #dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof - t0
            #if hittime_cor > parse_args.time_cut:continue
            if hittime_cor > parse_args.time_cut or hittime_cor < parse_args.time_cut_down:continue
            df[ie-start_event,0,tmp_idx] = edep
            df[ie-start_event,1,tmp_idx] = edepX
            df[ie-start_event,2,tmp_idx] = edepY
            df[ie-start_event,3,tmp_idx] = edepZ
            df[ie-start_event,4,tmp_idx] = hittime[ih]
            df[ie-start_event,5,tmp_idx] = hittime_cor
            df[ie-start_event,6,tmp_idx] = pmt_x
            df[ie-start_event,7,tmp_idx] = pmt_y
            df[ie-start_event,8,tmp_idx] = pmt_z
            if m_Id_type_dict[int(pmtID[ih])] == 'Hamamatsu': 
                df[ie-start_event,9,tmp_idx] = 1
                df[ie-start_event,10,tmp_idx] = 0
            else:
                df[ie-start_event,9,tmp_idx] = 0
                df[ie-start_event,10,tmp_idx] = 1
            tmp_idx += 1

        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = rect0
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data shape=',df.shape,', label=', df_true.shape)





def root2hdf5_2D_detsim (batch_size, tree, start_event, out_name, id_dict, x_max, y_max, ID_x_y_z_dict, Draw_data, Draw_CK):


    df      = np.full((batch_size, y_max+1, x_max+1, 2), 0, np.float32)
    df_true = np.full((batch_size, 8+9), 0, np.float32)
    refractive = 1.54
    velocity = 0.3*1000/refractive # mm/ns
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")
        pmtID     = getattr(tree, "simhit_PMTIDs")
        hittime   = getattr(tree, "simhit_times")
        #charges   = getattr(tree, "calib_charges")
        simhit_isCerenkov = getattr(tree, "simhit_isCerenkov")
        simhit_isOriginalOP = getattr(tree, "simhit_isOriginalOP")

        pmtID_new = []
        hittime_new = []
        charges_new = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            if parse_args.Save_only_ck:
                if simhit_isCerenkov[ih] == False  or simhit_isOriginalOP[ih] == False: continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            dist = math.sqrt(math.pow(pmt_x-edepX,2)+math.pow(pmt_y-edepY,2)+math.pow(pmt_z-edepZ,2) )
            #tof = dist/m_velocity ##ns
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=edepX, evty=edepY, evtz=edepZ)
            hittime_cor = hittime[ih] - tof
            if hittime_cor > parse_args.time_cut:continue
            pmtID_new.append(pmtID[ih])
            hittime_new.append(hittime_cor)
            charges_new.append(1)
        pmtID = pmtID_new
        hittime = hittime_new 
        charges = charges_new
        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue
        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = rect0
        #################
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
        ############################
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
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data2D', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data 2D shape=',df.shape,', label=', df_true.shape)

    if Draw_data:
        N_plots = 10
        for i in range(df.shape[0]):
            if N_plots < 0: break
            draw_data(outname='npe:E%.1f_x%d_y%d_z%d_px%.2f_py%.2f_pz%.2f'%(df_true[i,0], df_true[i,1], df_true[i,2], df_true[i,3], df_true[i,8], df_true[i,9], df_true[i,10] )  , x_max=x_max, y_max=y_max, df=df[i,:,:,0]  , df_ck = None, draw_ck = False )
            draw_data(outname='Ftime:E%.1f_x%d_y%d_z%d_px%.2f_py%.2f_pz%.2f'%(df_true[i,0], df_true[i,1], df_true[i,2], df_true[i,3], df_true[i,8], df_true[i,9], df_true[i,10] )  , x_max=x_max, y_max=y_max, df=df[i,:,:,1], df_ck = None, draw_ck = False )
            N_plots -= 1




def root2hdf5_2D (batch_size, tree, start_event, out_name, id_dict, x_max, y_max, ID_x_y_z_dict, Draw_data, Draw_CK):


    df      = np.full((batch_size, y_max+1, x_max+1, 2), 0, np.float32)
    df_true = np.full((batch_size, 8+9), 0, np.float32)
    refractive = 1.54
    velocity = 0.3*1000/refractive # mm/ns
    df_ck = np.full((batch_size, 100, 2), 0, np.float32)
    for ie in range(start_event, batch_size+start_event):
        print('for evt=',ie)
        tree.GetEntry(ie)
        tmp_dict = {}
        tmp_firstHitTime_dict = {}

        initX      = getattr(tree, "gen_InitX")
        initY      = getattr(tree, "gen_InitY")
        initZ      = getattr(tree, "gen_InitZ")
        initPX     = getattr(tree, "gen_InitPX")
        initPY     = getattr(tree, "gen_InitPY")
        initPZ     = getattr(tree, "gen_InitPZ")
        edep      = getattr(tree, "sim_Qedep")
        edepX     = getattr(tree, "sim_edepX")
        edepY     = getattr(tree, "sim_edepY")
        edepZ     = getattr(tree, "sim_edepZ")
        QTEn      = getattr(tree, "QTEn")
        QTL       = getattr(tree, "QTL")
        recx      = getattr(tree, "recQTx")
        recy      = getattr(tree, "recQTy")
        recz      = getattr(tree, "recQTz")
        rect0     = getattr(tree, "QTt0")
        pmtID     = getattr(tree, "calib_PMTIDs")
        hittime   = getattr(tree, "calib_times")
        charges   = getattr(tree, "calib_charges")
        if Draw_CK:
            simhit_PMTIDs = getattr(tree, "simhit_PMTIDs")
            simhit_isCerenkov = getattr(tree, "simhit_isCerenkov")
            simhit_isOriginalOP = getattr(tree, "simhit_isOriginalOP")
            tmp_i = 0
            for ih in range(len(simhit_PMTIDs)):
                if simhit_PMTIDs[ih] > 17611: continue
                if int(simhit_PMTIDs[ih]) not in id_dict: continue
                ix = id_dict[int(simhit_PMTIDs[ih])][0]
                iy = id_dict[int(simhit_PMTIDs[ih])][1]
                if simhit_isCerenkov[ih] and simhit_isOriginalOP[ih]:
                    df_ck[ie-start_event,tmp_i,0] = ix 
                    df_ck[ie-start_event,tmp_i,1] = iy
                    tmp_i += 1

        pmtID_new = []
        hittime_new = []
        charges_new = []
        for ih in range(len(pmtID)):
            if pmtID  [ih] > 17611: continue
            if parse_args.only_Hama and m_Id_type_dict[int(pmtID[ih])] != 'Hamamatsu': continue
            pmt_x = m_GeoSvc.get_pmt_x(int(pmtID[ih]))
            pmt_y = m_GeoSvc.get_pmt_y(int(pmtID[ih]))
            pmt_z = m_GeoSvc.get_pmt_z(int(pmtID[ih]))
            dist = math.sqrt(math.pow(pmt_x-recx,2)+math.pow(pmt_y-recy,2)+math.pow(pmt_z-recz,2) )
            #tof = dist/m_velocity ##ns
            tof = OMILREC_CalLTOF(pmt_pos_x=pmt_x, pmt_pos_y=pmt_y, pmt_pos_z=pmt_z, evtx=recx, evty=recy, evtz=recz)
            trigger_shift = 300 #ns
            hittime_cor = hittime[ih] - tof - trigger_shift 
            if hittime_cor < -100:continue
            pmtID_new.append(pmtID[ih])
            hittime_new.append(hittime_cor)
            charges_new.append(charges[ih])
        pmtID = pmtID_new
        hittime = hittime_new 
        charges = charges_new
        tmp_initP = math.sqrt(initPX*initPX + initPY*initPY + initPZ*initPZ)
        if tmp_initP <=0:continue
        df_true[ie-start_event, 0] = edep
        df_true[ie-start_event, 1] = edepX
        df_true[ie-start_event, 2] = edepY
        df_true[ie-start_event, 3] = edepZ
        df_true[ie-start_event, 4] = initX
        df_true[ie-start_event, 5] = initY
        df_true[ie-start_event, 6] = initZ
        df_true[ie-start_event, 7] = tmp_initP
        df_true[ie-start_event, 8] = initPX/tmp_initP
        df_true[ie-start_event, 9] = initPY/tmp_initP
        df_true[ie-start_event, 10] = initPZ/tmp_initP
        df_true[ie-start_event, 11] = QTEn
        df_true[ie-start_event, 12] = recx
        df_true[ie-start_event, 13] = recy
        df_true[ie-start_event, 14] = recz
        df_true[ie-start_event, 15] = rect0
        #################
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
        ############################
        for iD in id_dict:
            ix     = id_dict[iD][0]
            iy     = id_dict[iD][1]
            tmp_npe = 0
            tmp_firstHitTime = 0
            if iD in tmp_dict:
                tmp_npe          = tmp_dict[iD]
                tmp_firstHitTime = tmp_firstHitTime_dict[iD]
            df[ie-start_event,iy,ix,0] = tmp_npe
            #df[ie-start_event,iy,ix,1] = tmp_firstHitTime##FIXME, seems some hit time will < 0
            #df[ie-start_event,iy,ix,1] = tmp_firstHitTime if tmp_firstHitTime>=0 else 0 ##
            df[ie-start_event,iy,ix,1] = tmp_firstHitTime
        ###########################
    if True:
        tmp_index = []
        for i in range(df.shape[0]):
            if df_true[i,0]==0: ##edep
                tmp_index.append(i)
        df      = np.delete(df     , tmp_index, 0)
        df_true = np.delete(df_true, tmp_index, 0)
        if Draw_CK:
            df_ck   = np.delete(df_ck  , tmp_index, 0)


    hf = h5py.File(out_name, 'w')
    hf.create_dataset('data2D', data=df)
    hf.create_dataset('label', data=df_true)
    hf.close()
    print('saved %s'%out_name, 'with data 2D shape=',df.shape,', label=', df_true.shape)

    if Draw_data:
        N_plots = 10
        for i in range(df.shape[0]):
            if N_plots < 0: break
            draw_data(outname='npe:E%.1f_x%d_y%d_z%d_px%.2f_py%.2f_pz%.2f'%(df_true[i,0], df_true[i,1], df_true[i,2], df_true[i,3], df_true[i,8], df_true[i,9], df_true[i,10] )  , x_max=x_max, y_max=y_max, df=df[i,:,:,0]  , df_ck = df_ck[i,:,:] if Draw_CK else None, draw_ck = Draw_CK )
            draw_data(outname='Ftime:E%.1f_x%d_y%d_z%d_px%.2f_py%.2f_pz%.2f'%(df_true[i,0], df_true[i,1], df_true[i,2], df_true[i,3], df_true[i,8], df_true[i,9], df_true[i,10] )  , x_max=x_max, y_max=y_max, df=df[i,:,:,1], df_ck = df_ck[i,:,:] if Draw_CK else None, draw_ck = Draw_CK )
            N_plots -= 1


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
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()

def do_plot_gr(hist,out_name,title):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    hist.SetMarkerColor(2)
    hist.SetMarkerStyle(8)
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.Draw("AP")
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
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
    do_plot2d(hist=h_map, out_name='id_map', title={'X':'x(phi)','Y':'y(z)'})
    return (x_min, x_max, y_min, y_max)

def draw_data(outname, x_max, y_max, df, df_ck, draw_ck):
    print('start drawing, outname=',outname)
    h_map = rt.TH2F('map_%s'%outname,'',x_max+2, -1, x_max+1, y_max+2, -1, y_max+1)
    #if 'Ftime' in outname:
    #    for ix in range(1,h_map.GetNbinsX()+1):
    #        for iy in range(1,h_map.GetNbinsY()+1):
    #            h_map.SetBinContent(ix, iy, -99)##FIXME,for plot showing 
    for iy in range(df.shape[0]):
        for ix in range(df.shape[1]):
            if df[iy][ix]==0:continue
            h_map.SetBinContent(ix+1, iy+1, df[iy][ix]) 
            print('ix=',ix,',iy=',iy,',val=',df[iy][ix])
    if draw_ck:
        for i in range(df_ck.shape[0]):
            ix = int(df_ck[i,0])
            iy = int(df_ck[i,1])
            print('ix=',ix,',iy=',iy)
            if ix==0 and iy==0:break
            #h_map.SetBinContent(ix+1, iy+1, 99)##too ye
            h_map.SetBinContent(ix+1, iy+1, -99)##too ye
    do_plot2d(hist=h_map, out_name='%s/map_%s'%(plots_path, outname), title={'X':'x(phi)','Y':'y(z)'})

class GeoSvc(object):
    def __init__(self, file_pos, i_id, i_x, i_y, i_z, i_theta, i_phi ):
        f = open(file_pos,'r')
        lines = f.readlines()
        self.id_map = {}
        for line in lines:
            items = line.split()
            ID    = float(items[i_id])
            ID    = int(ID)
            x     = float(items[i_x])
            y     = float(items[i_y])
            z     = float(items[i_z])
            phi   = float(items[i_phi])
            theta = float(items[i_theta])
            self.id_map[ID]=[x,y,z,theta,phi] 
        f.close()
    def get_pmt_x(self, ID):
        #if ID < 0: print('ID=',ID)
        return self.id_map[ID][0]
    def get_pmt_y(self, ID):
        return self.id_map[ID][1]
    def get_pmt_z(self, ID):
        return self.id_map[ID][2]
    def get_pmt_theta(self, ID):
        return self.id_map[ID][3]
    def get_pmt_phi(self, ID):
        return self.id_map[ID][4]

def plot_grs(hs_dict, out_name, title, rangeX, rangeY, DrawGrid=False):#hs_dict={'leg_name':[hist,color,drawoption,MarkerStyle]}
    for i in hs_dict:
        hs_dict[i][0].SetLineWidth(2)
        hs_dict[i][0].SetLineColor(hs_dict[i][1])
        hs_dict[i][0].SetMarkerColor(hs_dict[i][1])
        hs_dict[i][0].SetMarkerStyle(hs_dict[i][3])
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    if DrawGrid: 
        canvas.SetGridx()
        canvas.SetGridy()
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    y_min = rangeY[0]
    y_max = rangeY[1]
    dummy = rt.TH2D("dummy","",1, x_min, x_max, 1, y_min, y_max)
    dummy.SetStats(rt.kFALSE)
    dummy.GetYaxis().SetTitle(title['Y'])
    dummy.GetXaxis().SetTitle(title['X'])
    dummy.GetYaxis().SetTitleSize(0.04)
    dummy.GetXaxis().SetTitleSize(0.04)
    dummy.GetYaxis().SetLabelSize(0.04)
    dummy.GetXaxis().SetLabelSize(0.04)
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("AXIS")
    for i in hs_dict:
        hs_dict[i][0].Draw("%s"%(hs_dict[i][2]))
    #dummy.Draw("AXISSAME")
    x_l = 0.2
    y_h = 0.88
    y_dh = 0.2
    x_dl = 0.6
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.SetNColumns(3)
    for i in hs_dict:
        legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'p')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()




def plot_hists(hs_dict, out_name, title, rangeX, rangeY, doNorm=False, DrawGrid=False, doNormOne=False, sep_list=[]):#hs_dict={'leg_name':[hist,color,drawoption,MarkerStyle]}
    max_y = 0
    min_y = 1
    min_x = 0
    max_x = 0
    N1 = 0
    if doNormOne:
        for i in hs_dict:
            hs_dict[i][0].Scale(1.0/hs_dict[i][0].GetSumOfWeights())
    for i in hs_dict:
        N1 = hs_dict[i][0].GetSumOfWeights()
        min_x = hs_dict[i][0].GetXaxis().GetXmin()
        max_x = hs_dict[i][0].GetXaxis().GetXmax()
        tmp_max_y = hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMaximumBin())
        if tmp_max_y > max_y : max_y = tmp_max_y 
    if doNorm:
        max_y = 0
        for i in hs_dict:
            hs_dict[i][0].Scale(1.0*N1/hs_dict[i][0].GetSumOfWeights() if hs_dict[i][0].GetSumOfWeights()>0 else 1 )
            tmp_max_y = hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMaximumBin())
            tmp_min_y = hs_dict[i][0].GetBinContent(hs_dict[i][0].GetMinimumBin())
            if tmp_max_y > max_y : max_y = tmp_max_y 
            if tmp_min_y < min_y : min_y = tmp_min_y 

    for i in hs_dict:
        hs_dict[i][0].SetLineWidth(2)
        hs_dict[i][0].SetLineColor(hs_dict[i][1])
        hs_dict[i][0].SetMarkerColor(hs_dict[i][1])
        hs_dict[i][0].SetMarkerStyle(hs_dict[i][3])
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    if DrawGrid: 
        canvas.SetGridx()
        canvas.SetGridy()
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    x_min = rangeX[0]
    x_max = rangeX[1]
    if rangeX[1]==rangeX[0]:
        x_min = min_x 
        x_max = max_x 
    y_min = rangeY[0]
    y_max = rangeY[1] if rangeY[1] > 0 else 1.8*max_y
    if 'logy' in out_name: 
        canvas.SetLogy()
        y_min = min_y if min_y>0 else 1
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
    dummy.GetYaxis().SetTitleOffset(1.5)
    dummy.GetXaxis().SetTitleOffset(1.2)
    dummy.GetXaxis().SetMoreLogLabels()
    dummy.GetXaxis().SetTitleFont(42)
    dummy.GetXaxis().SetLabelFont(42)
    dummy.GetXaxis().SetNdivisions(405)
    dummy.GetYaxis().SetTitleFont(42)
    dummy.GetYaxis().SetLabelFont(42)
    dummy.Draw("AXIS")
    sep_lines = {}
    for i in hs_dict:
        hs_dict[i][0].Draw("%s"%(hs_dict[i][2]))

        tmp_sumw2 = hs_dict[i][0].GetSumOfWeights()
        if tmp_sumw2 <=0:continue
        for sep in sep_list:
            for xbin in range(1, hs_dict[i][0].GetNbinsX()+1):
                if hs_dict[i][0].Integral(1,xbin)/tmp_sumw2 < sep and hs_dict[i][0].Integral(1,xbin+1)/tmp_sumw2 > sep:
                    sep_lines['%s_%s'%(i,str(sep))]=[hs_dict[i][0].GetXaxis().GetBinCenter(xbin+1),hs_dict[i][1]]
                    break
    tlines = []
    for i in sep_lines:
        tmp_line = rt.TLine(sep_lines[i][0],0,sep_lines[i][0],1)
        tmp_line.SetLineColor(sep_lines[i][1])
        tmp_line.SetLineWidth(2)
        tmp_line.Draw('same')
        tlines.append(tmp_line)



    dummy.Draw("AXISSAME")
    x_l = 0.2
    y_h = 0.88
    y_dh = 0.2
    x_dl = 0.6
    legend = rt.TLegend(x_l,y_h-y_dh,x_l+x_dl,y_h)
    legend.SetNColumns(3)
    for i in hs_dict:
        legend.AddEntry(hs_dict[i][0] ,'%s'%i  ,'ple')
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)
    legend.Draw()
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()


def OMILREC_CalLTOF(pmt_pos_x, pmt_pos_y, pmt_pos_z, evtx, evty, evtz):

    EvtR = math.sqrt(evtx*evtx + evty*evty + evtz*evtz)
    Rsp  = math.sqrt( (pmt_pos_x-evtx)*(pmt_pos_x-evtx) + (pmt_pos_y-evty)*(pmt_pos_y-evty) + (pmt_pos_z-evtz)*(pmt_pos_z-evtz))
    PMT_R = 19.434
    LS_R = 17.7 + 0.12
    c = 2.99792458e8
    CosThetaVC = (Rsp*Rsp + PMT_R*PMT_R*1.e6 - EvtR*EvtR)/(Rsp*PMT_R*1.e3*2.)
    LengthWater = 1.e3*PMT_R*CosThetaVC - 1.e3*math.sqrt(PMT_R*CosThetaVC*PMT_R*CosThetaVC - PMT_R*PMT_R + LS_R*LS_R)
    return RfrIndxLS*(Rsp - LengthWater)*1.e6/c + RfrIndxWR*LengthWater*1.e6/c


class PMT_effect:
    def __init__(self, name, para_file, time0, time1, addTimeOffSet=False):
        self.name = name
        self.paras = {} 
        self.time0 = time0
        self.time1 = time1
        self.addTimeOffSet = addTimeOffSet
        treeName='data'
        chain =rt.TChain(treeName)
        chain.Add(para_file)
        totalEntries=chain.GetEntries()
        h_tts_Hama  = rt.TH1F('h_tts_Hama','tts (ns)',100,0,10)
        h_tts_NNVT  = rt.TH1F('h_tts_NNVT','tts (ns)',100,0,10)
        h_toff_Hama  = rt.TH1F('h_toff_Hama','time offset (ns)',110,-10,100)
        h_toff_NNVT  = rt.TH1F('h_toff_NNVT','time offset (ns)',110,-10,100)
        for entryNum in range(totalEntries):
            chain.GetEntry(int(entryNum))
            tmp_pmtID      = getattr(chain, "pmtID")
            tmp_DCR        = getattr(chain, "DCR")
            tmp_TTS_SS     = getattr(chain, "TTS_SS")
            tmp_timeOffset = getattr(chain, "timeOffset")
            if tmp_pmtID not in self.paras:
                mean_DN = (self.time1-self.time0)*1e-9*tmp_DCR*1e3 ## s * Hz
                self.paras[tmp_pmtID]=[mean_DN,tmp_TTS_SS,tmp_timeOffset]
       
            pmt_type = m_Id_type_dict[tmp_pmtID]
            if pmt_type == 'Hamamatsu': 
                h_tts_Hama.Fill(tmp_TTS_SS)
                h_toff_Hama.Fill(tmp_timeOffset)
            else                      : 
                h_tts_NNVT.Fill(tmp_TTS_SS)
                h_toff_NNVT.Fill(tmp_timeOffset)
        gr_dict={}
        gr_dict['Hamamatsu']=[h_tts_Hama,2,'same:pel',20]
        gr_dict['NNVT'     ]=[h_tts_NNVT,4,'same:pel',20]
        plot_hists(hs_dict=gr_dict, out_name='TTS', title={'X':h_tts_Hama.GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True, DrawGrid=True)
        gr_dict={}
        gr_dict['Hamamatsu']=[h_toff_Hama,2,'same:pel',20]
        gr_dict['NNVT'     ]=[h_toff_NNVT,4,'same:pel',20]
        plot_hists(hs_dict=gr_dict, out_name='TimeOffset', title={'X':h_toff_Hama.GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True, DrawGrid=True)

    def produce_dark_noise(self):
        dark_noise_dict={}
        for ID in self.paras:
            N_dark_noise = np.random.poisson(self.paras[ID][0])
            if N_dark_noise <=0:
                pass
            else:
                N_times = np.random.uniform(self.time0, self.time1, N_dark_noise)
                dark_noise_dict[ID] = N_times
                #print('N_times=',N_times)
        return dark_noise_dict

    def produce_tts(self, pmt_id, time):
        if pmt_id not in self.paras:
            print('not find pmt id =',pmt_id)
            return time
        if self.addTimeOffSet==False:
            new_time = time + np.random.normal(0,1)*self.paras[pmt_id][1]
            return new_time
        else:
            new_time = time + np.random.normal(0,1)*self.paras[pmt_id][1] + self.paras[pmt_id][2]##this use timeOffset
            return new_time

    def get_timeOffset(self, pmt_id):
        return self.paras[pmt_id][2]

def vertex_semar(E,vx,vy,vz):
    assert E > 0
    #sigma = 100/math.sqrt(E)##mm/sqrt(MeV)
    sigma = 90/math.sqrt(E)##mm/sqrt(MeV)
    new_vx = vx + np.random.normal(0,1)*sigma
    new_vy = vy + np.random.normal(0,1)*sigma
    new_vz = vz + np.random.normal(0,1)*sigma
    return new_vx,new_vy,new_vz


def test_vertex_semar(E,evt=10000):
    assert E > 0
    sigma = 100/math.sqrt(E)##mm/sqrt(MeV)
    h1_r = rt.TH1F('E%.1f_rand_R'%E,'R (mm)',300,0,300)
    for i in range(evt):
        x = np.random.normal(0,1)*sigma
        y = np.random.normal(0,1)*sigma
        z = np.random.normal(0,1)*sigma
        r = math.sqrt(x*x+y*y+z*z) 
        h1_r.Fill(r)
    gr_dict={}
    gr_dict['r']=[h1_r,2,'same:pel',20]
    plot_hists(hs_dict=gr_dict, out_name='%s'%h1_r.GetName(), title={'X':h1_r.GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=False, DrawGrid=False, doNormOne=True, sep_list=[0.683])
 
def test_vertex_semar_v1(E0,E1,evt=10000):
    assert E1 > E0 and E0 >= 0
    h1_r = rt.TH1F('E0_%.1f_E1_%.1f_rand_R'%(E0,E1),'dR (mm)',50,0,500)
    s = np.random.uniform(E0,E1,evt)
    for i in range(s.shape[0]):
        E = s[i]
        sigma = 90/math.sqrt(E)##mm/sqrt(MeV)
        #sigma = 100/math.sqrt(E)##mm/sqrt(MeV)
        x = np.random.normal(0,1)*sigma
        y = np.random.normal(0,1)*sigma
        z = np.random.normal(0,1)*sigma
        r = math.sqrt(x*x+y*y+z*z) 
        h1_r.Fill(r)
    gr_dict={}
    gr_dict['r']=[h1_r,2,'same:pel',20]
    plot_hists(hs_dict=gr_dict, out_name='%s'%h1_r.GetName(), title={'X':h1_r.GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=False, DrawGrid=False, doNormOne=True, sep_list=[0.683])
    return h1_r
 

def energy_semar(E):
    new_E = E + np.random.normal(0,1)*0.03/math.sqrt(E)
    return new_E


def calculate_ck_angle(KE,n):#in MeV
    ## 0.5*m*v^{2} = KE, for electron: 0.5*m_{e}*v^{2} = KE, m_{e} = m0_{e}/sqrt(1-pow(v/c,2)),m0_{e}=0.511MeV/c^{2}
    me = 0.511 #in MeV
    b = 2*KE/me
    beta = math.sqrt(b*b+math.pow(b,4)/4)-math.pow(b,2)/2
    costheta = 1/(n*beta)
    print('beta=%f,n=%f,costheta=%f'%(beta,n,costheta))
if __name__ == '__main__':

    plots_path = '/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/plots_cls/'
    large_PMT_pos = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc1/data/Detector/Geometry/PMTPos_CD_LPMT.csv'#FIXME to J23.1.0-rc1 
    large_PMT_type= '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc1/data/Detector/Geometry/PMTType_CD_LPMT.csv'#FIXME to J23.1.0-rc1 
    m_Id_type_dict =  ID_type_map(file_pos=large_PMT_type, i_id=0)
    L_id_dict, Id_x_y_z_dict = vertexRec_map(file_pos=large_PMT_pos, i_id=0, i_x=1, i_y=2, i_z=3, i_theta=4, i_phi=5)
    m_GeoSvc = GeoSvc(file_pos=large_PMT_pos, i_id=0, i_x=1, i_y=2, i_z=3, i_theta=4, i_phi=5)
    print('L_id_dict=',len(L_id_dict),',len Id_x_y_z_dict=',len(Id_x_y_z_dict))
    (x_min, x_max, y_min, y_max) = draw_map(L_id_dict)
    assert ( x_min==0 and y_min==0 )
    ###########################################################
    parser = get_parser()
    parse_args = parser.parse_args()
    batch_size = parse_args.batch_size
    #m_use_mc_vertex = True
    #m_tcor_low = 225##trigger time and event time, 
    #m_tcor_high = m_tcor_low+70
    m_tcor_low = 200##trigger time and event time, 
    m_tcor_high = m_tcor_low+200
    m_time_match = 30
    m_t_shift = 10 #ns,FIXME, current do not understand why there is a shift for calib hit
    refractive = 1.54##FIXME, should we separate LS and Water ?
    m_velocity = 0.3*1000/refractive # mm/ns
    calculate_ck_angle(KE=2.45,n=refractive)#in MeV
    RfrIndxLS = 1.54
    RfrIndxWR = 1.355 ##/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc1/junosw/Reconstruction/OMILREC/share/tut_calib2rec.py
    R_scale  = 17700
    treeName='evt'
    tree =rt.TChain(treeName)
    tree.Add(parse_args.input)
    totalEntries=tree.GetEntries()
    ####### PMT Parameters
    para_file = '/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc1/data/Simulation/SimSvc/PMTSimParamSvc/PMTParam_CD_LPMT.root'
    pmt_effect = PMT_effect('pmt_effect', para_file, 0, 1000)
    ############## for gen direction semar#####################################
    m_h_smear = None
    if parse_args.save_dir_gen and parse_args.smear_gen:
        fin = rt.TFile(parse_args.direction_smear_input,'read')
        m_h_smear = fin.Get('h_e')
        fin.Close()
        print('m_h_smear sum=',m_h_smear.GetSumOfWeights())
    #######################################################
    #test_vertex_semar(E=1.,evt=10000)
    #test_dr = test_vertex_semar_v1(E0=1,E1=3,evt=10000)
    ########## for reweighting #############################################
    m_h_num = 0
    m_h_axis = 0
    m_x_min = 0
    m_x_max = 0
    if parse_args.doReWeight:
        assert ('.root' in parse_args.reweight_input_0 and '.root' in parse_args.reweight_input_1)
        f0 = rt.TFile(parse_args.reweight_input_0,'read')
        m_h_num =  f0.Get('h_sim_Qedep')
        f0.Close()
        f1 = rt.TFile(parse_args.reweight_input_1,'read')
        m_h_dem =  f1.Get('h_sim_Qedep')
        f1.Close()
        m_h_num.Divide(m_h_dem)
        m_h_axis = m_h_num.GetXaxis()
        m_x_min = m_h_axis.GetXmin()
        m_x_max = m_h_axis.GetXmax()
        print('do reweighting, m_x_min=',m_x_min,',m_x_max=',m_x_max)
    ###############################
    if parse_args.DoPlot:
        max_evt = parse_args.m_max_evt if (parse_args.m_max_evt > 0 and parse_args.m_max_evt < totalEntries) else totalEntries
        h_dict = draw_sim (tree, max_evt)
        color0 = 2
        color1 = 8
        color2 = 4
        color3 = 1
        normal_marker = 20
        fout = rt.TFile(parse_args.out_root,'recreate')
        fout.cd()
        for h in h_dict:
            if type(h_dict[h]) is list:continue
            h_dict[h].Write()
        fout.Write()
        fout.Close()
        for i in ['h_gen_N','h_gen_x','h_gen_y','h_gen_z','h_gen_r3','h_gen_p','h_gen_px','h_gen_py','h_gen_pz','h_gen_theta','h_gen_phi','h_gen_costheta','h_gen_cosdangle','h_gen2_x','h_gen2_y','h_gen2_z','h_gen2_r3','h_gen2_p','h_gen2_px','h_gen2_py','h_gen2_pz','h_gen2_theta','h_gen2_phi','h_gen2_costheta','h_dn_time','h_dn_tcor','h_sim_CK_ori_pass_ratio','h_sim_npe_CK_ori_Hama','h_sim_npe_CK_ori_NNVT','h_sim_totnpe','h_sim_Qedep','h_sim_n_ck_ori_early','h_sim_ck_ori_early_time','h_sim_s2b_early']:
            if h_dict[i].GetSumOfWeights() <=0: continue
            gr_dict={}
            gr_dict['#mu=%.3f'%(h_dict[i].GetMean())]=[h_dict[i],color0,'same:pel',normal_marker]
            plot_hists(hs_dict=gr_dict, out_name='%s'%h_dict[i].GetName(), title={'X':h_dict[i].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0])

        gr_dict={}
        gr_dict['tcor'      ]=[h_dict['h_sim_tcor']     ,color0,'same:pel',normal_marker]
        gr_dict['tcor(Hama)']=[h_dict['h_sim_tcor_Hama'],color1,'same:pel',normal_marker]
        gr_dict['tcor(NNVT)']=[h_dict['h_sim_tcor_NNVT'],color2,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp0_%s'%h_dict['h_sim_tcor'].GetName(), title={'X':h_dict['h_sim_tcor'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)
        gr_dict={}
        gr_dict['tcor(Hama)']       =[h_dict['h_sim_tcor_Hama']       ,color0,'same:pel',normal_marker]
        gr_dict['tcor(Hama,ck_ori)']=[h_dict['h_sim_tcor_CK_ori_Hama'],color1,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp1_%s'%h_dict['h_sim_tcor_Hama'].GetName(), title={'X':h_dict['h_sim_tcor_Hama'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)
        gr_dict={}
        gr_dict['tcor(NNVT)']       =[h_dict['h_sim_tcor_NNVT']       ,color0,'same:pel',normal_marker]
        gr_dict['tcor(NNVT,ck_ori)']=[h_dict['h_sim_tcor_CK_ori_NNVT'],color1,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp2_%s'%h_dict['h_sim_tcor_NNVT'].GetName(), title={'X':h_dict['h_sim_tcor_NNVT'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)
        gr_dict={}
        gr_dict['tcor']        =[h_dict['h_sim_tcor']       ,color0,'same:pel',normal_marker]
        gr_dict['tcor(ck_ori)']=[h_dict['h_sim_tcor_CK_ori'],color1,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp3_%s'%h_dict['h_sim_tcor'].GetName(), title={'X':h_dict['h_sim_tcor'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)
    
        gr_dict={}
        gr_dict['sel']        =[h_dict['h_dangle_gen_sel']       ,color0,'same:pel',normal_marker]
        gr_dict['ck_ori']     =[h_dict['h_dangle_gen_ck' ]       ,color1,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp4_%s'%h_dict['h_dangle_gen_sel'].GetName(), title={'X':h_dict['h_dangle_gen_sel'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)

        gr_dict={}
        gr_dict['sel']        =[h_dict['h_cosdangle_gen_sel']       ,color0,'same:pel',normal_marker]
        gr_dict['ck_ori']     =[h_dict['h_cosdangle_gen_ck' ]       ,color1,'same:pel',normal_marker]
        plot_hists(hs_dict=gr_dict, out_name='comp4_%s'%h_dict['h_cosdangle_gen_sel'].GetName(), title={'X':h_dict['h_cosdangle_gen_sel'].GetTitle(),'Y':'Events'}, rangeX=[0,0], rangeY=[0,0], doNorm=True)

        do_plot2d(hist=h_dict['h2_gen_KE1_KE2'],out_name='h2_gen_KE1_KE2',title={'X':'gen_{KE}^{1} (MeV)','Y':'gen_{KE}^{2} (MeV)'})
        for i in range(len(h_dict['gr_sim_xy_ckori'])):
            do_plot_gr(hist=h_dict['gr_sim_xy_ckori'][i],out_name='gr_sim_xy_ckori_rcut%.1f_idx%d'%(parse_args.gen_r_cut,i),title={'X':'x(mm)','Y':'y(mm)'})


    if parse_args.SaveH5:
        if batch_size < 0 : 
            batch_size = totalEntries
        batch = int(float(totalEntries)/batch_size)
        print ('total events=%d, batch_size=%d, batchs=%d, last=%d'%(totalEntries, batch_size, batch, totalEntries%batch_size))
        start = 0
        for i in range(batch):
            out_name = parse_args.output.replace('.h5','_batch%d_N%d.h5'%(i, batch_size))
            if parse_args.Save2D:
                root2hdf5_2D (batch_size, tree, start, out_name, L_id_dict, x_max, y_max, Id_x_y_z_dict, parse_args.Draw_data, parse_args.Draw_CK)
            elif parse_args.Save2D_detsim:
                root2hdf5_2D_detsim (batch_size, tree, start, out_name, L_id_dict, x_max, y_max, Id_x_y_z_dict, parse_args.Draw_data, False)
            elif parse_args.SavePoints_detsim:
                root2hdf5_Points_detsim (batch_size, tree, start, out_name)
            elif parse_args.SavePoints_detsim_v2:
                root2hdf5_Points_detsim_v2 (batch_size, tree, start, out_name)
            elif parse_args.SavePoints_detsim_v3:
                root2hdf5_Points_detsim_v3 (batch_size, tree, start, out_name)
            elif parse_args.SavePoints_calib:
                root2hdf5_Points_calib (batch_size, tree, start, out_name)
            elif parse_args.SavePoints_calib_v2:
                root2hdf5_Points_calib_v2 (batch_size, tree, start, out_name)
            start = start + batch_size
            if parse_args.m_max_evt>0 and start >= parse_args.m_max_evt:break
    print('done')  
