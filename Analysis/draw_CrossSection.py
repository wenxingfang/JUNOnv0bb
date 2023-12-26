import ROOT as rt
import math
import sys
import gc
import numpy as np
import random
rt.gROOT.SetBatch(rt.kTRUE)
rt.TGaxis.SetMaxDigits(3);
rt.TH1.AddDirectory(0)


def cross_section(Enu,Te,flavor='e'):
    g1=0
    g2=0
    if flavor=='e':
        g1 = 0.73
        g2 = 0.23
    elif flavor=='anti-e':
        g2 = 0.73
        g1 = 0.23
    elif flavor=='mu':
        g1 = -0.27
        g2 =  0.23
    elif flavor=='anti-mu':
        g2 = -0.27
        g1 =  0.23
    cs = math.pow(g1,2) + math.pow(g2*(1-(Te/Enu)),2)-g1*g2*me*Te/math.pow(Enu,2)
    return cs


def cross_section_dcos(costh, Enu,flavor='e'):
    g1=0
    g2=0
    if flavor=='e':
        g1 = 0.73
        g2 = 0.23
    elif flavor=='anti-e':
        g2 = 0.73
        g1 = 0.23
    elif flavor=='mu':
        g1 = -0.27
        g2 =  0.23
    elif flavor=='anti-mu':
        g2 = -0.27
        g1 =  0.23
    me_Enu_sq = math.pow(Enu*(me+Enu),2)
    Enu_costh_sq = math.pow(Enu*costh,2)
    term1 = 4*me_Enu_sq*costh/math.pow(me_Enu_sq-Enu_costh_sq,2)
    term2 = g1*g1 + g2*g2*math.pow(1-(2*me*Enu*costh*costh/(me_Enu_sq-Enu_costh_sq)),2)-g1*g2*2*me*me*costh*costh/(me_Enu_sq-Enu_costh_sq)
    cs = term1*term2
    return cs




def draw_cs(flavor):
    gr = rt.TGraph()
    for Enu in range(25,100):
        Enu /= 10.
        cs = cross_section(Enu,m_Te,flavor)
        gr.SetPoint(gr.GetN(),Enu,cs)
    
    do_plot_gr(hist=gr,out_name='cs_%s'%(flavor),title={'X':'E_{#nu} (MeV)','Y':'#sigma (#sigma_{0}/m_{e})'})


def draw_cs_dcos(flavor, Enu):
    gr = rt.TGraph()
    for costh in range(-10,10):
        costh /= 10.
        cs = cross_section_dcos(costh, Enu,flavor)
        gr.SetPoint(gr.GetN(),costh,cs)
    
    do_plot_gr(hist=gr,out_name='cs_costh_%s_Enu_%.1f'%(flavor,Enu),title={'X':'cos#theta','Y':'d#sigma/dcos#theta (#sigma_{0})'})


def cal_costh(Enu,Te):
    return math.sqrt(Te/(2*me+Te))*(me+Enu)/Enu


def draw_cs_cos(flavor='e'):
    #h1 = rt.TH1F("h1","",-1,1,20)
    #xaxis = h1.GetXaxis()
    gr = rt.TGraph()
    values = np.linspace(2.46, 16, 50000)
    for Enu in values:
    #for Enu in range(2460,16000):
    #for Enu in range(25,160):
    #for Enu in range(25,100):
        #Enu /= 1000.
        costh = cal_costh(Enu,m_Te)
        if costh > 1:continue
        cs = cross_section_dcos(costh, Enu,flavor)
        flux = m_gr_b8sp.Eval(Enu)
        #print("Enu=",Enu,',costh=',costh,',flux=',flux)
        cs *= flux
        #bin = xaxis.FindBin(costh)
        #h1.SetBinContent(bin,h1.GetBinContent(bin)+cs)
        gr.SetPoint(gr.GetN(), costh, cs)
    do_plot_gr(hist=gr,out_name='cs_costh_%s'%(flavor),title={'X':'cos#theta','Y':'#sigma (A.U.)'})
    h1 = gr2TH(gr=gr, BinX=[70,0.86,1], norm = True)
    return h1
 
def gr2TH(gr, BinX, norm):

    h1 = rt.TH1F('h1_%s'%gr.GetName(),'',BinX[0],BinX[1],BinX[2])
    xaxis = h1.GetXaxis()
    for i in range(0,gr.GetN()):
        x = gr.GetX()[i]
        y = gr.GetY()[i]
        xbin = xaxis.FindBin(x)
        h1.SetBinContent(xbin,h1.GetBinContent(xbin)+y)
    if norm:
        h1.Scale(1/h1.GetSumOfWeights())
    for i in range(1,h1.GetNbinsX()+1):
        h1.SetBinError(i,0)
    return h1

def plot_hists(hs_dict, out_name, title, rangeX, rangeY, text ='', doNorm=False, DrawGrid=False, doNormOne=False, sep_list=[]):#hs_dict={'leg_name':[hist,color,drawoption,MarkerStyle]}
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
    dummy.GetXaxis().SetTitleOffset(1.)
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


    latex = rt.TLatex()
    latex.SetTextAlign(12)  # Text alignment (12 for left-aligned)
    latex.SetTextSize(0.04)  # Text size
    latex.SetNDC(True)  # Use normalized device coordinates
    
    x_pos = 0.4  # X position of the text (normalized device coordinates)
    y_pos = 0.92 # Y position of the text (normalized device coordinates)
    
    latex.DrawLatex(x_pos, y_pos, text)

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
    if 'logx' in out_name:
        canvas.SetLogx()
    if 'logy' in out_name:
        canvas.SetLogy()
    hist.SetMarkerColor(2)
    hist.SetMarkerStyle(8)
    hist.GetXaxis().SetTitle(title['X'])
    hist.GetYaxis().SetTitle(title['Y'])
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.Draw("AP")
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()

def do_plot_gr_v2(hist,out_name,title,Range):
    canvas=rt.TCanvas("%s"%(out_name),"",800,800)
    canvas.cd()
    canvas.SetTopMargin(0.13)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.15)
    if 'logx' in out_name:
        canvas.SetLogx()
    if 'logy' in out_name:
        canvas.SetLogy()
    hframe = rt.TH1F("hframe_%s"%out_name, "", 1, Range['X'][0], Range['X'][1])
    # Set the minimum and maximum values of the Y-axis for the histogram frame
    hframe.SetStats(rt.kFALSE)
    hframe.SetMinimum(Range['Y'][0])
    hframe.SetMaximum(Range['Y'][1])
    hframe.GetXaxis().SetTitle(title['X'])
    hframe.GetYaxis().SetTitle(title['Y'])
    hframe.GetXaxis().SetTitleOffset(1.2)
    hframe.Draw()
    hist.SetMarkerColor(2)
    hist.SetMarkerStyle(8)
    hist.Draw("PL")
    canvas.SaveAs("%s/%s.png"%(plots_path,out_name))
    del canvas
    gc.collect()
    



def draw_b8sp():
    gr = rt.TGraph()
    with open(b8sp_file,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            item = line.split()
            E = float(item[0])
            flux = float(item[1])
            gr.SetPoint(gr.GetN(),E,flux)
    do_plot_gr(hist=gr,out_name='B8sp',title={'X':'E_{#nu} [MeV]','Y':'A.U.'})
    do_plot_gr_v2(hist=gr,out_name='B8sp_logx_logy',title={'X':'E_{#nu} [MeV]','Y':'A.U.'}, Range={'X':[0.01,20],'Y':[1e-5,1e6]})
    return gr 


if __name__ == '__main__':
    print('start...')
    me=0.511#MeV
    m_Te = 2.458
    b8sp_file = '/junofs/users/wxfang/JUNO/OEC/J23.1.0-rc2/junosw/Generator/NuSolGen/data/b8spectrum.dat'
    plots_path = './plot_cs'
    draw_cs(flavor='e')
    draw_cs(flavor='anti-e')
    draw_cs(flavor='mu')
    draw_cs(flavor='anti-mu')
    #draw_cs_dcos(flavor='e', Enu=3)
    #draw_cs_dcos(flavor='e', Enu=4)
    #draw_cs_dcos(flavor='e', Enu=5)
    
    m_gr_b8sp = draw_b8sp() 
    h_e      = draw_cs_cos(flavor='e') 
    h_anti_e = draw_cs_cos(flavor='anti-e') 
    h_mu     = draw_cs_cos(flavor='mu') 
    h_anti_mu= draw_cs_cos(flavor='anti-mu') 
    fout = rt.TFile('costh_smearing.root','recreate')
    fout.cd()
    fout.WriteObject(h_e     ,"h_e")
    fout.WriteObject(h_anti_e,"h_anti_e")
    fout.WriteObject(h_mu    ,"h_mu")
    fout.WriteObject(h_anti_mu,"h_anti_mu")
    fout.Close()
        
    h_dict={}
    h_dict['#nu_{e}']             =[h_e      ,2,'same:pl',20]
    h_dict['#bar{#nu}_{e}']       =[h_anti_e ,4,'same:pl',20]
    h_dict['#nu_{#mu/#tau}']      =[h_mu     ,1,'same:pl',20]
    h_dict['#bar{#nu}_{#mu/#tau}']=[h_anti_mu,8,'same:pl',20]
    plot_hists(hs_dict=h_dict, out_name='h1_cs_costh',title={'X':'cos#theta (#nu_{i},e_{f})','Y':'#sigma (A.U.)'}, rangeX=[0.85,1.01], rangeY=[0,0.03], text='T_{e}=%.3f MeV'%m_Te)
    print('done')
