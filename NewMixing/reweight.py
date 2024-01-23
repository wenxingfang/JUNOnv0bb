import ROOT as rt
import random

def reweight(fin_name,fout_name,h_ratio):
 
    axis = h_ratio.GetXaxis() 
    # Open the input rt files
    fin = rt.TFile.Open(fin_name,'read')
    
    # Get the TTrees from the input rt files
    treein = fin.Get("evt")
    
    # Create a new output rt file
    outputFile = rt.TFile(fout_name, "RECREATE")
    
    # Create new TTrees in the output rt file
    outputTree = treein.CloneTree(0)  # Create an empty TTree with the same structure as tree1
    
    # Copy the entries randomly from the input TTrees to the output TTrees
    nEntries = treein.GetEntries()
    
    for i in range(nEntries):
        treein.GetEntry(i)
        sum_Qedep = getattr(treein, "sum_Qedep")
        val = h_ratio.GetBinContent(axis.FindBin(sum_Qedep))
        if val < 1: 
            outputTree.Fill()
        else:
            if random.random() < (1/val):
                outputTree.Fill()
    
    # Write the output TTrees to the output rt file and close the files
    outputTree.Write()
    outputFile.Close()
    fin.Close()
    print('saved %s'%fout_name)

if __name__ == '__main__':


    n_bin = 100
    bin_low = 2.3   
    bin_high = 2.4

    dict_file = {}
    dict_file['bb0n'] = ['/cefs/higgs/wxfang/JUNO/nv0bb/bb0n/detsim_assemb_bb0n.root', rt.TH1F("h_sum_E_bb0n",'',n_bin,bin_low,bin_high)]
    dict_file['bkgs'] = ['/cefs/higgs/wxfang/JUNO/nv0bb/detsim_assemb_tot_bkgs.root' , rt.TH1F("h_sum_E_bkgs",'',n_bin,bin_low,bin_high)]

 
    for i in dict_file:
        f = rt.TFile.Open(dict_file[i][0],'read')
        tree = f.Get('evt')
        for k in range(tree.GetEntries()):
            tree.GetEntry(k)
            sum_E = getattr(tree, "sum_Qedep")
            dict_file[i][1].Fill(sum_E)
        print('%s sumw2 %d'%(i,dict_file[i][1].GetSumOfWeights()))
        f.Close()

    h_bb0n = dict_file['bb0n'][1]
    h_bkgs = dict_file['bkgs'][1]

    h_bb0n_copy = h_bb0n.Clone('clone_%s'%h_bb0n.GetName())
    h_bkgs_copy = h_bkgs.Clone('clone_%s'%h_bkgs.GetName())

    h_bb0n.Divide(h_bkgs) 
    h_bkgs_copy.Divide(h_bb0n_copy) 


    reweight(fin_name=dict_file['bb0n'][0],fout_name='/cefs/higgs/wxfang/JUNO/nv0bb/bb0n/detsim_assemb_bb0n_reweight.root',h_ratio=h_bb0n)
    reweight(fin_name=dict_file['bkgs'][0],fout_name='/cefs/higgs/wxfang/JUNO/nv0bb/detsim_assemb_tot_bkgs_reweight.root' ,h_ratio=h_bkgs_copy)

    print('Done!')
