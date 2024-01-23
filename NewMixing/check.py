import ROOT as rt


if __name__ == '__main__':

    dict_file = {}
    dict_file['bb0n']  = ['/cefs/higgs/wxfang/JUNO/nv0bb/bb0n/detsim_assemb_bb0n.root'  ,0.7]
    dict_file['bb2n']  = ['/cefs/higgs/wxfang/JUNO/nv0bb/bb2n/detsim_assemb_bb2n.root'  ,0.2]
    dict_file['em'  ]  = ['/cefs/higgs/wxfang/JUNO/nv0bb/e-/detsim_assemb_em.root'      ,0.7]
    dict_file['Bi214'] = ['/cefs/higgs/wxfang/JUNO/nv0bb/Bi214/detsim_assemb_Bi214.root',0.2]
    dict_file['C10']   = ['/cefs/higgs/wxfang/JUNO/nv0bb/C10/detsim_assemb_C10.root'    ,0.053]
    dict_file['He6']   = ['/cefs/higgs/wxfang/JUNO/nv0bb/He6/detsim_assemb_He6.root'    ,0.063]
    dict_file['Li8']   = ['/cefs/higgs/wxfang/JUNO/nv0bb/Li8/detsim_assemb_Li8.root'    ,0.016]
    dict_file['Xe137'] = ['/cefs/higgs/wxfang/JUNO/nv0bb/Xe137/detsim_assemb_Xe137.root',0.07 ]

    n_bin = 100
    bin_low = 2.3   
    bin_high = 2.4
 
    for i in dict_file:
        f = rt.TFile.Open(dict_file[i][0],'read')
        tree = f.Get('evt')
        dict_file[i].append(tree.GetEntries())
        '''
        h_sum_E = rt.TH1F("h_sum_E_%s"%i,'',n_bin,bin_low,bin_high)
        for k in range(tree.GetEntries()):
            tree.GetEntry(k)
            sum_E = getattr(tree, "sum_Qedep")
            h_sum_E.Fill(sum_E)
        print('%s sumw2 %d'%(i,h_sum_E.GetSumOfWeights()))
        dict_file[i].append(h_sum_E)
        '''
        f.Close()
    
    for i in dict_file:
        print('%s:evt %d,exp %d'%(i,dict_file[i][2], dict_file['em'][2]*dict_file[i][1]/dict_file['em'][1]))
 

    '''    
    h_tot_bkgs = rt.TH1F("h_sum_E_tot_bkgs",'',n_bin,bin_low,bin_high)
    for i in dict_file:
        if i == 'bb0n':continue
        h_tot_bkgs.Add(dict_file[i][3])
    '''    
    '''
    # Open the input rt files
    file1 = rt.TFile.Open("file1.root")
    file2 = rt.TFile.Open("file2.root")
    
    # Get the TTrees from the input rt files
    tree1 = file1.Get("tree_name")  # Replace "tree_name" with the actual name of the TTree in file1
    tree2 = file2.Get("tree_name")  # Replace "tree_name" with the actual name of the TTree in file2
    
    # Create a new output rt file
    outputFile = rt.TFile("output.root", "RECREATE")
    
    # Create new TTrees in the output rt file
    outputTree1 = tree1.CloneTree(0)  # Create an empty TTree with the same structure as tree1
    outputTree2 = tree2.CloneTree(0)  # Create an empty TTree with the same structure as tree2
    
    # Copy the entries randomly from the input TTrees to the output TTrees
    nEntries1 = tree1.GetEntries()
    nEntries2 = tree2.GetEntries()
    
    for i in range(nEntries1 + nEntries2):
        sourceTree = None
        if i < nEntries1:
            sourceTree = tree1
        else:
            sourceTree = tree2
    
        sourceTree.GetEntry(i % sourceTree.GetEntries())
        if sourceTree == tree1:
            outputTree1.Fill()
        else:
            outputTree2.Fill()
    
    # Write the output TTrees to the output rt file and close the files
    outputTree1.Write()
    outputTree2.Write()
    outputFile.Close()
    file1.Close()
    file2.Close()
    '''
