
import os
import ROOT as rt
import random

def create_scripts(batch_size, rand, input_file, out_path, script_path, tag):

    t_f = rt.TFile(input_file)
    t_tree = t_f.Get('evt') 
    total_evt = t_tree.GetEntries()
    t_f.Close()
    total_list = list(range(0,total_evt))
    if rand:
        random.shuffle(total_list)
    tmp_index = 0
    for i in range(0,total_evt,batch_size):
        tmp_list = total_list[i:i+batch_size]
        if len(tmp_list) != batch_size:continue
        with open('%s/evt_%d.txt'%(script_path,tmp_index),'w') as f:
            for ie in tmp_list:
                f.write('%s\n'%str(ie))
        with open('%s/job_%d.sh'%(script_path,tmp_index),'w') as f:
            f.write('#!/bin/bash\n')
            f.write('source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624\n')
            str_cmd ='''
python /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Mixing/root2h5_cls.py --SaveH5 True --addTTS %(addTTS)s --TTS_realistic %(realTTS)s --addT0Smear %(addT0Smear)s --T0_sigma %(T0_sigma)s --addDN %(addDN)s --addEnergySmear %(addEnergySmear)s --addVertexSmear %(addVertexSmear)s --smear_gen %(smear_gen)s --direction_smear_input /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/costh_smearing.root --sim_ratio %(simRatio)f --time_cut_down %(tcutDown)f --input %(input)s --input_evt %(evt_txt)s --output %(output)s\n
'''
            f.write(str_cmd%{'addTTS':str(addTTS),'realTTS':str(realTTS),'addT0Smear':str(addT0Smear),'T0_sigma':str(T0_sigma),'addDN':str(addDN),'addEnergySmear':str(addEnergySmear),'addVertexSmear':str(addVertexSmear),'smear_gen':str(smear_gen),'simRatio':simRatio,'tcutDown':tcutDown,'input':input_file,'evt_txt':'%sevt_%d.txt'%(script_path,tmp_index),'output':'%s/%s_%d.h5'%(out_path,tag,tmp_index)})
            tmp_index += 1
            

if __name__ == '__main__':

    m_particle = 'bkgs'#'bb0n'
    assert m_particle in ['bb0n','bkgs']
    
    addTTS  = False#True 
    realTTS = False#True
    addT0Smear = False
    T0_sigma = 2
    addDN = False
    addEnergySmear = False
    addVertexSmear = False
    simRatio = 0.05 
    tcutDown = -10 
    smear_gen = False
    #########################
    batch_size = 1000
    if m_particle == 'bb0n': 
        input_file = '/cefs/higgs/wxfang/JUNO/nv0bb/bb0n/detsim_assemb_bb0n_reweight.root' 
        out_path   = '/cefs/higgs/wxfang/JUNO/ForReco/cls_new_mix/bb0n_points_m10_earlys0p05_h5'
        script_path = '%s/jobs_bb0n/'%os.getcwd()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        create_scripts(batch_size=batch_size, rand=False, input_file=input_file, out_path=out_path, script_path=script_path, tag='bb0n')
        os.system('chmod +x %s/*.sh'%script_path)
    elif m_particle == 'bkgs': 
        input_file = '/cefs/higgs/wxfang/JUNO/nv0bb/detsim_assemb_tot_bkgs_reweight.root' 
        out_path   = '/cefs/higgs/wxfang/JUNO/ForReco/cls_new_mix/bkgs_points_m10_earlys0p05_h5'
        script_path = '%s/jobs_bkgs/'%os.getcwd()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        create_scripts(batch_size=batch_size, rand=True, input_file=input_file, out_path=out_path, script_path=script_path, tag='bkgs')
        os.system('chmod +x %s/*.sh'%script_path)

    print('done')
