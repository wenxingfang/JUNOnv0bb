
import os

m_particle = 'e-'
assert m_particle in ['e-','bb0n','C10','Bi214']

addTTS  = True#False#True 
realTTS = True#False#True
addT0Smear = False
T0_sigma = 2
addDN = True
addEnergySmear = True
addVertexSmear = False
simRatio = 0.05 
tcutDown = -10 
f_in1       = '' 
out_path    = '' 
script_path = '' 
if m_particle=='e-':
    f_in1       = 'e-_2p4_2p46MeV_detsim.txt'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_tts_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_tts_gen_smear_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_gen_smear_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_all_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_tts_t0_h5'
    out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/e-_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    script_path = './jobs_each_em/'
elif m_particle=='bb0n':
    f_in1       = 'bb0n_detsim.txt'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/bb0n_points_m10_earlys0p05_1D_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/bb0n_points_m10_earlys0p05_1D_tts_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/bb0n_points_m10_earlys0p05_1D_all_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/bb0n_points_m10_earlys0p05_1D_tts_t0_h5'
    out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/bb0n_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    script_path = './jobs_each_bb0n/'
elif m_particle=='C10':
    f_in1       = 'C10_detsim.txt'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_all_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_t0_h5'
    out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    script_path = './jobs_each_C10/'
elif m_particle=='Bi214':
    f_in1       = 'Bi214_detsim.txt'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_all_h5'
    #out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_t0_h5'
    out_path    = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    script_path = './jobs_each_Bi214/'


if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(script_path):
    os.makedirs(script_path)

f_list =[]
f_list.append(f_in1)


f_out =None
ibat = 0
isFirst=True
batch_size=1


index = 0
for f_In in f_list:
    with open(f_In,'r') as f:
        for line in f.readlines():
            if '#' in line:continue
            line = line.replace('\n','')
            tag = line.split('/')[-1]
            tag = tag.replace('.root','')
            if isFirst or ibat%batch_size==0:
                if isFirst==False: f_out.close()
                f_out = open('%s/job_%d.sh'%(script_path, index),'w')
                index += 1
                f_out.write('#!/bin/bash\n')
                f_out.write('hostname\n')
                #f_out.write('source /junofs/users/wxfang/FastSim/setup_conda.sh\n')
                #f_out.write('conda activate root2hdf5_pyTorch\n')
                f_out.write('source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624\n')
                isFirst=False
            #f_out.write('python /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/e-/Ana/root2hits1D.py --SaveH5 True --SavePoints_detsim_v2 True --only_Hama True --sim_ratio 0.05 --addT0Smear True --T0_sigma 2 --addTTS True --TTS_realistic True --addDN True --time_cut_down -10 --addEnergySmear True --addVertexSmear True --input "%s" --output %s/%s_%d.h5 --batch_size -1 \n'%(line,out_path,tag,index))
            str_cmd = ''
            if m_particle=='e-':
                str_cmd ='''
python /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/root2h5_cls.py --SaveH5 True --SavePoints_detsim_v3 True  --addTTS %(addTTS)s --TTS_realistic %(realTTS)s --addT0Smear %(addT0Smear)s --T0_sigma %(T0_sigma)s --addDN %(addDN)s --addEnergySmear %(addEnergySmear)s --addVertexSmear %(addVertexSmear)s --doEcut True  --save_dir_gen True --smear_gen True --direction_smear_input /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/costh_smearing.root  --sim_ratio %(simRatio)f  --time_cut_down %(tcutDown)f --input %(input)s --output %(output)s --batch_size -1 --doReWeight True --reweight_input_0 /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_em.root --reweight_input_1 /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_bbn0.root\n
'''
            elif m_particle=='bb0n':
                str_cmd ='''
python /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/root2h5_cls.py --SaveH5 True --SavePoints_detsim_v3 True  --addTTS %(addTTS)s --TTS_realistic %(realTTS)s --addT0Smear %(addT0Smear)s --T0_sigma %(T0_sigma)s --addDN %(addDN)s --addEnergySmear %(addEnergySmear)s --addVertexSmear %(addVertexSmear)s --doEcut True  --save_dir_random True --sim_ratio %(simRatio)f  --time_cut_down %(tcutDown)f --input %(input)s --output %(output)s --batch_size -1 --doReWeight True --reweight_input_1 /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_em.root --reweight_input_0 /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_bbn0.root\n
'''
            elif m_particle in ['C10','Bi214']:
                str_cmd ='''
python /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/root2h5_cls.py --SaveH5 True --SavePoints_detsim_v3 True  --addTTS %(addTTS)s --TTS_realistic %(realTTS)s --addT0Smear %(addT0Smear)s --T0_sigma %(T0_sigma)s --addDN %(addDN)s --addEnergySmear %(addEnergySmear)s --addVertexSmear %(addVertexSmear)s --doEcut True --E_min 2.35 --E_max 2.38 --save_dir_random True --sim_ratio %(simRatio)f  --time_cut_down %(tcutDown)f --input %(input)s --output %(output)s --batch_size -1\n
'''
            f_out.write(str_cmd%{'addTTS':str(addTTS),'realTTS':str(realTTS),'addT0Smear':str(addT0Smear),'T0_sigma':str(T0_sigma),'addDN':str(addDN),'addEnergySmear':str(addEnergySmear),'addVertexSmear':str(addVertexSmear),'simRatio':simRatio,'tcutDown':tcutDown,'input':line,'output':'%s/%s_%d.h5'%(out_path,tag,index)})

            ibat+=1
if f_out is not None: f_out.close()
os.system('chmod +x %s/*.sh'%script_path)
print('done')
