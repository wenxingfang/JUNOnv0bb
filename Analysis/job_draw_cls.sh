#!/bin/bash
#hostname
source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root624
export mydir=/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana
#python $mydir/root2h5_cls.py --sim_ratio 0.05 --time_cut_down -10 --DoPlot True --input "/cefs/higgs/wxfang/JUNO/nv0bb/C10/detsim_assemb/assemb_*.root" --out_root $mydir/hist_C10.root --m_max_evt 20000
python $mydir/root2h5_cls.py --sim_ratio 0.05 --time_cut_down -10 --DoPlot True --input "/cefs/higgs/wxfang/JUNO/nv0bb/e-/detsim_2p4_2p46MeV/Assmeb_*.root" --out_root $mydir/hist_e-.root --m_max_evt 10000
#python $mydir/root2h5_cls.py --sim_ratio 0.05 --time_cut_down -10 --DoPlot True --input "/cefs/higgs/wxfang/JUNO/nv0bb/bbn0/detsim/Assmeb_*.root"   --out_root $mydir/hist_bb0n.root --m_max_evt 10000
#python $mydir/root2h5_cls.py --sim_ratio 0.05 --time_cut_down -10 --DoPlot True --input "/cefs/higgs/wxfang/JUNO/nv0bb/e-/detsim_2p4_2p46MeV/Assmeb_*.root" --out_root $mydir/hist_e-.root --doReWeight True --reweight_input_0 '/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_em.root' --reweight_input_1 '/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_bbn0.root' --m_max_evt 10000
#python $mydir/root2h5_cls.py --sim_ratio 0.05 --time_cut_down -10 --DoPlot True --input "/cefs/higgs/wxfang/JUNO/nv0bb/bbn0/detsim/Assmeb_*.root" --out_root $mydir/hist_bb0n.root --doReWeight True --reweight_input_1 '/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_em.root' --reweight_input_0 '/junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/norm_hist_bbn0.root' --m_max_evt 10000
#python root2h5_cls.py --only_Hama True --DoPlot True --addEnergySmear True --addVertexSmear True --addDN True --addTTS True --TTS_realistic True --time_cut_down -10 --time_cut 1 --sim_ratio 0.05 --calib_ratio 0.05 --calibtime_cut_low -10 --input "/cefs/higgs/wxfang/JUNO/nv0bb/e-/Assemb_T0/Assmeb_*.root" --m_max_evt 1000
