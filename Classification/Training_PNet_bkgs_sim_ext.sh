#! /bin/bash
######## Part 1 #########
# Script parameters     #
#########################
  
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
  
# Specify the QOS, mandatory option
#SBATCH --qos=normal
  
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=pn_ext
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_train_PNet_bkgs_sim_ext.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_train_PNet_bkgs_sim_ext.err
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000
#
# Specify how many GPU cards to us:
#SBATCH --gres=gpu:v100:1
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
#hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
#conda activate pytorch2p0
conda activate pyTorch_1p8
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls
### use earlys ratio = 0.05, tts, dn, energy smear effects,res:
#export datasetpath=$workpath/dataset/detsim_m10_earlys0p05_tts_dn_ens_points
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_tts_dn_ens_sim.pth
### use earlys ratio = 0.05, tts, t0 effects,res:epoch49,train_loss=0.6599050250315127,valid_loss=0.6689804299567688, lr=9.803357293520456e-05
#export datasetpath=$workpath/dataset/detsim_m10_earlys0p05_tts_t0_points
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_tts_t0_sim.pth
### use earlys ratio = 0.05, all effects,res:epoch34,train_loss=0.6788771330048191,valid_loss=0.6809972333864968, lr=0.000361230
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_all_points
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_all_sim.pth
### use earlys ratio = 0.05, more train, e- dir smear, no tts ,res:epoch35,train_loss=0.4900724100215094,valid_loss=0.4958924563945883, lr=0.0003455198
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/bkgs_em_dir_smear
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_dirSmear_sim.pth
### use earlys ratio = 0.05, more train, tts, e- dir smear, T0 shift ,res:epoch35,train_loss=0.6034487774262923,valid_loss=0.6093624276441855, lr=0.000332162
export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/bkgs_em_dir_smear
python $workpath/train_PNet_bkgs.py --T0_shift True --T0_shift_val 5 --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_tts_dirSmear_t0s5_sim.pth
### use earlys ratio = 0.05, more train, tts ,res:epoch39,train_loss=0.6358970573475194,valid_loss=0.6402444883841141, lr=0.000270723
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/bkgs
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_tts_sim.pth
### use earlys ratio = 0.05, more train,res:epoch36,train_loss=0.46163976615151536,valid_loss=0.46333999269935977, lr=0.00032496206176311484
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_add_bkgs
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/PNet_bkgs_sim.pth
