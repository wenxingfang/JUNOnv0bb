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
#SBATCH --job-name=pn_bkgs_test
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNet_bkgs_sim.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNet_bkgs_sim.err
  
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
hostname
df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
#conda activate pytorch2p0
conda activate pyTorch_1p8
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls

### add direction semaring, tts, T0 shift
export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/bkgs_em_dir_smear
export model=$workpath/model/PNet_bkgs_tts_dirSmear_t0s5_sim_epoch35.pth
python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_dirSmear_tts_t0s5_sim.h5
python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_dirSmear_tts_t0s5_sim.h5
python $workpath/train_PNet_bkgs.py --T0_shift True --T0_shift_val 5 --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_dirSmear_tts_same_t0s5_sim.h5
python $workpath/train_PNet_bkgs.py --T0_shift True --T0_shift_val 5 --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_dirSmear_tts_same_t0s5_sim.h5
### all effects
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_all_points
#export model=$workpath/model/PNet_bkgs_all_sim_epoch34.pth
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_all_sim.h5
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_all_sim.h5
### add direction semaring, tts
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/bkgs_em_dir_smear
#export model=$workpath/model/PNet_bkgs_tts_dirSmear_sim_epoch32.pth
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_dirSmear_tts_sim.h5
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_dirSmear_tts_sim.h5
### add direction semaring, no tts
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/bkgs_em_dir_smear
#export model=$workpath/model/PNet_bkgs_dirSmear_sim_epoch35.pth
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_dirSmear_sim.h5
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $model --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_dirSmear_sim.h5
### add tts
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/bkgs
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_bkgs_tts_sim_epoch39.pth --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_tts_sim.h5
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_bkgs_tts_sim_epoch39.pth --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_tts_sim.h5
### use earlys ratio = 0.05, more train,res:epoch36,train_loss=0.46163976615151536,valid_loss=0.46333999269935977, lr=0.00032496206176311484
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_add_bkgs
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_bkgs_sim_epoch36.pth --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --outFile $workpath/Pred/cls_PNet_bkgs_test_sim.h5
#python $workpath/train_PNet_bkgs.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_bkgs_sim_epoch36.pth --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_train.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_train.txt --outFile $workpath/Pred/cls_PNet_bkgs_train_sim.h5
