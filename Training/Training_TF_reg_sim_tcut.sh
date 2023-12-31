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
#SBATCH --job-name=tf_tanh
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_TF_sim_tcut.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_TF_sim_tcut.err
  
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
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb
### unsort, no ps encoding, using loss AngleThetaPhi, res:
python $workpath/train_TF_reg.py --Restore True  --restore_file $workpath/model/reg_TF_detsim_tcut_thetaphi_epoch48.pth --last_act 'tanh' --loss 'AngleThetaPhi' --sort_idx -1 --psencoding False --fcs 1024 128 2 --ps_features 11 --epochs 100 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_detsim_tcut_thetaphi_re.pth
#python $workpath/train_TF_reg.py --last_act 'tanh' --loss 'AngleThetaPhi' --sort_idx -1 --psencoding False --fcs 1024 128 2 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_detsim_tcut_thetaphi.pth
### unsort, no ps encoding, using last activation tanh, res:epoch48,train_loss=0.9829543652031573,valid_loss=0.9814210061470817, lr=7.150532420057921e-07
#python $workpath/train_TF_reg.py --last_act 'tanh' --loss 'Angle' --sort_idx -1 --psencoding False --fcs 1024 128 3 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_detsim_tcut_tanh.pth
### unsort, no ps encoding, using Angle loss, res:
#python $workpath/train_TF_reg.py --loss 'Angle' --sort_idx -1 --psencoding False --fcs 1024 128 3 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_detsim_tcut_loss.pth
### unsort, no ps encoding, longer training, res: similar
#python $workpath/train_TF_reg.py --sort_idx -1 --psencoding False --fcs 1024 128 3 --ps_features 11 --epochs 100 --lr 1e-3  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_detsim_tcut_lr.pth
### (sort by hit_cor, with ps encoding), similar performance with (unsort, no ps encoding)
#python $workpath/train_TF_reg.py --sort_idx 5 --psencoding True --fcs 1024 128 3 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_sort5_psen_detsim_tcut.pth
### unsort by hit_cor, with ps encoding, res: not good. has large fluctation
#python $workpath/train_TF_reg.py --sort_idx -1 --psencoding True --fcs 1024 128 3 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_unsort_psen_detsim_tcut.pth
### unsort, no ps encoding, res: stable
#python $workpath/train_TF_reg.py --sort_idx -1 --psencoding False --fcs 1024 128 3 --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tcut_points/test.txt --out_name $workpath/model/reg_TF_unsort_detsim_tcut.pth
