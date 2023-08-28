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
#SBATCH --job-name=test_gn_sim_ck
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_test_gnn_sim_ck.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_test_gnn_sim_ck.err
  
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
python $workpath/train_gnn_reg.py  --epochs 50 --lr 5e-4  --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_gnn_detsim_ck_epoch50.pth  --train_file $workpath/dataset/e-/detsim_ck_points/train.txt --valid_file $workpath/dataset/e-/detsim_ck_points/valid.txt --test_file $workpath/dataset/e-/detsim_ck_points/train.txt --outFile $workpath/Pred/reg_gnn_train_sim_ck.h5
python $workpath/train_gnn_reg.py  --epochs 50 --lr 5e-4  --batch 256 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_gnn_detsim_ck_epoch50.pth  --train_file $workpath/dataset/e-/detsim_ck_points/train.txt --valid_file $workpath/dataset/e-/detsim_ck_points/valid.txt --test_file $workpath/dataset/e-/detsim_ck_points/test.txt --outFile $workpath/Pred/reg_gnn_test_sim_ck.h5
