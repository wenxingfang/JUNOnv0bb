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
#SBATCH --job-name=te_tcut
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_test_2d_reg_sim_tcut.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_test_2d_reg_sim_tcut.err
  
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
#df -h
cd /hpcfs/juno/junogpu/fangwx
source /hpcfs/juno/junogpu/fangwx/setup_conda.sh
#conda activate pytorch1.71 
conda activate pytorch2p0
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/
python $workpath/train_2d_reg.py --fcs 1024 128 3 --Dropout 0.1 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_2d_detsim_tcut_epoch8.pth --cfg 'A3' --BatchNorm True --epochs 50 --lr 5e-4 --batch 128 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut/train.txt --valid_file $workpath/dataset/e-/detsim_tcut/valid.txt --test_file $workpath/dataset/e-/detsim_tcut/test.txt --outFile $workpath/Pred/reg_2d_test_sim_tcut.h5
python $workpath/train_2d_reg.py --fcs 1024 128 3 --Dropout 0.1 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_2d_detsim_tcut_epoch8.pth --cfg 'A3' --BatchNorm True --epochs 50 --lr 5e-4 --batch 128 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tcut/train.txt --valid_file $workpath/dataset/e-/detsim_tcut/valid.txt --test_file $workpath/dataset/e-/detsim_tcut/train.txt --outFile $workpath/Pred/reg_2d_train_sim_tcut.h5
