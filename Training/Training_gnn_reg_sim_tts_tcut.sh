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
#SBATCH --job-name=sim_tts_tcut
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_gnn_sim_tts_tcut.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_gnn_sim_tts_tcut.err
  
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
### loss 'Angle': res, epoch93,train_loss=1.3739497403846626,valid_loss=1.4004655303151217, lr=1.1745415408039613e-05
python $workpath/train_gnn_reg.py  --loss 'Angle' --epochs 100 --lr 5e-4  --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_tcut_points/test.txt --out_name $workpath/model/reg_gnn_detsim_tts_tcut_lossAngle.pth
##res: epoch49,train_loss=1.7983039914333723,valid_loss=1.805913388386324, lr=8.616789873315381e-07
#python $workpath/train_gnn_reg.py  --epochs 50 --lr 5e-4  --batch 256 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_tcut_points/test.txt --out_name $workpath/model/reg_gnn_detsim_tts_tcut.pth
