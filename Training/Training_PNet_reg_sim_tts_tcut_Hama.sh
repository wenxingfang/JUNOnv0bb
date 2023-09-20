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
#SBATCH --job-name=hama_pn_sim_tts_cut
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut_Hama.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut_Hama.err
  
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
### using angle loss, and Tanh() activation for output,res:epoch48,train_loss=1.4163813460525565,valid_loss=1.4286628590702481, lr=7.150532420057921e-07
python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_tcut_Hama_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_tcut_Hama_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_tcut_Hama_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_tcut_angle_hama.pth
