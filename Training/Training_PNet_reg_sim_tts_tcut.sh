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
#SBATCH --job-name=pn_sim_tts_cut
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut.err
  
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
### using real TTS and 1ns time cut, add T0 shift 5ns,res:epoch48,train_loss=1.403165043109712,valid_loss=1.4036736492002486, lr=7.150532420057921e-07
python $workpath/train_PNet_reg.py --T0_shift True --T0_shift_val 5 --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_1ns_T05ns.pth
### using real TTS and 1ns time cut, add T0 shift 2ns,res:epoch48,train_loss=1.4202215602604595,valid_loss=1.412492979963841, lr=7.150532420057921e-07
#python $workpath/train_PNet_reg.py --T0_shift True --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_1ns_T0.pth
### using real TTS and 1ns time cut, remove original hittime,res:epoch48,train_loss=1.4231770319453296,valid_loss=1.416029101358202, lr=7.150532420057921e-07.Con:better to have ori time
#python $workpath/train_PNet_reg.py --rm_tori True --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_1ns_rmtori.pth
### using real TTS and 1ns time cut,res:epoch48,train_loss=1.396540325646024,valid_loss=1.4032877853499084, lr=7.150532420057921e-07
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_1ns.pth
### using angle loss, and Tanh() activation for output,res:epoch48,train_loss=1.3750812391460738,valid_loss=1.3812620458472045, lr=7.150532420057921e-07
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_tcut_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_tcut_angle.pth
#python $workpath/train_PNet_reg.py --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_tcut_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_tcut_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_tcut_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_tcut.pth
