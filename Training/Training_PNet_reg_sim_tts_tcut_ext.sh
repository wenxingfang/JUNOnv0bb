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
#SBATCH --job-name=pn_vtx_en
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut_ext.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_sim_tts_tcut_ext.err
  
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
### knn=5, use earlys ratio = 0.1, add DN, using real TTS and , no T0 shift,res:epoch48,train_loss=1.470160436542504,valid_loss=1.4830072988713834
#python $workpath/train_PNet_reg.py --knn 5 --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p1_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p1_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p1_points/test.txt --out_name $workpath/model/reg_PNet_detsim_early0p1.pth
### use earlys ratio = 0.09, add DN, using real TTS and , no T0 shift,res:epoch48,train_loss=1.473422231347964,valid_loss=1.4864057159047555
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p08_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p08_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_m10ns_earlys0p08_points/test.txt --out_name $workpath/model/reg_PNet_detsim_early0p08.pth
### use 250 earlys, add DN, using real TTS and 1ns time cut, no T0 shift,res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 11 --epochs 10 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_1000earlys_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_1000earlys_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_DN_vtx_en_1000earlys_points/test.txt --out_name $workpath/model/reg_PNet_detsim_1000early.pth
### add vertex and energy smearing, add DN, using real TTS and 3ns time cut, add T0 shift 5ns,res:epoch48,train_loss=1.4774672634106214,valid_loss=1.4822358642925957, lr=7.150532420057921e-07
#python $workpath/train_PNet_reg.py --T0_shift True --T0_shift_val 5 --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_3ns_DN_m10ns_vtx_en_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_3ns_DN_m10ns_vtx_en_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_3ns_DN_m10ns_vtx_en_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_3ns_T05ns_DN_vtx_en.pth
### add vertex and energy smearing, add DN, using real TTS and 1ns time cut, add T0 shift 5ns,res:epoch48,train_loss=1.4810812652516399,valid_loss=1.4861026513787179
python $workpath/train_PNet_reg.py --T0_shift True --T0_shift_val 5 --loss 'Angle' --ps_features 11 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_DN_m10ns_vtx_en_points/train.txt --valid_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_DN_m10ns_vtx_en_points/valid.txt --test_file $workpath/dataset/e-/detsim_tts_real_tcut_1ns_DN_m10ns_vtx_en_points/test.txt --out_name $workpath/model/reg_PNet_detsim_tts_real_tcut_1ns_T05ns_DN_vtx_en.pth
