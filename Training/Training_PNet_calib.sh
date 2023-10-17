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
#SBATCH --job-name=pn_calib
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_calib.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/log_train_PNet_calib.err
  
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
### Hama, 0.1 earlys, res:
python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points_Hama_m20ns_earlys0p1/train.txt --valid_file $workpath/dataset/e-/calib_points_Hama_m20ns_earlys0p1/valid.txt --test_file $workpath/dataset/e-/calib_points_Hama_m20ns_earlys0p1/test.txt --out_name $workpath/model/reg_PNet_calib.pth
### Hama, 0.05 earlys, res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points_Hama_m10ns_earlys0p05/train.txt --valid_file $workpath/dataset/e-/calib_points_Hama_m10ns_earlys0p05/valid.txt --test_file $workpath/dataset/e-/calib_points_Hama_m10ns_earlys0p05/test.txt --out_name $workpath/model/reg_PNet_calib.pth
### Hama 500 earlys, res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 25 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points_500earlys_Hama/train.txt --valid_file $workpath/dataset/e-/calib_points_500earlys_Hama/valid.txt --test_file $workpath/dataset/e-/calib_points_500earlys_Hama/test.txt --out_name $workpath/model/reg_PNet_calib.pth
### 500 earlys, res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points_500earlys/train.txt --valid_file $workpath/dataset/e-/calib_points_500earlys/valid.txt --test_file $workpath/dataset/e-/calib_points_500earlys/test.txt --out_name $workpath/model/reg_PNet_calib.pth
### only Hama, res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points_Hama/train.txt --valid_file $workpath/dataset/e-/calib_points_Hama/valid.txt --test_file $workpath/dataset/e-/calib_points_Hama/test.txt --out_name $workpath/model/reg_PNet_calib_hama.pth
### res:
#python $workpath/train_PNet_reg.py --loss 'Angle' --ps_features 12 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file $workpath/dataset/e-/calib_points/train.txt --valid_file $workpath/dataset/e-/calib_points/valid.txt --test_file $workpath/dataset/e-/calib_points/test.txt --out_name $workpath/model/reg_PNet_calib.pth
