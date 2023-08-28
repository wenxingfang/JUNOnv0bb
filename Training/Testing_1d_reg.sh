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
#SBATCH --job-name=test_reg
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/CNN_plane/log_test_1d_reg.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/CNN_plane/log_test_1d_reg.err
  
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
conda activate pytorch1.71 
which python
/usr/local/cuda/bin/nvcc --version
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/C14Mix/CNN_plane/
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.3 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/tuning/model/m_1d_reg_index_7_epoch35.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/train.txt --channel 2 --outFile $workpath/Pred/reg_1d_train_7ep35.h5
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.3 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/tuning/model/m_1d_reg_index_7_epoch35.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/valid.txt --channel 2 --outFile $workpath/Pred/reg_1d_valid_7ep35.h5
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.3 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/tuning/model/m_1d_reg_index_7_epoch35.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/test.txt  --channel 2 --outFile $workpath/Pred/reg_1d_test_7ep35.h5
##python $workpath/train_1d.py --fcs 1024 128 2 --Dropout 0.3 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/tuning/model/index_7_epoch18.pth --cfg 'A3' --BatchNorm True --frac_ep 0.2 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --check_train False --test_file $workpath/dataset/add_1d/test_iso_40kHz.txt --channel 2 --outFile $workpath/Pred/Cls_1d_test_iso_40kHz_ep18.h5
#######
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.1 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_1d_epoch19.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/train.txt --channel 2 --outFile $workpath/Pred/reg_1d_train_ep19.h5
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.1 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_1d_epoch19.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/valid.txt --channel 2 --outFile $workpath/Pred/reg_1d_valid_ep19.h5
python $workpath/train_1d_reg.py --fcs 1024 128 1 --Dropout 0.1 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/reg_1d_epoch19.pth --cfg 'A3' --BatchNorm True --frac_ep 0 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --test_file $workpath/dataset/add_1d/test.txt  --channel 2 --outFile $workpath/Pred/reg_1d_test_ep19.h5
#python $workpath/train_1d.py --fcs 1024 128 2 --Dropout 0.3 --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/tuning/model/index_7_epoch18.pth --cfg 'A3' --BatchNorm True --frac_ep 0.2 --nhit_c14 50 --batch 128 --train_file $workpath/dataset/add_1d/train.txt --valid_file $workpath/dataset/add_1d/valid.txt --check_train False --test_file $workpath/dataset/add_1d/test_iso_40kHz.txt --channel 2 --outFile $workpath/Pred/Cls_1d_test_iso_40kHz_ep18.h5
