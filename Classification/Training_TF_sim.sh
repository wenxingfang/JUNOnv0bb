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
#SBATCH --job-name=tf
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_train_TF_sim.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_train_TF_sim.err
  
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
export workpath=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls
### for test, 
export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_add_bkgs
#add shuffle,epoch46,train_loss=0.44818802355606663,valid_loss=0.45859026880147147, lr=0.000153001393648866
python $workpath/train_TF.py --train_file_bsize 100 --psencoding False --fcs 1024 128 2  --ps_features 14 --epochs 50 --lr 5e-4  --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/test_TF_bkgs_sim.pth
#--epoch47,train_loss=0.44581314476845646,valid_loss=0.4569906702833643, lr=0.0001371849865816552
#python $workpath/train_TF.py --train_file_bsize 100 --psencoding False --fcs 1024 128 2  --ps_features 14 --epochs 50 --lr 5e-4  --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/test_TF_bkgs_sim.pth
#--epoch43,train_loss=0.4498303631817688,valid_loss=0.4567419829369331, lr=0.00010948159389972083
#python $workpath/train_TF.py --psencoding False --fcs 1024 128 2  --ps_features 14 --epochs 50 --lr 5e-4  --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/sig_train.txt --valid_file_sig $datasetpath/sig_valid.txt --test_file_sig $datasetpath/sig_test.txt --train_file_bkg $datasetpath/bkgs_train.txt --valid_file_bkg $datasetpath/bkgs_valid.txt --test_file_bkg $datasetpath/bkgs_test.txt --out_name $workpath/model/test_TF_bkgs_sim.pth
### unsort, no ps encoding, more train data, res:epoch36,train_loss=0.4910599619627574,valid_loss=0.4967305327761972, lr=0.0002021892816208171
#export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train
#python $workpath/train_TF.py --psencoding False --fcs 1024 128 2  --ps_features 14 --epochs 50 --lr 5e-4  --batch 128 --scheduler 'OneCycleLR' --train_file_sig $datasetpath/train_bb0n.txt --valid_file_sig $datasetpath/valid_bb0n.txt --test_file_sig $datasetpath/test_bb0n.txt --train_file_bkg $datasetpath/train_e-.txt --valid_file_bkg $datasetpath/valid_e-.txt --test_file_bkg $datasetpath/test_e-.txt --out_name $workpath/model/TF_sim_moreTrain.pth
### unsort, no ps encoding, res:epoch47,train_loss=0.4638484974633386,valid_loss=0.4872591082844618, lr=1.1237642023971053e-05
#python $workpath/train_TF.py --psencoding False --fcs 1024 128 2  --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/test_e-.txt --out_name $workpath/model/TF_sim_early0p05.pth
