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
#SBATCH --job-name=testpnTF
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNetTF_sim.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNetTF_sim.err
  
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
python $workpath/train_PNetTF.py --nlayers 3 --nhead 16 --fcs_TF 1024 128 --fcs 512 256 128 2 --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNetTF_l3h16_more_sim_epoch33.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/test_e-.txt --outFile $workpath/Pred/cls_PNetTF_test_sim_moreTrain.h5
python $workpath/train_PNetTF.py --nlayers 3 --nhead 16 --fcs_TF 1024 128 --fcs 512 256 128 2 --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNetTF_l3h16_more_sim_epoch33.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train/train_e-.txt --outFile $workpath/Pred/cls_PNetTF_train_sim_moreTrain.h5
### use earlys ratio = 0.05,res:epoch44,train_loss=0.4519961969502,valid_loss=0.4700099072362833, lr=3.9270292245404256e-05
#python $workpath/train_PNetTF.py --fcs_TF 1024 128 --fcs 512 256 128 2 --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNetTF_sim_early0p05_epoch44.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/test_e-.txt --outFile $workpath/Pred/cls_PNetTF_test_sim.h5
#python $workpath/train_PNetTF.py --fcs_TF 1024 128 --fcs 512 256 128 2 --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNetTF_sim_early0p05_epoch44.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --outFile $workpath/Pred/cls_PNetTF_train_sim.h5
