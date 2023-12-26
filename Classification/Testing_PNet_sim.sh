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
#SBATCH --job-name=test_pn_sim
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1
  
# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNet_sim.out
#SBATCH --error=//hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/log_test_PNet_sim.err
  
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
### add real tts, use earlys ratio = 0.05,res:
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_earlys0p05_tts_epoch49.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/test_e-.txt --outFile $workpath/Pred/cls_PNet_test_sim_tts.h5
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_earlys0p05_tts_epoch49.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_realTTS_points/train_e-.txt --outFile $workpath/Pred/cls_PNet_train_sim_tts.h5
## use earlys ratio = 0.05, dir cut,res:
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_dircut_epoch49.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/test_e-.txt --outFile $workpath/Pred/cls_PNet_test_sim_dircut.h5
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_dircut_epoch49.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_dircut_points/train_e-.txt --outFile $workpath/Pred/cls_PNet_train_sim_dircut.h5
### use earlys ratio = 0.05, more train,res:epoch35,train_loss=0.49075583145887824,valid_loss=0.5109533366799824, lr=0.00019477748320726883
export datasetpath=$workpath/dataset/detsim_m10ns_earlys0p05_points_add_1d/more_train
python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_moreTrain_epoch35.pth --train_file_sig $datasetpath/train_bb0n.txt --valid_file_sig $datasetpath/valid_bb0n.txt --test_file_sig $datasetpath/test_bb0n.txt --train_file_bkg $datasetpath/train_e-.txt --valid_file_bkg $datasetpath/valid_e-.txt --test_file_bkg $datasetpath/test_e-.txt --outFile $workpath/Pred/cls_PNet_test_moreTrain_sim.h5
python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --train_file_bsize 100 --batch 128 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_moreTrain_epoch35.pth --train_file_sig $datasetpath/train_bb0n.txt --valid_file_sig $datasetpath/valid_bb0n.txt --test_file_sig $datasetpath/train_bb0n.txt --train_file_bkg $datasetpath/train_e-.txt --valid_file_bkg $datasetpath/valid_e-.txt --test_file_bkg $datasetpath/train_e-.txt --outFile $workpath/Pred/cls_PNet_train_moreTrain_sim.h5
### use earlys ratio = 0.05,res:epoch50,train_loss=0.4812318780667785,valid_loss=0.5053049452548959, lr=1.476346442326146e-07
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_epoch40.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/test_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/test_e-.txt --outFile $workpath/Pred/cls_PNet_test_sim.h5
#python $workpath/train_PNet.py --ps_features 14 --epochs 50 --lr 5e-4  --batch 32 --scheduler 'OneCycleLR' --DoOptimization False  --DoTest True --Restore True  --restore_file $workpath/model/PNet_sim_early0p05_epoch40.pth --train_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --valid_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_bb0n.txt --test_file_sig $workpath/dataset/detsim_m10ns_earlys0p05_points/train_bb0n.txt --train_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --valid_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/valid_e-.txt --test_file_bkg $workpath/dataset/detsim_m10ns_earlys0p05_points/train_e-.txt --outFile $workpath/Pred/cls_PNet_train_sim.h5
