#!/bin/bash
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Training/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Training/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Classification/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Classification/ 
cp /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Analysis/ 
cp /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Analysis/ 
cp /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/checkPred/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Analysis/checkPred 
cp /junofs/users/wxfang/JUNO/nv0bb/J23.1.0-rc1/cls/Ana/checkPred/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Analysis/checkPred 
git add .
git commit -m "auto backup"
git push
