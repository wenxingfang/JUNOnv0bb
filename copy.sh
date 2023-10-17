#!/bin/bash
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Training/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Training/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/*.py /junofs/users/wxfang/MyGit/JUNOnv0bb/Classification/ 
cp /hpcfs/juno/junogpu/fangwx/FastSim/JUNO/nv0bb_cls/*.sh /junofs/users/wxfang/MyGit/JUNOnv0bb/Classification/ 
git add .
git commit -m "auto backup"
git push
