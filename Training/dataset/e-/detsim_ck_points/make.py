import os
import random
path = '/hpcfs/juno/junogpu/junoML/fangwx/nv0bb/e-_pointsForDir_sim_ck_1to3MeVh5/'
files = os.listdir(path)
with open('train.txt','w') as f:
    for file in files:
        file = file.replace('\n','')
        f.write('%s/%s\n'%(path,file))
