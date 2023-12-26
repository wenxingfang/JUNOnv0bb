import h5py
import os
import numpy as np


def save_h5(outname,dh_dict):
    merged_file = h5py.File(outname, 'w')
    for dataset_name in dh_dict:
        merged_file.create_dataset(dataset_name, data=dh_dict[dataset_name])
    merged_file.close()


if __name__ == '__main__':


    input_path_bb0n = ''
    input_path_em   = ''
    input_path_C10  = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5_merged'
    input_path_Bi214= '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5_merged'


    count_evt(input_path_bb0n)

    UseN_bb0n = 500000
    UseN_em   = 500000
    UseN_C10  = 500000
    UseN_Bi214= 500000

    i_list = list(range(UseN_bb0n+UseN_em+UseN_C10+UseN_Bi214))
    random.shuffle(i_list)
    m_bat_evt = 1000

    m_dh_dict = {}
    index = 0

    for i in i_list:
        if i < UseN_bb0n:##select from bb0n
        elif i < UseN_bb0n+UseN_em:##select from e-
            k = i-UseN_bb0n
        elif i < UseN_bb0n+UseN_em+UseN_C10:##select from C10
            k = i-UseN_bb0n-UseN_em
        elif i < UseN_bb0n+UseN_em+UseN_C10+UseN_Bi214:##select from Bi214
            k = i-UseN_bb0n-UseN_em-UseN_C10
        if m_dh_dict['label'].shape[0]==m_bat_evt:
            save_h5(outname='%s/mixed_%d.h5'%(m_output_path,index),dh_dict=m_dh_dict)
            m_dh_dict = {}              
            index += 1
               
     

    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5_merged'
    m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5'
    m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5_merged'
    os.system('realpath %s/* > tmp.txt'%m_input_path)
    m_bat_evt = 1000
    if not os.path.exists(m_output_path):
        os.makedirs(m_output_path)
    m_list = []
    with open('tmp.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n','')
            m_list.append(line)
    m_dh_dict = {}
    index = 0
    for file_name in m_list:
        with h5py.File(file_name, 'r') as file:
            for dataset_name in file:
                dataset = file[dataset_name][:]
                if dataset_name not in m_dh_dict: m_dh_dict[dataset_name] = dataset
                else: 
                    if m_dh_dict[dataset_name].ndim == 3:
                        if m_dh_dict[dataset_name].shape[2] != dataset.shape[2]:
                            diff = abs(m_dh_dict[dataset_name].shape[2]-dataset.shape[2])
                            if m_dh_dict[dataset_name].shape[2] > dataset.shape[2]:
                                padding = np.zeros((dataset.shape[0],dataset.shape[1],diff), dtype=int)
                                dataset = np.concatenate((dataset,padding), axis=2)
                            else:
                                padding = np.zeros((m_dh_dict[dataset_name].shape[0],m_dh_dict[dataset_name].shape[1],diff), dtype=int)
                                m_dh_dict[dataset_name] = np.concatenate((m_dh_dict[dataset_name],padding), axis=2)
                    m_dh_dict[dataset_name] = np.concatenate((m_dh_dict[dataset_name], dataset), axis=0)
        if m_dh_dict['label'].shape[0]>m_bat_evt:
            save_h5(outname='%s/merge_%d.h5'%(m_output_path,index),dh_dict=m_dh_dict)
            m_dh_dict = {}              
            index += 1
