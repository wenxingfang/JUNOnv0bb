import h5py
import os
import numpy as np


def save_h5(outname,dh_dict):
    merged_file = h5py.File(outname, 'w')
    for dataset_name in dh_dict:
    #    print(dataset_name,dh_dict[dataset_name].shape)
        merged_file.create_dataset(dataset_name, data=dh_dict[dataset_name])
    merged_file.close()


if __name__ == '__main__':

    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_all_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_all_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_all_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_all_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_t0_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_t0_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_t0_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_t0_h5_merged'
    #m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    #m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/Bi214_points_m10_earlys0p05_1D_tts_dn_ens_h5_merged'
    m_input_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_dn_ens_h5'
    m_output_path = '/cefs/higgs/wxfang/JUNO/ForReco/cls/C10_points_m10_earlys0p05_1D_tts_dn_ens_h5_merged'
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
