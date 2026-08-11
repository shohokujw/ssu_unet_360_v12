import os
import pickle
from op_ct_sstlabs import Set_operation,Projection
from config_utils import load_ct_params
import numpy as np

from glob import glob

current_folder = os.path.dirname(os.path.realpath(__file__))

# System 파라미터 불러오기 (model_profiles 자동 적용)
params = load_ct_params()
model_name = params['model_name']
params['full_view'] = 720

label_path = params['label_path']
label_name = params['label_name']
label_key = params['label_key']

w_l = params['w_l']
w_h = params['w_h']


# params['full_view'] = 720
OP_CT = Set_operation(so_file_dir=f"{current_folder}/include",**params)

dir_root     = label_path

dir_save_prj = f'{current_folder}/datasets/{model_name}/sino_set'
os.makedirs(dir_save_prj,exist_ok=True)

##
lst_data = sorted(glob(os.path.join(dir_root, '*.pkl')))

for idata in range(0, len(lst_data)):
    path_data = lst_data[idata]
    name_data = path_data.split('/')[-1][:-4]
    name_data = name_data.split('_')[-1]
    
    path_save = os.path.join(dir_save_prj, f"Pfull{params['full_view']}_{name_data}.pkl")
    if os.path.exists(path_save) : 
        continue
        
    # data_type = path_data.split('/')[-2]
    print(path_data)
    print(name_data)
    
    with open(path_data, 'rb') as fid:  # 'rb'는 읽기 모드에서 바이너리 파일을 의미
        data = pickle.load(fid)
        
    mbir_l = data['mbir_l'].copy()
    mbir_h = data['mbir_h'].copy()


    ## full view Projection
    prjfull_l = Projection(input=mbir_l,OP_CT=OP_CT,mode="full", **params)
    prjfull_h = Projection(input=mbir_h,OP_CT=OP_CT,mode="full", **params)

    prjfull_lh = w_l*prjfull_l + w_h*prjfull_h

    prjfull_lh_tp = prjfull_lh.transpose(1,0,2).copy()


    scaled_prj_lh = (prjfull_lh_tp - np.min(prjfull_lh_tp)) / (np.max(prjfull_lh_tp) - np.min(prjfull_lh_tp))

    data = {'prj_full_lh': scaled_prj_lh}

    with open(path_save, 'wb') as fid:
        pickle.dump(data, fid)
    

