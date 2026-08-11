import pickle
import numpy as np
import os
from glob import glob

current_folder = os.path.dirname(os.path.realpath(__file__))

from op_ct_sstlabs import FBP, Set_operation,Mbir
from config_utils import load_ct_params
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--view', type=int,default='720')
parser.add_argument('--img_size', type=int,default='512')

args = parser.parse_args()
print(args)

# System 파라미터 불러오기 (model_profiles 자동 적용)
params = load_ct_params()
model_name = params['model_name']

path_save_fbp_full = f"{current_folder}/datasets/{model_name}/fbp_full"
os.makedirs(path_save_fbp_full, exist_ok=True)

w_l = params['w_l']
w_h = params['w_h']


params['full_view'] = args.view # 720 or 360
params['img_mat'][0]= args.img_size # 128 or 256 or 512
params['img_mat'][1]= args.img_size # 128 or 256 or 512

if params['img_mat'][0] == 256 :
    params['voxel'][0] = 2.0
    params['voxel'][1] = 2.0
    
elif params['img_mat'][0] == 512 :
    params['voxel'][0] = 1.0
    params['voxel'][1] = 1.0
    
elif params['img_mat'][0] == 128 :
    params['voxel'][0] = 4.0
    params['voxel'][1] = 4.0
    
OP_CT = Set_operation(so_file_dir=f"{current_folder}/include",**params)

prj_path = params['prj_path']
dir_root = f'{current_folder}/datasets/{model_name}/sino_set'

lst_data = sorted(glob(os.path.join(dir_root, '*')))
for path in lst_data:
    filename = os.path.basename(path)
    data_no = filename.split('.')[0].split("_")[1]
    print(data_no)

    path_save = os.path.join(path_save_fbp_full,f"FBPFull{params['img_mat'][0]}_{params['full_view']}view_{data_no}.pkl")
    if os.path.exists(path_save) : 
        continue
    
    # pickle 파일에서 데이터를 불러오기
    with open(path, 'rb') as fid:  # 'rb'는 읽기 모드에서 바이너리 파일을 의미
        data = pickle.load(fid)

    sino_lh = data['prj_full_lh']    

    prj_path_tmp = f'{prj_path}/P_{data_no}.pkl'
    # pickle 파일에서 데이터를 불러오기
    with open(prj_path_tmp, 'rb') as fid:  # 'rb'는 읽기 모드에서 바이너리 파일을 의미
        data = pickle.load(fid)
    prj_h = data['prj_h'].copy()
    prj_l = data['prj_l'].copy()

    prj_lh = w_l*prj_l + w_h*prj_h

    sino_lh_tp  = sino_lh.transpose((1,0,2)).copy()
    scaled_sino_lh = (sino_lh_tp - np.min(sino_lh_tp)) * np.max(prj_lh)/(np.max(sino_lh_tp)- np.min(sino_lh_tp))

    
    scaled_sino_lh_add =scaled_sino_lh.copy()    
    fbp_lh_add = FBP(prj=scaled_sino_lh_add,OP_CT=OP_CT,mode="full", **params)
    fbp_lh_add[fbp_lh_add<0] = 0

    data = {f'fbp_lh' : fbp_lh_add}
    
    with open(path_save, 'wb') as fid:
        pickle.dump(data, fid)