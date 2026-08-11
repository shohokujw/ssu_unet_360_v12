import os
import pickle
import time
import yaml

from op_ct_sstlabs import Mbir,FBP,get_params, Set_operation
from preprocessing import get_obj_height, apply_corner_triangles_3d

from glob import glob
import numpy as np

current_folder = os.path.dirname(os.path.realpath(__file__))
# System 파라미터 지정 파일위치
yaml_path = f"{current_folder}/include/params.yml"
# System 파라미터 불러오기
params = get_params(file_path=yaml_path, geo_path="")
OP_CT = Set_operation(so_file_dir=f"{current_folder}/include",**params)

model_name = params['model_name']
print(model_name)

OUTPUT_SIZE = 512


w_l = 0.7
w_h = 0.3


## 데이터 저장 위치 지정
dir_root     = "/mnt/c/Dropbox/work/SSTLabs/360CT/volume_data//prj"
dir_save_fbp = f"{current_folder}/datasets/{model_name}/fbp_lh_{int(10*w_l)}"

os.makedirs(dir_save_fbp, exist_ok=True)

##
lst_data = sorted(glob(os.path.join(dir_root, '*.pkl')))
# print(lst_data)
##

ud = params['Ud']  # 디텍터 픽셀수
do_remove_air_level = True
if ud == 768:
    do_remove_air_img_idx = 1 # 어떤 뷰의 이미지로 air level 날리는 높이를 구할지
else:
    do_remove_air_img_idx = 1 # 어떤 뷰의 이미지로 air level 날리는 높이를 구할지

if params['img_mat'][0] == 512:
    th = 18
else:
    th = 9

for idata in range(0, len(lst_data)):
    path_data = lst_data[idata]
    name_data = path_data.split('/')[-1].split("_")[-1][:-4]
    
    print(path_data)
    print(name_data)

    # pickle 파일에서 데이터를 불러오기
    with open(path_data, 'rb') as fid:  # 'rb'는 읽기 모드에서 바이너리 파일을 의미
        data_prj = pickle.load(fid)
    
    prj_l_all = data_prj['prj_l'].copy()
    prj_h_all = data_prj['prj_h'].copy()

    geo_path = f"{current_folder}/geo_param/{name_data}.yaml"
    params = get_params(file_path=yaml_path, geo_path=geo_path, view_9=True)
    OP_CT = Set_operation(so_file_dir=f"{current_folder}/include",**params)

    ang_idx_9view = params['ang_idx_9view']

    prj_l = prj_l_all[:,ang_idx_9view,:].copy()
    prj_h = prj_h_all[:,ang_idx_9view,:].copy()    

    nz = len(prj_l)
    
    params['img_mat'][2] = nz    
    

    if do_remove_air_level:
        obj_height = get_obj_height(prj_l, do_remove_air_img_idx, ud, OUTPUT_SIZE, params)    
    else:
        obj_height = OUTPUT_SIZE
    

    prj_lh = w_l*prj_l + w_h*prj_h

    # Recon using FBP
    start_time = time.monotonic()
    fbp_lh = FBP(prj=prj_lh,obj_height=obj_height,OP_CT=OP_CT,mode="sparse", **params)
    end_time = time.monotonic()
    print(f"{end_time - start_time:.5f} sec,\n")
    
    fbp_lh[fbp_lh<0] = 0
    

    fbp_lh = apply_corner_triangles_3d(fbp_lh, th)
    print(fbp_lh.min(), fbp_lh.max())
    
    data_fbp = {'fbp_lh': fbp_lh}

    # Save with pickle
    path_save = os.path.join(dir_save_fbp, f'FBP_{name_data}.pkl')
    with open(path_save, 'wb') as fid:
        pickle.dump(data_fbp, fid)

