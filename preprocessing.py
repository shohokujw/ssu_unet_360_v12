import ctypes
from ctypes import *
import numpy as np
import os 

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p   = lambda x: x.ctypes.data_as(POINTER(c_int))

X         = 0
Y         = 1
Z         = 2


def get_Tv( path_dll, **params):
    #
    # with open(file_path, 'r') as file:
    #     config = yaml.safe_load(file)

    path_dll_tv = os.path.join(path_dll, 'libchambolleTV_3D_gpu.so')
    _dll_tv = ctypes.CDLL(path_dll_tv)

    #
    RunTV = _dll_tv.RunChambolleProxTV
    RunTV.argtypes = (POINTER(c_float), POINTER(c_float),
                      c_float, c_float,
                      c_int,
                      c_int, c_int, c_int)
    RunTV.restypes = c_void_p

    #
    OP_TV={}
    OP_TV['Tv'] = lambda output, input: \
        RunTV(
            c_float_p(output), c_float_p(input),
            params['lambda'], params['tau'],
            params['niter'],
            params['img_mat'][X], params['img_mat'][Y], params['img_mat'][Z])


    return OP_TV

def TV(src, OP_TV,**params):
    src = src.astype(np.float32).copy()
    dst = np.zeros((params['img_mat'][Z], params['img_mat'][Y], params['img_mat'][X]), dtype=np.float32)
    OP_TV['Tv'](dst, src)
    return dst.copy()




def apply_corner_triangles_3d(volume, triangle_size):
    """
    주어진 3D 배열의 각 꼭짓점에 직삼각형을 0으로 설정하는 함수.
    
    Parameters:
    volume (np.ndarray): 3D 배열 (z, x, y)
    triangle_size (int): 삼각형의 밑변과 높이 크기
    
    Returns:
    np.ndarray: 수정된 3D 배열
    """
    modified_volume = volume.copy()
    z_size, y_size, x_size = modified_volume.shape
    
    # 왼쪽 위 (front-top-left)
    for z in range(z_size):
        for i in range(triangle_size):
            for j in range(triangle_size - i):
                modified_volume[z, j, i] = 0
    
    # 오른쪽 위 (front-top-right)
    for z in range(z_size):
        for i in range(triangle_size):
            for j in range(triangle_size - i):
                modified_volume[z, j, y_size - i - 1] = 0

    # 왼쪽 아래 (front-bottom-left)
    for z in range(z_size):
        for i in range(triangle_size):
            for j in range(triangle_size - i):
                modified_volume[z, x_size - j - 1, i] = 0

    # 오른쪽 아래 (front-bottom-right)
    for z in range(z_size):
        for i in range(triangle_size):
            for j in range(triangle_size - i):
                modified_volume[z, x_size - j - 1, y_size - i - 1] = 0

    return modified_volume


def normalize(array, max_value,mean=1.0,std=0.5):
    
    array = (array/max_value)/std - mean
    
    return array

def denormalize(array, max_value, mean=1.0, std = 0.5):
    
    array = ((array + mean )*std)*max_value
    
    return array


def get_obj_height(prj, do_remove_air_img_idx, ud, OUTPUT_SIZE, params):
    profile = prj[do_remove_air_img_idx].mean(axis=0) 
    threshold = 0.001

    profile = profile[1:-1]  # 가장자리 효과 제거                        

    #plt.plot(profile)
    #plt.show()

    profile_reverse = profile[::-1]
    edge_candidates = np.where(profile_reverse >= threshold)[0]
    if len(edge_candidates) > 0:
        rightmost_edge_x = ud - (edge_candidates[0] + 1)  # +1 보정
    else:
        rightmost_edge_x = ud - 10

    SID = params['SID'][do_remove_air_img_idx]
    SOD = params['SOD'][do_remove_air_img_idx]
    ang_primary = params['ang_primary'][do_remove_air_img_idx]
    ang_offset = params['angle_offset'][do_remove_air_img_idx]
    angle_primary = (ang_primary - ang_offset)*np.pi/180
    magnitd = SID/SOD
    obj_height = rightmost_edge_x/(magnitd*abs(np.cos(angle_primary)))
    
    print(f"Rightmost edge x-position: {rightmost_edge_x}, Object Height: {int(obj_height)}")

    if obj_height < 10:
        obj_height = ud - 10

    obj_height = int(OUTPUT_SIZE * obj_height/ud)

    return obj_height