import ctypes
from ctypes import *
import numpy as np
import yaml
import os 
c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p   = lambda x: x.ctypes.data_as(POINTER(c_int))

X         = 0
Y         = 1
Z         = 2

def find_closest_indices(angles, targets):
    indices = []
    for t in targets:
        if t < 0:
            t += 360
        closest_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - t))
        indices.append(closest_idx)
    return indices

def get_params(file_path, geo_path="", view_9=False):

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve model_profiles into top-level config
    if 'model_profiles' in config and 'model_name' in config:
        profile = config['model_profiles'].get(config['model_name'], {})
        for key, value in profile.items():
            config[key] = value
    
    if geo_path:
        with open(geo_path, 'r') as file:
            geo_param = yaml.safe_load(file)    
                
        config['SID'] = np.array(geo_param['SID'].copy()).astype(np.float32)
        config['SOD'] = np.array(geo_param['SOD'].copy()).astype(np.float32)
        config['ang_primary']    = np.array(geo_param['ang_primary'].copy()).astype(np.float32)
        config['angle_offset']   = np.array(geo_param['angle_offset'].copy()).astype(np.float32)
        config['ang_iso_center'] = np.array((config['ang_primary']-90.0-config['angle_offset']),dtype=np.float32)*np.pi/180.0
        config['angle_offset']   *= np.pi/180.0    
        config['det_tilt'] = np.array(geo_param['det_tilt'].copy()).astype(np.float32) 
    else:
        config['SID'] = np.array(config['SID'].copy()).astype(np.float32)
        config['SOD'] = np.array(config['SOD'].copy()).astype(np.float32)
        config['ang_primary']    = np.array(config['ang_primary'].copy()).astype(np.float32)
        config['angle_offset']   = np.array(config['angle_offset'].copy()).astype(np.float32)
        config['ang_iso_center'] = np.array((config['ang_primary']-90.0-config['angle_offset']),dtype=np.float32)*np.pi/180.0
        config['angle_offset']   *= np.pi/180.0    
        config['det_tilt'] = np.array(config['det_tilt'].copy()).astype(np.float32) 

    if view_9:
        ang_9view = config['ang']    
        ang_idx_9view = find_closest_indices(geo_param['ang_primary'], ang_9view)

        config['ang_9view']     = ang_9view
        config['ang_idx_9view'] = ang_idx_9view

        # Calibration parameter
        config['ang_primary']     = config['ang_primary'][ang_idx_9view]
        config['angle_offset']    = config['angle_offset'][ang_idx_9view]
        config['ang_iso_center']  = config['ang_iso_center'][ang_idx_9view]
        config['det_tilt']        = config['det_tilt'][ang_idx_9view]
        config['SID']             = config['SID'][ang_idx_9view]
        config['SOD']             = config['SOD'][ang_idx_9view]
    
    config['sparse_view'] = int(len(config['ang_iso_center']))

    # Detector parameter
    n_dct = int(config['Ud'])
    config['n_dct'] = n_dct
    dct_angle_pitch = float(config['Pd'] /config['Rd'])
    config['dct_angle_pitch'] = dct_angle_pitch

    # Operation parameter
    config['img_mat'] = np.array(config['img_mat'],dtype=np.int32)
    config['voxel']   = np.array(config['voxel'],dtype=np.float32)
    sample            = float(min(config['voxel'][X],config['voxel'][Y])/2.0)
    config['sample']  = sample
    
    config.pop('Pd')
    
    return config


def Set_operation(so_file_dir,**params):
    
    op_ct_cu = os.path.join(so_file_dir,"libop_ct_cu.so")
    op_mbir_cu = os.path.join(so_file_dir,"libop_mbir_cu.so")
    _op_cu_= ctypes.CDLL(op_ct_cu)
    _mbir_cu = ctypes.CDLL(op_mbir_cu)
    OP_CT={}
    
    operation_artypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float),
                         POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_float,
                         c_float,c_int, c_int, c_float,
                         POINTER(c_int), POINTER(c_float), 
                         c_float, c_float, 
                         c_int,c_float,
                         c_int)

    backprojection_artypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), 
                         POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_float,
                         c_float,c_int, c_int, c_int, c_float,   
                        POINTER(c_int), POINTER(c_float), 
                        c_float, c_float, 
                        c_int,c_float,
                        c_int)
    
    mbir_artypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), 
                         POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_float,
                         c_float,c_int, c_int, c_float,
                         POINTER(c_int), POINTER(c_float), 
                         c_float, c_float, 
                         c_int,c_float,
                         c_int,
                         c_int, c_int,
                         c_float, c_float, c_float, c_float)
    # 
   
    
    operation_args = (params['n_dct'], params['Rd'],
                      c_int_p(params['img_mat']),c_float_p(params['voxel']),
                      params['sample'],params['dct_angle_pitch'],
                      params['filter_mode'],params['d'],
                      params['n_thread']
                      )
    
    mbir_args = (params['n_dct'], params['Rd'],
                      c_int_p(params['img_mat']),c_float_p(params['voxel']),
                      params['sample'],params['dct_angle_pitch'],
                      params['filter_mode'],params['d'],
                      params['n_thread'],
                      params['iter_outer'],params['iter_inner'],
                      params['alpha'],params['beta'],params['rho'],params['mu']
                      )
    # 


    _prj_clang = _op_cu_.Run_projection

    _prj_clang.argtypes = operation_artypes
        
    _prj_clang.restype = c_void_p
    

    _flt_clang = _op_cu_.Run_filteration

    _flt_clang.argtypes = operation_artypes
    
    _flt_clang.restype = c_void_p
    
    
    _bpj_clang = _op_cu_.Run_backprojection

    _bpj_clang.argtypes = backprojection_artypes
    
    _bpj_clang.restype = c_void_p
    
    
    _prjT_clang = _op_cu_.Run_projectionT

    _prjT_clang.argtypes = operation_artypes
    
    _prjT_clang.restype = c_void_p
    
    # # 
    _mbir_clang = _mbir_cu.Run_mbir
    
    _mbir_clang.argtypes = mbir_artypes
    
    _mbir_clang.restype = c_void_p
    # # 
    
    
    OP_CT['Projection'] = _prj_clang, operation_args
    OP_CT['Filteration'] = _flt_clang, operation_args
    OP_CT['Backprojection'] = _bpj_clang, operation_args
    OP_CT['ProjectionT'] = _prjT_clang, operation_args
    OP_CT['Mbir'] = _mbir_clang, mbir_args
    
    return OP_CT


def Projection( input, mode, OP_CT,**params):
    
    projection,args = OP_CT['Projection']

    if (mode == "full"):
        # print("Checking Full view.....\n")
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID']).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD']).astype(np.float32)
       
        prj = np.zeros((params['img_mat'][Z],params['full_view'],params['n_dct']),dtype=np.float32)

        projection(c_float_p(prj), c_float_p(input), c_float_p(ang_iso_center),
                   c_float_p(angle_offset), c_float_p(det_tilt), c_float_p(SID), c_float_p(SOD),
                   float(params['full_SOD']),dbeta,params['full_view'],*args)

    elif (mode == "sparse"):
        # print("Checking Sparse view.....\n")
        dbeta = 2.0 * np.abs(np.mean(np.diff(params['ang_iso_center'])))#half view여서 곱하기 2?
        # dbeta = np.abs(np.mean(np.diff(params['ang_iso_center'])))
        # dbeta = float(np.abs(params['ang_iso_center'][0] - params['ang_iso_center'][1]))
        prj = np.zeros((params['img_mat'][Z],params['sparse_view'],params['n_dct']),dtype=np.float32)
        
        projection(c_float_p(prj), c_float_p(input), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   float(params['full_SOD']),dbeta,params['sparse_view'],*args)
        

    else:
        raise ValueError(f"Unknown mode: {mode}")


  
    return prj.copy()
    
    

def Filteration( input, mode, OP_CT,**params):
    

    filteration,args = OP_CT['Filteration']
    
    
    if (mode == "full"):
        
        flt = np.zeros((params['img_mat'][Z],params['full_view'],params['n_dct']),dtype=np.float32)
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID'] ).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD'] ).astype(np.float32)
        
        filteration(c_float_p(flt), c_float_p(input), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                   float(params['full_SOD']),dbeta,params['full_view'],*args)
        
        
    elif (mode == "sparse"):
        
        
        flt = np.zeros((params['img_mat'][Z],params['sparse_view'],params['n_dct']),dtype=np.float32)

        dbeta = 2.0 * np.abs(np.mean(np.diff(params['ang_iso_center'])))#half view여서 곱하기 2?
        # dbeta = np.abs(np.mean(np.diff(params['ang_iso_center'])))
        # dbeta = float(np.abs(params['ang_iso_center'][0] - params['ang_iso_center'][1]))
       
        
        filteration(c_float_p(flt), c_float_p(input), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   float(params['full_SOD']),dbeta,params['sparse_view'],*args)

    else:
        raise ValueError(f"Unknown mode: {mode}")

  
    return flt.copy()



def Backprojection(input, obj_height, mode, OP_CT,**params):
    

    backprojection,args = OP_CT['Backprojection']
    
    bpj = np.zeros((params['img_mat'][Z],params['img_mat'][Y],params['img_mat'][X]),dtype=np.float32)
    
    if (mode == "full"):
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID'] ).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD'] ).astype(np.float32)
        
        backprojection(c_float_p(bpj), c_float_p(input), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                   float(params['full_SOD']),dbeta,params['full_view'],obj_height,*args)

    elif (mode == "sparse"):
        dbeta = 2.0 * np.abs(np.mean(np.diff(params['ang_iso_center'])))#half view여서 곱하기 2?

        # dbeta = np.abs(np.mean(np.diff(params['ang_iso_center'])))
        # dbeta = float(np.abs(params['ang_iso_center'][0] - params['ang_iso_center'][1]))
        
        backprojection(c_float_p(bpj), c_float_p(input), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   float(params['full_SOD']),dbeta,params['sparse_view'],obj_height,*args)

    else:
        raise ValueError(f"Unknown mode: {mode}")


    
    return bpj.copy()

def ProjectionT( input, mode, OP_CT,**params):
    

    projectionT,args = OP_CT['ProjectionT']
    
    prjT = np.zeros((params['img_mat'][Z],params['img_mat'][Y],params['img_mat'][X]),dtype=np.float32)
    # projection_op(c_float_p(prj), c_float_p(input), *proj_args)
    
    if (mode == "full"):
        
        
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID'] ).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD'] ).astype(np.float32)
        
        projectionT(c_float_p(prjT), c_float_p(input), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                   float(params['full_SOD']),dbeta,params['full_view'],*args)

    elif (mode == "sparse"):
        
        dbeta = 2.0 * np.abs(np.mean(np.diff(params['ang_iso_center'])))#half view여서 곱하기 2?
        # dbeta = np.abs(np.mean(np.diff(params['ang_iso_center'])))
        # dbeta = float(np.abs(params['ang_iso_center'][0] - params['ang_iso_center'][1]))
        projectionT(c_float_p(prjT), c_float_p(input), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   float(params['full_SOD']),dbeta,params['sparse_view'],*args)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    
    return prjT.copy()


def FBP(prj, obj_height, mode,OP_CT,**params):
    
    filteration, args = OP_CT['Filteration']
    backprojection, _ = OP_CT['Backprojection']
    
    bpj = np.zeros((params['img_mat'][Z],params['img_mat'][Y],params['img_mat'][X]),dtype=np.float32)
    
    if (mode == "full"):
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID'] ).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD'] ).astype(np.float32)
        
        flt = np.zeros((params['img_mat'][Z],params['full_view'],params['n_dct']),dtype=np.float32) 
        
        # projection(c_float_p(prj), c_float_p(input), c_float_p(betas),params['full_view'],*args)
        filteration(c_float_p(flt), c_float_p(prj), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                   params['full_SOD'], dbeta, params['full_view'],*args)
        
        backprojection(c_float_p(bpj), c_float_p(flt), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                   params['full_SOD'], dbeta, params['full_view'], obj_height, *args)        

    elif (mode == "sparse"):        
        
        ang_iso_center_tmp = params['ang_iso_center']
        for i_view in range(len(ang_iso_center_tmp)):
            if ang_iso_center_tmp[i_view] + np.pi/2 > 1.5*np.pi:
                ang_iso_center_tmp[i_view] -= 2*np.pi

        dbeta = 2.0 * np.abs(np.mean(np.diff(ang_iso_center_tmp)))
        
        flt = np.zeros((params['img_mat'][Z],params['sparse_view'],params['n_dct']),dtype=np.float32)
        
        
        filteration(c_float_p(flt), c_float_p(prj), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   params['full_SOD'], dbeta, params['sparse_view'],*args)
        
        backprojection(c_float_p(bpj), c_float_p(flt), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   params['full_SOD'], dbeta, params['sparse_view'], obj_height, *args)
        
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return bpj.copy()




def Mbir( prj, init, mode, OP_CT,**params):
    

    mbir,args = OP_CT['Mbir']
    
    recon = np.zeros((params['img_mat'][Z],params['img_mat'][Y],params['img_mat'][X]),dtype=np.float32)
    
    if (mode == "full"):
        
        dbeta = float(2.0*np.pi/params['full_view'])
        ang_iso_center = (np.linspace(0,params['full_angle'],params['full_view'],endpoint=False)*np.pi/180.0).astype(np.float32)
        angle_offset = np.full(params['full_view'], params['full_view_angle_offset']*np.pi/180.0).astype(np.float32)
        det_tilt = np.full(params['full_view'], params['full_view_tilt_angle']*np.pi/180.0).astype(np.float32)
        SID = np.full(params['full_view'], params['full_SID'] ).astype(np.float32)
        SOD = np.full(params['full_view'], params['full_SOD'] ).astype(np.float32)
        
        mbir(c_float_p(recon), c_float_p(prj), c_float_p(init), c_float_p(ang_iso_center),
                   c_float_p(angle_offset),c_float_p(det_tilt),c_float_p(SID),c_float_p(SOD),
                  params['full_SOD'],dbeta,params['full_view'],*args)

    elif (mode == "sparse"):
        dbeta = 2.0 * np.abs(np.mean(np.diff(params['ang_iso_center'])))#half view여서 곱하기 2?
        # dbeta = np.abs(np.mean(np.diff(params['ang_iso_center'])))
        # dbeta = float(np.abs(params['ang_iso_center'][0] - params['ang_iso_center'][1]))
        
        mbir(c_float_p(recon), c_float_p(prj), c_float_p(init), c_float_p(params['ang_iso_center']),
                   c_float_p(params['angle_offset']),c_float_p(params['det_tilt']),c_float_p(params['SID']),c_float_p(params['SOD']),
                   params['full_SOD'],dbeta,params['sparse_view'],*args)

    else:
        raise ValueError(f"Unknown mode: {mode}")


    
    return recon.copy()

