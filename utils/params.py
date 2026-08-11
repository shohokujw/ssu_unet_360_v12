import os
import numpy as np

import yaml

import ctypes
from ctypes import *

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))

X = 0
Y = 1
Z = 2

def get_params(yaml_path,mode,dir_dll='../lib'):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    n_dct = int(config['Ud'])
    config['n_dct'] = n_dct
    dct_angle_pitch = float(config['Pd'] /config['Rd'])
    config['dct_angle_pitch'] = dct_angle_pitch
    
    

    
    # Calibration parameter
    config['ang_primary']     = np.array((config['ang_primary']),dtype=np.float32)
    config['angle_offset']    = np.array(config['angle_offset'],dtype=np.float32)
    config['ang_iso_center']  = np.array((config['ang_primary']-90.0-config['angle_offset']),dtype=np.float32)*np.pi/180.0
    config['angle_offset']    *= np.pi/180.0
    config['det_tilt']        = np.array(config['det_tilt'],dtype=np.float32)*np.pi/180.0
    config['SID']             = np.array((config['SID']),dtype=np.float32)
    config['SOD']             = np.array((config['SOD']),dtype=np.float32)
    
    
    
    if mode == "sparse":
        config['nNumView'] = config['nSparseView']
        config['dbeta'] = 2.0 * np.abs(np.mean(np.diff(config['ang_iso_center'])))#half view여서 곱하기 2?
        print("mode is sparse")
    elif mode =="full":
        config['nNumView'] = config['nFullView']
        print("mode is full")
    else : 
        print("\n########Unknown Mode!##########\n")
        
        return
    
    config['img_mat'] = np.array(config['img_mat'],dtype=np.int32)
    config['voxel']   = np.array(config['voxel'],dtype=np.float32)
    sample            = float(min(config['voxel'][X],config['voxel'][Y])/2.0)
    config['sample']  = sample
    
    config.pop('Ud')
    config.pop('Pd')
    
  

    return config.copy()


def get_operations(dir_dll='../lib', device='cuda', params=None):
    dir_dll_mbir = os.path.join(os.path.dirname(__file__), dir_dll, 'librecon_admm_single_edit.so')
    _dll_mbir = ctypes.CDLL(dir_dll_mbir)
    
    '''
    dir_dll_projection = os.path.join(os.path.dirname(__file__), dir_dll, 'libprojection_gpu_stream_edit.so')
    _dll_projection = ctypes.CDLL(dir_dll_projection)
    
    dir_dll_backprojection = os.path.join(os.path.dirname(__file__), dir_dll, 'libbackprojection_gpu_stream_edit.so')
    _dll_backprojection = ctypes.CDLL(dir_dll_backprojection)
    
    dir_dll_filtration = os.path.join(os.path.dirname(__file__), dir_dll, 'libfiltration_gpu_stream_edit.so')
    _dll_filtration = ctypes.CDLL(dir_dll_filtration)
    '''

    ##
    RunMBIR = _dll_mbir.RunMBIR

    RunMBIR.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_float,
                        c_float, c_int, c_int, c_float, POINTER(c_int), POINTER(c_float),
                        c_float, c_float, c_float, c_float, c_float, c_float, 
                        c_int, c_int , c_int, c_int, c_int)
    RunMBIR.restypes = c_void_p
    
    '''
    RunProjection = _dll_projection.Projection
    
    RunProjection.argtypes = (POINTER(c_float), POINTER(c_float),POINTER(c_float),POINTER(c_int),
                             POINTER(c_float),POINTER(c_int),POINTER(c_float),POINTER(c_float),
                             c_float, c_int, c_float, c_float)
    
    RunProjection.restypes = c_void_p
    
    RunBackProjection = _dll_backprojection.BackProjection
    
    RunBackProjection.argtypes = (POINTER(c_float), POINTER(c_float),POINTER(c_float),POINTER(c_int),
                             POINTER(c_float),POINTER(c_int),POINTER(c_float),POINTER(c_float),
                             c_float, c_int, c_float, c_float)
    
    RunBackProjection.restypes = c_void_p
    
    RunFiltration = _dll_filtration.Filtration
    RunFiltration.argtypes = (POINTER(c_float), POINTER(c_float),POINTER(c_float),POINTER(c_int),
                             POINTER(c_float),POINTER(c_int),POINTER(c_float),POINTER(c_float),
                             c_float, c_int, c_float, c_float,
                             c_int, c_float) # nFilter, cut_off 추가
    RunFiltration.restypes = c_void_p
    '''
    


    ##
    params['c_float_p'] = c_float_p
    params['c_int_p'] = c_int_p
    

        
    params['MBIR']  = lambda output, input: \
        RunMBIR(params['c_float_p'](output), params['c_float_p'](input), params['c_float_p'](params['ang_iso_center']), params['c_float_p'](params['angle_offset']),
                params['c_float_p'](params['det_tilt']), params['c_float_p'](params['SID']), params['c_float_p'](params['SOD']), params['full_SOD'],
                params['dbeta'], params['nNumView'], params['n_dct'], params['Rd'], params['c_int_p'](params['img_mat']), params['c_float_p'](params['voxel']),
                params['sample'], params['dct_angle_pitch'], params['dTh1'], params['dTh2'], params['dMu1'], params['dMu2'],
                params['n_Iter'], params['n_CG_init'], params['n_CG'], params['filter_mode'], params['n_thread'])
    '''
    params['Projection']  = lambda output, input: \
        RunProjection(params['c_float_p'](output), params['c_float_p'](input), params['c_float_p'](params['pdStepImg']), params['c_int_p'](params['pnSizeImg']),
                      params['c_float_p'](params['pdStepDct']), params['c_int_p'](params['pnSizeDct']), params['c_float_p'](params['pdOffsetDct']), params['c_float_p'](params['pdRotDct']),
                      params['dStepView'], params['nNumView'], params['dDSO'], params['dDSD'])

    params['BackProjection']  = lambda output, input: \
        RunBackProjection(params['c_float_p'](output), params['c_float_p'](input), params['c_float_p'](params['pdStepImg']), params['c_int_p'](params['pnSizeImg']),
                      params['c_float_p'](params['pdStepDct']), params['c_int_p'](params['pnSizeDct']), params['c_float_p'](params['pdOffsetDct']), params['c_float_p'](params['pdRotDct']),
                      params['dStepView'], params['nNumView'], params['dDSO'], params['dDSD'])
        
    params['Filtration']  = lambda output, input: \
        RunFiltration(params['c_float_p'](output), params['c_float_p'](input), params['c_float_p'](params['pdStepImg']), params['c_int_p'](params['pnSizeImg']),
                      params['c_float_p'](params['pdStepDct']), params['c_int_p'](params['pnSizeDct']), params['c_float_p'](params['pdOffsetDct']), params['c_float_p'](params['pdRotDct']),
                      params['dStepView'], params['nNumView'], params['dDSO'], params['dDSD'],
                      params['nFilter'],params['cut_off'])
    '''

    params['device'] = device

    return params
