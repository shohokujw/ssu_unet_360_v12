import numpy as np

X         = 0
Y         = 1
Z         = 2


## Setting operation
def MBIR(prj, params):
    rec = np.zeros((params['img_mat'][Z], params['img_mat'][Y], params['img_mat'][X]), dtype=np.float32)
    params['MBIR'](rec, prj.copy())
    return rec.copy()

def Projection(img, params):
    # (view, z_slice, dct)
    prj = np.zeros((params['nNumView'], params['pnSizeDct'][X], params['pnSizeDct'][Y]), dtype=np.float32)
    params['Projection'](prj, img.copy())
    return prj.copy()

def BackProjection(prj, params):
    rec = np.zeros((params['img_mat'][Z], params['img_mat'][Y], params['img_mat'][X]), dtype=np.float32)
    params['BackProjection'](rec, prj.copy())
    return rec.copy()

def Filtration(prj, params):
    flt = np.zeros((params['nNumView'], params['pnSizeDct'][X], params['pnSizeDct'][Y]), dtype=np.float32)
    params['Filtration'](flt, prj.copy())
    return flt.copy()

def FBP(prj, params):
    flt = Filtration(prj, params)
    rec = BackProjection(flt, params)
    return rec.copy()