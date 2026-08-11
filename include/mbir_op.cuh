#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "util.cuh"

/*
##################operation########################
*/
// define
DLLEXPORT __global__ void Dx(const float* src, float* dst, int width, int height, int depth) ;
DLLEXPORT __global__ void DxT(const float* src, float* dst, int width, int height, int depth) ;
DLLEXPORT __global__ void Dy(const float* src, float* dst, int width, int height, int depth) ;
DLLEXPORT __global__ void DyT(const float* src, float* dst, int width, int height, int depth) ;
DLLEXPORT __global__ void Dz(const float* src, float* dst, int width, int height, int depth) ;
DLLEXPORT __global__ void DzT(const float* src, float* dst, int width, int height, int depth) ;

// execute
DLLEXPORT void runDx(const float* d_src, float* d_dst, int width, int height, int depth);
DLLEXPORT void runDxT(const float* d_src, float* d_dst, int width, int height, int depth);
DLLEXPORT void runDy(const float* d_src, float* d_dst, int width, int height, int depth);
DLLEXPORT void runDyT(const float* d_src, float* d_dst, int width, int height, int depth);
DLLEXPORT void runDz(const float* d_src, float* d_dst, int width, int height, int depth);
DLLEXPORT void runDzT(const float* d_src, float* d_dst, int width, int height, int depth);

/*
##################matrix calculate########################
*/

// define
DLLEXPORT __global__ void vector_op_3(const float* a, const float* b,const float* c, float* result, int n, float coef_a, float coef_b, float coef_c);
DLLEXPORT __global__ void vector_op_2(const float* a, const float* b, float* result, int n, float coef_a, float coef_b) ;
DLLEXPORT __global__ void dot_product(const float* x, const float* y, float* result, int n);

/*
##################soft thresholding########################
*/
DLLEXPORT __global__ void soft_thresholding(const float* y, float* result, float lam, int size);

/*
##################A_cg########################
*/
// Dxterm
DLLEXPORT void Dxterm_A_cg(float *d_Dxterm, float *d_x, int *img_mat, int N,float rho);

/*
##################b########################
*/
DLLEXPORT void Dxterm_b(float *d_Dxterm, float *d_DX,float *d_DY,float *d_DZ,float *d_BX,float *d_BY,float *d_BZ, int *img_mat, int N,float rho);

DLLEXPORT void b(float *d_b, float *y, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread,
                float mu, float rho, int N,
                float *d_U, float *d_V,
                float *d_DX, float *d_DY, float *d_DZ,
                float *d_BX, float *d_BY, float *d_BZ) ;



DLLEXPORT void Projection(float *d_pout, float *d_pin, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread
                );

DLLEXPORT void ProjectionT(float *d_pout, float *d_pin, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread
                );


// 
DLLEXPORT void filteration(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread );





// CUDA 커널: Dx 연산
DLLEXPORT __global__ void Dx(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_next = z * width * height + y * width + ((x + 1) % width);  // cyclic boundary
        if (x == 0) {
            dst[idx] = src[idx] - src[z * width * height + y * width + (width - 1)];
        } else {
            dst[idx] = src[idx_next] - src[idx];
        }
    }
}

// CUDA 커널: DxT 연산
DLLEXPORT __global__ void DxT(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_prev = z * width * height + y * width + ((x - 1 + width) % width);  // cyclic boundary
        if (x == width - 1) {
            dst[idx] = src[idx] - src[z * width * height + y * width];
        } else {
            dst[idx] = src[idx] - src[idx_prev];
        }
    }
}

// CUDA 커널: Dy 연산
DLLEXPORT __global__ void Dy(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_next = z * width * height + ((y + 1) % height) * width + x;
        if (y == 0) {
            dst[idx] = src[idx] - src[z * width * height + (height - 1) * width + x];
        } else {
            dst[idx] = src[idx_next] - src[idx];
        }
    }
}

// CUDA 커널: DyT 연산
DLLEXPORT __global__ void DyT(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_prev = z * width * height + ((y - 1 + height) % height) * width + x;
        if (y == height - 1) {
            dst[idx] = src[idx] - src[z * width * height + x];
        } else {
            dst[idx] = src[idx] - src[idx_prev];
        }
    }
}

// CUDA 커널: Dz 연산
DLLEXPORT __global__ void Dz(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_next = ((z + 1) % depth) * width * height + y * width + x;
        if (z == 0) {
            dst[idx] = src[idx] - src[(depth - 1) * width * height + y * width + x];
        } else {
            dst[idx] = src[idx_next] - src[idx];
        }
    }
}

// CUDA 커널: DzT 연산
DLLEXPORT __global__ void DzT(const float* src, float* dst, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int idx = z * width * height + y * width + x;
        int idx_prev = ((z - 1 + depth) % depth) * width * height + y * width + x;
        if (z == depth - 1) {
            dst[idx] = src[idx] - src[y * width + x];
        } else {
            dst[idx] = src[idx] - src[idx_prev];
        }
    }
}



DLLEXPORT __global__ void vector_op_3(const float* a, const float* b,const float* c, float* result, int n, float coef_a, float coef_b, float coef_c) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (coef_a * a[idx])  + (coef_b * b[idx]) + (coef_c * c[idx]);
    }
}

DLLEXPORT __global__ void vector_op_2(const float* a, const float* b, float* result, int n, float coef_a, float coef_b) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        result[idx] = (coef_a * a[idx])  + (coef_b * b[idx]);
    }
}



DLLEXPORT __global__ void soft_thresholding(const float* y, float* result, float lam, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float abs_y = fabs(y[i]);
        result[i] = copysignf(fmaxf(abs_y - lam, 0.0), y[i]);
    }
}



// CUDA 호출을 위한 호스트 함수
DLLEXPORT void runDx(const float* d_src, float* d_dst, int width, int height, int depth) {

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    Dx<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}

DLLEXPORT void runDy(const float* d_src, float* d_dst, int width, int height, int depth) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    Dy<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}

DLLEXPORT void runDz(const float* d_src, float* d_dst, int width, int height, int depth) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    Dz<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}

DLLEXPORT void runDxT(const float* d_src, float* d_dst, int width, int height, int depth) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    DxT<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}

DLLEXPORT void runDyT(const float* d_src, float* d_dst, int width, int height, int depth) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    DyT<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}

DLLEXPORT void runDzT(const float* d_src, float* d_dst, int width, int height, int depth) {
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16, depth);
    DzT<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, width, height, depth);
}





DLLEXPORT void Dxterm_A_cg(float *d_Dxterm, float *d_x, int *img_mat, int N,float rho){


    float *d_Dx_x = 0;
    cudaMalloc(&d_Dx_x,sizeof(float)*N);
    cudaMemset(d_Dx_x,0,sizeof(float)*N);

    float *d_Dy_x = 0;
    cudaMalloc(&d_Dy_x,sizeof(float)*N);
    cudaMemset(d_Dy_x,0,sizeof(float)*N);

    float *d_Dz_x = 0;
    cudaMalloc(&d_Dz_x,sizeof(float)*N);
    cudaMemset(d_Dz_x,0,sizeof(float)*N);


    runDx(d_x,d_Dx_x,img_mat[X],img_mat[Y],img_mat[Z]);
    runDy(d_x,d_Dy_x,img_mat[X],img_mat[Y],img_mat[Z]);
    runDz(d_x,d_Dz_x,img_mat[X],img_mat[Y],img_mat[Z]);



    float *d_DxT_Dx_x = 0;
    cudaMalloc(&d_DxT_Dx_x,sizeof(float)*N);
    cudaMemset(d_DxT_Dx_x,0,sizeof(float)*N);

    float *d_DyT_Dy_x = 0;
    cudaMalloc(&d_DyT_Dy_x,sizeof(float)*N);
    cudaMemset(d_DyT_Dy_x,0,sizeof(float)*N);

    float *d_DzT_Dz_x = 0;
    cudaMalloc(&d_DzT_Dz_x,sizeof(float)*N);
    cudaMemset(d_DzT_Dz_x,0,sizeof(float)*N);



    runDxT(d_Dx_x,d_DxT_Dx_x,img_mat[X],img_mat[Y],img_mat[Z]);
    runDyT(d_Dy_x,d_DyT_Dy_x,img_mat[X],img_mat[Y],img_mat[Z]);
    runDzT(d_Dz_x,d_DzT_Dz_x,img_mat[X],img_mat[Y],img_mat[Z]);





    dim3    nBlockNum(256);
    dim3    nGridNum(ceil(N)/(float)256);

    vector_op_3<<<nGridNum,nBlockNum>>>(d_DxT_Dx_x,d_DyT_Dy_x,d_DzT_Dz_x,d_Dxterm,N,rho,rho,rho);


    cudaFree(d_Dx_x);
    cudaFree(d_Dy_x);
    cudaFree(d_Dz_x);
    
    cudaFree(d_DxT_Dx_x);
    cudaFree(d_DyT_Dy_x);
    cudaFree(d_DzT_Dz_x);
    
    // cudaFree(d_sum);

}


DLLEXPORT void Dxterm_b(float *d_Dxterm, float *d_DX,float *d_DY,float *d_DZ,float *d_BX,float *d_BY,float *d_BZ, int *img_mat, int N,float rho){

 
    float *d_subX = 0;
    cudaMalloc(&d_subX,sizeof(float)*N);
    cudaMemset(d_subX,0,sizeof(float)*N);

    float *d_subY = 0;
    cudaMalloc(&d_subY,sizeof(float)*N);
    cudaMemset(d_subY,0,sizeof(float)*N);

    float *d_subZ = 0;
    cudaMalloc(&d_subZ,sizeof(float)*N);
    cudaMemset(d_subZ,0,sizeof(float)*N);

    
    dim3    nBlockNum(256);
    dim3    nGridNum(ceil(N)/(float)256);

    vector_op_2<<<nGridNum,nBlockNum>>>(d_DX,d_BX,d_subX,N,1.0,-1.0);
    vector_op_2<<<nGridNum,nBlockNum>>>(d_DY,d_BY,d_subY,N,1.0,-1.0);
    vector_op_2<<<nGridNum,nBlockNum>>>(d_DZ,d_BZ,d_subZ,N,1.0,-1.0);



    float *d_DxT_subX = 0;
    cudaMalloc(&d_DxT_subX,sizeof(float)*N);
    cudaMemset(d_DxT_subX,0,sizeof(float)*N);

    float *d_DyT_subY = 0;
    cudaMalloc(&d_DyT_subY,sizeof(float)*N);
    cudaMemset(d_DyT_subY,0,sizeof(float)*N);

    float *d_DzT_subZ = 0;
    cudaMalloc(&d_DzT_subZ,sizeof(float)*N);
    cudaMemset(d_DzT_subZ,0,sizeof(float)*N);


    runDxT(d_subX,d_DxT_subX,img_mat[X],img_mat[Y],img_mat[Z]);
    runDyT(d_subY,d_DyT_subY,img_mat[X],img_mat[Y],img_mat[Z]);
    runDzT(d_subZ,d_DzT_subZ,img_mat[X],img_mat[Y],img_mat[Z]);


    vector_op_3<<<nGridNum,nBlockNum>>>(d_DxT_subX,d_DyT_subY,d_DzT_subZ,d_Dxterm,N,rho,rho,rho);

    cudaFree(d_subX);
    cudaFree(d_subY);
    cudaFree(d_subZ);

    cudaFree(d_DxT_subX);
    cudaFree(d_DyT_subY);
    cudaFree(d_DzT_subZ);

}   







// 벡터 내적 커널
DLLEXPORT __global__ void dot_product(const float* x, const float* y, float* result, int n) {
    __shared__ float cache[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0;
    while (idx < n) {
        temp += x[idx] * y[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}







DLLEXPORT void Projection(float *d_pout, float *d_pin, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread
                )

{


    cudaTextureObject_t tex_img;
    cudaArray_t   cu_3DArray_img   = 0;
    cudaChannelFormatDesc channelDesc_img = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize_img = make_cudaExtent( img_mat[X],  img_mat[Y], img_mat[Z]);

    cudaMalloc3DArray(&cu_3DArray_img, &channelDesc_img, volumeSize_img);


    cudaMemcpy3DParms copyParams_img = {0};
    copyParams_img.srcPtr = make_cudaPitchedPtr((void*)d_pin, volumeSize_img.width * sizeof(float), volumeSize_img.width, volumeSize_img.height);
    copyParams_img.dstArray = cu_3DArray_img;
    copyParams_img.extent = volumeSize_img;
    copyParams_img.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams_img);


    cudaResourceDesc resDesc_img;
    memset(&resDesc_img, 0, sizeof(resDesc_img));
    resDesc_img.resType = cudaResourceTypeArray; //resource type설정 --> array로 설정
    resDesc_img.res.array.array = cu_3DArray_img; 

    cudaTextureDesc texDesc_img;
    memset(&texDesc_img, 0, sizeof(texDesc_img));
    texDesc_img.addressMode[0] = cudaAddressModeBorder;
    texDesc_img.addressMode[1] = cudaAddressModeBorder;
    texDesc_img.addressMode[2] = cudaAddressModeBorder;
    texDesc_img.filterMode = cudaFilterModeLinear;
    texDesc_img.readMode = cudaReadModeElementType;
    texDesc_img.normalizedCoords = 0;

    cudaCreateTextureObject(&tex_img, &resDesc_img, &texDesc_img, NULL);
    
    

    dim3    nBlockNum(n_thread);
    dim3    nGridNum_img(ceil((n_view*n_dct*img_mat[Z])/(float)n_thread));


    float dct_unit_vector[2]  = {0.0f,1.0f};
    float center_of_circle[2] = {0.0f,-full_SOD};


    projection<<<nGridNum_img,nBlockNum>>>( tex_img, d_pout, 
                                        n_view, n_dct, Rd,
                                        img_mat[X],img_mat[Y],img_mat[Z],
                                        voxel[X],voxel[Y],voxel[Z],
                                        dct_angle_pitch, sample, 
                                        dct_unit_vector[T], dct_unit_vector[S],
                                        center_of_circle[T], center_of_circle[S],
                                        d_ang_iso_center,d_angle_offset,d_det_tilt,
                                        d_SOD, d_SID);


    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(cu_3DArray_img); 


}




DLLEXPORT void ProjectionT(float *d_pout, float *d_pin, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread
                )
{


    float dct_unit_vector[2]  = {0.0f,1.0f};
    float center_of_circle[2] = {0.0f,-full_SOD};
    float half_angle = dct_angle_pitch * (n_dct-1)/2;


    dim3    nBlockNum(n_thread);
    dim3    nGridNum_plain(ceil((n_view*n_dct)/(float)n_thread));
    dim3    nGridNum_total(ceil((n_dct*n_view*img_mat[Z])/(float)n_thread));
    



    float *d_wgt =0;
    cudaMalloc(&d_wgt,sizeof(float)*n_view*n_dct);
    cudaMemset(d_wgt,0,sizeof(float)*n_view*n_dct);


    float *d_pin_wgt =0;
    cudaMalloc(&d_pin_wgt,sizeof(float)*n_dct*n_view*img_mat[Z]);
    cudaMemset(d_pin_wgt,0,sizeof(float)*n_dct*n_view*img_mat[Z]);


    //weight 생성   
    generate_wgt_eqa<<<nGridNum_plain,nBlockNum >>>(d_wgt,n_dct,n_view,dct_angle_pitch,d_SOD);

    multiply_wgt_T<<<nGridNum_total,nBlockNum>>>(d_pin,d_wgt,d_pin_wgt,img_mat[Z],n_view,n_dct);

    cudaFree(d_wgt);

    cudaTextureObject_t tex;
    cudaArray_t   cu_3DArray   = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(n_dct, n_view, img_mat[Z]);

    cudaMalloc3DArray(&cu_3DArray, &channelDesc, volumeSize);


    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)d_pin_wgt, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = cu_3DArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);


    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; //resource type설정 --> array로 설정
    resDesc.res.array.array = cu_3DArray; 

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);



    dim3    nGridNum_img(ceil((img_mat[X]*img_mat[Y]*img_mat[Z])/(float)n_thread));
    


    projectionT<<<nGridNum_img, nBlockNum>>>(  tex, d_pout, 
                                                n_view, n_dct, Rd,
                                                img_mat[X],img_mat[Y],img_mat[Z],
                                                voxel[X],voxel[Y],voxel[Z],
                                                dct_angle_pitch, dbeta, half_angle,
                                                dct_unit_vector[T],dct_unit_vector[S],
                                                center_of_circle[T],center_of_circle[S],
                                                d_ang_iso_center,d_angle_offset,d_det_tilt,
                                                d_SOD,d_SID );

    cudaDestroyTextureObject(tex);
    cudaFreeArray(cu_3DArray); 
    cudaFree(d_pin_wgt);
}








DLLEXPORT void filteration(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread )

{

    int ext_dct = int(pow(2.0, floor(log2(2.0 * n_dct) + 0.0)));
    int ext_out = ceil(ext_dct/2.0) + 1;


    dim3    nBlockNum(n_thread);
    dim3    nGridNum_ext(ceil((ext_dct)/(float)n_thread));
    dim3    nGridNum_dct(ceil((n_dct)/(float)n_thread));
    dim3    nGridNum_plain(ceil((n_view*n_dct)/(float)n_thread));
    dim3    nGridNum_view(ceil((n_view)/(float)n_thread));
    dim3    nGridNum_total(ceil((n_dct*n_view*img_mat[Z])/(float)n_thread));


    cufftComplex *d_pin_complex;
    cudaMalloc(&d_pin_complex,sizeof(cufftComplex)*ext_out*n_view*img_mat[Z]);
    cudaMemset(d_pin_complex,0,sizeof(cufftComplex)*ext_out*n_view*img_mat[Z]);

    cufftReal *d_pin_real; 
    cudaMalloc(&d_pin_real,sizeof(cufftComplex)*ext_dct*n_view*img_mat[Z]);
    cudaMemset(d_pin_real,0,sizeof(cufftComplex)*ext_dct*n_view*img_mat[Z]);

    cudaMemcpy((cufftReal *)d_pin_complex,pin,sizeof(cufftReal)*n_dct*n_view*img_mat[Z],cudaMemcpyHostToDevice);




    float *d_pcos =0;
    cudaMalloc(&d_pcos,sizeof(float)*n_view*n_dct);
    cudaMemset(d_pcos,0,sizeof(float)*n_view*n_dct);

    float *d_SOD = 0;
    cudaMalloc(&d_SOD,sizeof(float)*n_view);
    cudaMemset(d_SOD,0,sizeof(float)*n_view);

    cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);



    //weight 생성   
    generate_wgt_eqa<<<nGridNum_plain,nBlockNum >>>(d_pcos,n_dct,n_view,dct_angle_pitch,d_SOD);

    // weight 곱해주기
    multiply_wgt<<<nGridNum_total,nBlockNum>>>((cufftReal *)d_pin_complex,d_pcos,img_mat[Z],n_view,n_dct);

    cudaFree(d_pcos);
    cudaFree(d_SOD);

    zeropad3d<<<nGridNum_total, nBlockNum>>>(d_pin_real, (cufftReal *)d_pin_complex, ext_dct, n_dct,n_view, img_mat[Z]);


    cufftReal *d_filter;
    cudaMalloc(&d_filter,sizeof(cufftReal)*ext_dct);
    cudaMemset(d_filter,0,sizeof(cufftReal)*ext_dct);

    cufftComplex *d_filter_ft;
    cudaMalloc(&d_filter_ft,sizeof(cufftComplex)*(ext_out));
    cudaMemset(d_filter_ft, 0 , sizeof(cufftComplex)*(ext_out));

    // 필터 생성
    generate_filter_fft_eqa<<<nGridNum_ext,nBlockNum>>>(d_filter,dct_angle_pitch,ext_dct);
    

    // cufft plan 생성
    cufftHandle filt, view, inv;
    // cufftMakePlan1d(filt,ext_dct,CUFFT_C2R,1,sizeof())
    cufftPlan1d(&filt, ext_dct, CUFFT_R2C,1);
    cufftPlan1d(&view, ext_dct, CUFFT_R2C,1);
    cufftPlan1d(&inv, ext_dct, CUFFT_C2R,1);


    //h(t)(gpu) --> H(w)(gpu) 필터 변환
    cufftExecR2C(filt, d_filter, d_filter_ft);

    cufftDestroy(filt);

    magnitude<<<nGridNum_ext, nBlockNum>>>(d_filter, d_filter_ft, ext_out);

    cudaFree(d_filter_ft);
    
    filter_select<<<nGridNum_ext,nBlockNum>>>(d_filter,ext_out,filter_mode,d);

    cudaMemset(d_pin_complex, 0, sizeof(cufftComplex)*ext_out*n_view*img_mat[Z]);

    for (int i_slice = 0; i_slice<img_mat[Z]; i_slice++)
    {
        for (int i_view = 0 ; i_view < n_view ; i_view++)
        {
            cufftExecR2C(view, &d_pin_real[i_slice*ext_dct*n_view + ext_dct*i_view],&d_pin_complex[i_slice*ext_out*n_view + ext_out*i_view]);
            multiply_filter<<<nGridNum_ext,nBlockNum>>>(&d_pin_complex[i_slice*ext_out*n_view + ext_out*i_view],d_filter,ext_out,ext_dct);
            cudaMemset(&d_pin_real[i_slice*ext_dct*n_view + ext_dct*i_view], 0,sizeof(cufftReal)*ext_dct);
            cufftExecC2R(inv,&d_pin_complex[i_slice*ext_out*n_view + ext_out*i_view], &d_pin_real[i_slice*ext_dct*n_view + ext_dct*i_view]);
        }
    }


    cudaFree(d_filter);
    cufftDestroy(view);
    cufftDestroy(inv);
    
    
    
    crop3d<<<nGridNum_total, nBlockNum>>>((cufftReal *)d_pin_complex, d_pin_real, ext_dct, n_dct, n_view, img_mat[Z]);
    
    cudaMemcpy(pout, (cufftReal *)d_pin_complex, sizeof(cufftReal)*n_dct* n_view* img_mat[Z], cudaMemcpyDeviceToHost);
    cudaFree(d_pin_complex);
    cudaFree(d_pin_real);


}




