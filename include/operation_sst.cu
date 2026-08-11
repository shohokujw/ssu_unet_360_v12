#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "util.cuh"





DLLEXPORT void Run_projection(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread  );

DLLEXPORT void Run_filteration(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread );

DLLEXPORT void Run_backprojection(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int air_pos, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread );

DLLEXPORT void Run_projectionT(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread );


DLLEXPORT void Run_FBP(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, int air_pos, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread);



int main()
{
    int a=1;
    printf("%d\n",a);
    return 0;
}




DLLEXPORT void Run_projection(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
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
    copyParams_img.srcPtr = make_cudaPitchedPtr((void*)pin, volumeSize_img.width * sizeof(float), volumeSize_img.width, volumeSize_img.height);
    copyParams_img.dstArray = cu_3DArray_img;
    copyParams_img.extent = volumeSize_img;
    copyParams_img.kind = cudaMemcpyHostToDevice;
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
    
    
    


    float *d_pout = 0;
    cudaMalloc(&d_pout, n_view*n_dct*img_mat[Z] *sizeof(float));
    cudaMemset(d_pout,0,sizeof(float)* n_view*n_dct*img_mat[Z]);
    

    dim3    nBlockNum(n_thread);
    dim3    nGridNum_img(ceil((n_view*n_dct*img_mat[Z])/(float)n_thread));



    float *d_ang_iso_center = 0;
    cudaMalloc(&d_ang_iso_center,sizeof(float)*n_view);
    cudaMemset(d_ang_iso_center,0,sizeof(float)*n_view);

    cudaMemcpy(d_ang_iso_center,ang_iso_center,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    float *d_SOD = 0;
    cudaMalloc(&d_SOD,sizeof(float)*n_view);
    cudaMemset(d_SOD,0,sizeof(float)*n_view);

    cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_SID = 0;
    cudaMalloc(&d_SID,sizeof(float)*n_view);
    cudaMemset(d_SID,0,sizeof(float)*n_view);

    cudaMemcpy(d_SID,SID,sizeof(float)*n_view,cudaMemcpyHostToDevice);



    float *d_angle_offset = 0;
    cudaMalloc(&d_angle_offset,sizeof(float)*n_view);
    cudaMemset(d_angle_offset,0,sizeof(float)*n_view);

    cudaMemcpy(d_angle_offset,angle_offset,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_det_tilt = 0;
    cudaMalloc(&d_det_tilt,sizeof(float)*n_view);
    cudaMemset(d_det_tilt,0,sizeof(float)*n_view);

    cudaMemcpy(d_det_tilt,det_tilt,sizeof(float)*n_view,cudaMemcpyHostToDevice);


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


    cudaMemcpy(pout, d_pout, n_view*n_dct*img_mat[Z]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_pout);
    cudaDestroyTextureObject(tex_img);
    cudaFreeArray(cu_3DArray_img); 
    cudaFree(d_ang_iso_center);
    cudaFree(d_angle_offset);
    cudaFree(d_det_tilt);
    cudaFree(d_SID);
    cudaFree(d_SOD);

}












DLLEXPORT void Run_filteration(float *pout, float *pin, float *ang_iso_center, 
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













DLLEXPORT void Run_backprojection(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int air_pos, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread)
{


    
    float dct_unit_vector[2]  = {0.0f,1.0f};
    float center_of_circle[2] = {0.0f,-full_SOD};
    float half_angle = dct_angle_pitch * (n_dct-1)/2.0;

    cudaTextureObject_t tex;
    cudaArray_t   cu_3DArray   = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(n_dct, n_view, img_mat[Z]);

    cudaMalloc3DArray(&cu_3DArray, &channelDesc, volumeSize);


    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)pin, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = cu_3DArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
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

    
    float *d_pout = 0;
    cudaMalloc(&d_pout, img_mat[X]*img_mat[Y]*img_mat[Z] *sizeof(float));
    cudaMemset(d_pout,0,sizeof(float)*img_mat[X]*img_mat[Y]*img_mat[Z]);
    


    dim3    nBlockNum(n_thread);
    dim3    nGridNum(ceil((img_mat[X]*img_mat[Y]*img_mat[Z])/(float)n_thread));
    
    
    float *d_ang_iso_center = 0;
    cudaMalloc(&d_ang_iso_center,sizeof(float)*n_view);
    cudaMemset(d_ang_iso_center,0,sizeof(float)*n_view);

    cudaMemcpy(d_ang_iso_center,ang_iso_center,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    float *d_SOD = 0;
    cudaMalloc(&d_SOD,sizeof(float)*n_view);
    cudaMemset(d_SOD,0,sizeof(float)*n_view);

    cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_SID = 0;
    cudaMalloc(&d_SID,sizeof(float)*n_view);
    cudaMemset(d_SID,0,sizeof(float)*n_view);

    cudaMemcpy(d_SID,SID,sizeof(float)*n_view,cudaMemcpyHostToDevice);



    float *d_angle_offset = 0;
    cudaMalloc(&d_angle_offset,sizeof(float)*n_view);
    cudaMemset(d_angle_offset,0,sizeof(float)*n_view);

    cudaMemcpy(d_angle_offset,angle_offset,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_det_tilt = 0;
    cudaMalloc(&d_det_tilt,sizeof(float)*n_view);
    cudaMemset(d_det_tilt,0,sizeof(float)*n_view);

    cudaMemcpy(d_det_tilt,det_tilt,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    backprojection<<<nGridNum, nBlockNum>>>( tex, d_pout, 
                                                n_view, n_dct, Rd,
                                                img_mat[X],img_mat[Y],img_mat[Z],
                                                voxel[X],voxel[Y],voxel[Z],
                                                dct_angle_pitch, dbeta, half_angle, air_pos,
                                                dct_unit_vector[T],dct_unit_vector[S],
                                                center_of_circle[T],center_of_circle[S],
                                                d_ang_iso_center,d_angle_offset,d_det_tilt,
                                                d_SOD,d_SID
                                                 );
    cudaMemcpy(pout, d_pout, img_mat[X]*img_mat[Y]*img_mat[Z]*sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cu_3DArray); 
    cudaFree(d_pout);
    cudaFree(d_ang_iso_center);
    cudaFree(d_angle_offset);
    cudaFree(d_det_tilt);
    cudaFree(d_SID);
    cudaFree(d_SOD);




}












DLLEXPORT void Run_projectionT(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
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
    

    float *d_pin =0;
    cudaMalloc(&d_pin,sizeof(float)*n_dct*n_view*img_mat[Z]);
    cudaMemset(d_pin,0,sizeof(float)*n_dct*n_view*img_mat[Z]);
    cudaMemcpy(d_pin,pin,sizeof(float)*n_dct*n_view*img_mat[Z],cudaMemcpyHostToDevice);

    float *d_pin_wgt =0;
    cudaMalloc(&d_pin_wgt,sizeof(float)*n_dct*n_view*img_mat[Z]);
    cudaMemset(d_pin_wgt,0,sizeof(float)*n_dct*n_view*img_mat[Z]);

    float *d_wgt =0;
    cudaMalloc(&d_wgt,sizeof(float)*n_view*n_dct);
    cudaMemset(d_wgt,0,sizeof(float)*n_view*n_dct);


    float *d_SOD = 0;
    cudaMalloc(&d_SOD,sizeof(float)*n_view);
    cudaMemset(d_SOD,0,sizeof(float)*n_view);
    cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    //weight 생성   
    generate_wgt_eqa<<<nGridNum_plain,nBlockNum >>>(d_wgt,n_dct,n_view,dct_angle_pitch,d_SOD);

    multiply_wgt_T<<<nGridNum_total,nBlockNum>>>(d_pin,d_wgt,d_pin_wgt,img_mat[Z],n_view,n_dct);

    // cudaMemcpy(pout, d_pin_wgt, sizeof(float)*n_dct*n_view*img_mat[Z], cudaMemcpyDeviceToHost);

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

    
    float *d_pout = 0;
    cudaMalloc(&d_pout, img_mat[X]*img_mat[Y]*img_mat[Z] *sizeof(float));
    cudaMemset(d_pout,0,sizeof(float)*img_mat[X]*img_mat[Y]*img_mat[Z]);
    


    dim3    nGridNum_img(ceil((img_mat[X]*img_mat[Y]*img_mat[Z])/(float)n_thread));
    
    float *d_ang_iso_center = 0;
    cudaMalloc(&d_ang_iso_center,sizeof(float)*n_view);
    cudaMemset(d_ang_iso_center,0,sizeof(float)*n_view);
    cudaMemcpy(d_ang_iso_center,ang_iso_center,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    
    float *d_SID = 0;
    cudaMalloc(&d_SID,sizeof(float)*n_view);
    cudaMemset(d_SID,0,sizeof(float)*n_view);
    cudaMemcpy(d_SID,SID,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_angle_offset = 0;
    cudaMalloc(&d_angle_offset,sizeof(float)*n_view);
    cudaMemset(d_angle_offset,0,sizeof(float)*n_view);
    cudaMemcpy(d_angle_offset,angle_offset,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_det_tilt = 0;
    cudaMalloc(&d_det_tilt,sizeof(float)*n_view);
    cudaMemset(d_det_tilt,0,sizeof(float)*n_view);
    cudaMemcpy(d_det_tilt,det_tilt,sizeof(float)*n_view,cudaMemcpyHostToDevice);



    projectionT<<<nGridNum_img, nBlockNum>>>(  tex, d_pout, 
                                                n_view, n_dct, Rd,
                                                img_mat[X],img_mat[Y],img_mat[Z],
                                                voxel[X],voxel[Y],voxel[Z],
                                                dct_angle_pitch, dbeta, half_angle,
                                                dct_unit_vector[T],dct_unit_vector[S],
                                                center_of_circle[T],center_of_circle[S],
                                                d_ang_iso_center,d_angle_offset,d_det_tilt,
                                                d_SOD,d_SID );
    cudaMemcpy(pout, d_pout, img_mat[X]*img_mat[Y]*img_mat[Z]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cu_3DArray); 
    cudaFree(d_pout);
    cudaFree(d_ang_iso_center);
    cudaFree(d_angle_offset);
    cudaFree(d_det_tilt);
    cudaFree(d_SID);
    cudaFree(d_SOD);
    cudaFree(d_pin_wgt);
}











DLLEXPORT void Run_FBP(float *pout, float *pin, float *ang_iso_center, 
                float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, int air_pos, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread)
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
    // cudaFree(d_SOD);

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
    


    // float *d_pin = 0;
    // cudaMalloc(&d_pin, n_dct* n_view* img_mat[Z] *sizeof(float));
    // cudaMemset(d_pin,0,sizeof(float)*n_dct* n_view* img_mat[Z]);




    // cudaMemcpy(d_pin, (cufftReal *)d_pin_complex, sizeof(cufftReal)*n_dct* n_view* img_mat[Z], cudaMemcpyDeviceToHost);
    // cudaFree(d_pin_complex);
    cudaFree(d_pin_real);





    
    float dct_unit_vector[2]  = {0.0f,1.0f};
    float center_of_circle[2] = {0.0f,-full_SOD};
    float half_angle = dct_angle_pitch * (n_dct-1)/2.0;

    cudaTextureObject_t tex;
    cudaArray_t   cu_3DArray   = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent volumeSize = make_cudaExtent(n_dct, n_view, img_mat[Z]);

    cudaMalloc3DArray(&cu_3DArray, &channelDesc, volumeSize);


    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)d_pin_complex, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
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

    
    float *d_pout = 0;
    cudaMalloc(&d_pout, img_mat[X]*img_mat[Y]*img_mat[Z] *sizeof(float));
    cudaMemset(d_pout,0,sizeof(float)*img_mat[X]*img_mat[Y]*img_mat[Z]);
    


    dim3    nGridNum(ceil((img_mat[X]*img_mat[Y]*img_mat[Z])/(float)n_thread));
    
    
    float *d_ang_iso_center = 0;
    cudaMalloc(&d_ang_iso_center,sizeof(float)*n_view);
    cudaMemset(d_ang_iso_center,0,sizeof(float)*n_view);

    cudaMemcpy(d_ang_iso_center,ang_iso_center,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    // float *d_SOD = 0;
    // cudaMalloc(&d_SOD,sizeof(float)*n_view);
    // cudaMemset(d_SOD,0,sizeof(float)*n_view);

    // cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_SID = 0;
    cudaMalloc(&d_SID,sizeof(float)*n_view);
    cudaMemset(d_SID,0,sizeof(float)*n_view);

    cudaMemcpy(d_SID,SID,sizeof(float)*n_view,cudaMemcpyHostToDevice);



    float *d_angle_offset = 0;
    cudaMalloc(&d_angle_offset,sizeof(float)*n_view);
    cudaMemset(d_angle_offset,0,sizeof(float)*n_view);

    cudaMemcpy(d_angle_offset,angle_offset,sizeof(float)*n_view,cudaMemcpyHostToDevice);


    float *d_det_tilt = 0;
    cudaMalloc(&d_det_tilt,sizeof(float)*n_view);
    cudaMemset(d_det_tilt,0,sizeof(float)*n_view);

    cudaMemcpy(d_det_tilt,det_tilt,sizeof(float)*n_view,cudaMemcpyHostToDevice);

    backprojection<<<nGridNum, nBlockNum>>>( tex, d_pout, 
                                                n_view, n_dct, Rd,
                                                img_mat[X],img_mat[Y],img_mat[Z],
                                                voxel[X],voxel[Y],voxel[Z],
                                                dct_angle_pitch, dbeta, half_angle, air_pos,
                                                dct_unit_vector[T],dct_unit_vector[S],
                                                center_of_circle[T],center_of_circle[S],
                                                d_ang_iso_center,d_angle_offset,d_det_tilt,
                                                d_SOD,d_SID
                                                 );
    cudaMemcpy(pout, d_pout, img_mat[X]*img_mat[Y]*img_mat[Z]*sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cu_3DArray); 
    cudaFree(d_pout);
    cudaFree(d_ang_iso_center);
    cudaFree(d_angle_offset);
    cudaFree(d_det_tilt);
    cudaFree(d_SID);
    cudaFree(d_SOD);




}










