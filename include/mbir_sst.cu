#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mbir_op.cuh"




DLLEXPORT void conjugate_gradient(
                        float *d_x, float *d_y_cg, float *d_ang_iso_center, 
                        float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                        float dbeta, int n_view, int n_dct, float Rd,
                        int *img_mat, float *voxel,
                        float sample, float dct_angle_pitch, 
                        int filter_mode, float d,
                        int n_thread,
                        float mu, float rho, int N,
                        int max_iter);


DLLEXPORT void b(float *d_b, float *d_y, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread,
                float mu, float rho, int N,
                float *d_U, float *d_V,
                float *d_DX, float *d_DY, float *d_DZ,
                float *d_BX, float *d_BY, float *d_BZ) ;

DLLEXPORT void A_cg(float *d_A_cg_x, float *d_x, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread,
                float mu, float rho, int N);

DLLEXPORT void Run_mbir(float *pout, float *pin, float *pinit, float *ang_iso_center, 
            float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
            float dbeta, int n_view, int n_dct, float Rd,
            int *img_mat, float *voxel,
            float sample, float dct_angle_pitch, 
            int filter_mode, float d,
            int n_thread,
            int iter_outer, int iter_inner,
            float alpha, float beta, float rho, float mu) ;




// pin : prj(y)
// pout : reconimage(x)
DLLEXPORT void Run_mbir(float *pout, float *pin, float *pinit, float *ang_iso_center, 
            float *angle_offset,float *det_tilt,float *SID, float *SOD, float full_SOD,
            float dbeta, int n_view, int n_dct, float Rd,
            int *img_mat, float *voxel,
            float sample, float dct_angle_pitch, 
            int filter_mode, float d,
            int n_thread,
            int iter_outer, int iter_inner,
            float alpha, float beta, float rho, float mu) {
 



    float *d_pin = 0;
    cudaMalloc(&d_pin, n_view*n_dct*img_mat[Z] *sizeof(float));
    cudaMemset(d_pin,0,sizeof(float)* n_view*n_dct*img_mat[Z]);
    cudaMemcpy(d_pin,pin,sizeof(float)* n_view*n_dct*img_mat[Z],cudaMemcpyHostToDevice);

    float *d_SOD = 0;
    cudaMalloc(&d_SOD,sizeof(float)*n_view);
    cudaMemset(d_SOD,0,sizeof(float)*n_view);
    cudaMemcpy(d_SOD,SOD,sizeof(float)*n_view,cudaMemcpyHostToDevice);

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

// 



    int N = img_mat[X]*img_mat[Y]*img_mat[Z];
    // 
    dim3    nBlockNum(n_thread);
    dim3    nGridNum(ceil(N)/(float)n_thread);

    float alpha_over_rho = alpha / rho ;
    float beta_over_mu = beta / mu ;

    printf("alpha : %f\n",alpha);
    printf("beta : %f\n",beta);
    printf("rho : %f\n",rho);
    printf("mu : %f\n",mu);
    printf("alpha_over_rho : %f\n",alpha_over_rho);
    printf("beta_over_mu : %f\n",beta_over_mu);

    float *d_x = 0;
    cudaMalloc(&d_x,sizeof(float)*N);
    cudaMemset(d_x,0,sizeof(float)*N);

    float *d_DX = 0;
    cudaMalloc(&d_DX,sizeof(float)*N);
    cudaMemset(d_DX,0,sizeof(float)*N);

    float *d_DY = 0;
    cudaMalloc(&d_DY,sizeof(float)*N);
    cudaMemset(d_DY,0,sizeof(float)*N);

    float *d_DZ = 0;
    cudaMalloc(&d_DZ,sizeof(float)*N);
    cudaMemset(d_DZ,0,sizeof(float)*N);

    float *d_BX = 0;
    cudaMalloc(&d_BX,sizeof(float)*N);
    cudaMemset(d_BX,0,sizeof(float)*N);

    float *d_BY = 0;
    cudaMalloc(&d_BY,sizeof(float)*N);
    cudaMemset(d_BY,0,sizeof(float)*N);

    float *d_BZ = 0;
    cudaMalloc(&d_BZ,sizeof(float)*N);
    cudaMemset(d_BZ,0,sizeof(float)*N);
    
    float *d_U = 0;
    cudaMalloc(&d_U,sizeof(float)*N);
    cudaMemset(d_U,0,sizeof(float)*N);

    float *d_V = 0;
    cudaMalloc(&d_V,sizeof(float)*N);
    cudaMemset(d_V,0,sizeof(float)*N);

    float *d_y_cg = 0;
    cudaMalloc(&d_y_cg,sizeof(float)*N);
    cudaMemset(d_y_cg,0,sizeof(float)*N);
    cudaMemcpy(d_y_cg,pinit,sizeof(float)*N,cudaMemcpyHostToDevice);

    float *d_Dx_x = 0;
    cudaMalloc(&d_Dx_x,sizeof(float)*N);
    cudaMemset(d_Dx_x,0,sizeof(float)*N);

    float *d_Dy_x = 0;
    cudaMalloc(&d_Dy_x,sizeof(float)*N);
    cudaMemset(d_Dy_x,0,sizeof(float)*N);

    float *d_Dz_x = 0;
    cudaMalloc(&d_Dz_x,sizeof(float)*N);
    cudaMemset(d_Dz_x,0,sizeof(float)*N);


    float *d_Dx_x_BX_sum = 0;
    cudaMalloc(&d_Dx_x_BX_sum,sizeof(float)*N);
    cudaMemset(d_Dx_x_BX_sum,0,sizeof(float)*N);

    float *d_Dy_x_BY_sum = 0;
    cudaMalloc(&d_Dy_x_BY_sum,sizeof(float)*N);
    cudaMemset(d_Dy_x_BY_sum,0,sizeof(float)*N);

    float *d_Dz_x_BZ_sum = 0;
    cudaMalloc(&d_Dz_x_BZ_sum,sizeof(float)*N);
    cudaMemset(d_Dz_x_BZ_sum,0,sizeof(float)*N);


    float *d_x_v_sum = 0;
    cudaMalloc(&d_x_v_sum,sizeof(float)*N);
    cudaMemset(d_x_v_sum,0,sizeof(float)*N);



    for (int i = 0; i < iter_outer; ++i) {
        
        //Ap = A_cg(p)
        b(d_y_cg, d_pin, d_ang_iso_center, 
                d_angle_offset,d_det_tilt,d_SID, d_SOD, full_SOD,
                 dbeta,  n_view,  n_dct,  Rd,
                 img_mat, voxel,
                 sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread,
                 mu,  rho,  N,
                 d_U, d_V,
                 d_DX, d_DY, d_DZ,
                 d_BX, d_BY, d_BZ);
    



        
        conjugate_gradient(
                        d_x, d_y_cg, d_ang_iso_center, 
                        d_angle_offset,d_det_tilt,d_SID, d_SOD, full_SOD,
                        dbeta, n_view, n_dct, Rd,
                        img_mat, voxel,
                        sample, dct_angle_pitch, 
                        filter_mode, d,
                        n_thread,
                        mu, rho, N,
                        iter_inner);
        



        runDx(d_x,d_Dx_x,img_mat[X],img_mat[Y],img_mat[Z]);
        runDy(d_x,d_Dy_x,img_mat[X],img_mat[Y],img_mat[Z]);
        runDz(d_x,d_Dz_x,img_mat[X],img_mat[Y],img_mat[Z]);

        vector_op_2<<<nGridNum,nBlockNum>>>(d_Dx_x,d_BX,d_Dx_x_BX_sum,N,1.0,1.0);
        vector_op_2<<<nGridNum,nBlockNum>>>(d_Dy_x,d_BY,d_Dy_x_BY_sum,N,1.0,1.0);
        vector_op_2<<<nGridNum,nBlockNum>>>(d_Dz_x,d_BZ,d_Dz_x_BZ_sum,N,1.0,1.0);
        vector_op_2<<<nGridNum,nBlockNum>>>(d_x,d_V,d_x_v_sum,N,1.0,1.0);





        soft_thresholding<<<nGridNum,nBlockNum>>>(d_Dx_x_BX_sum,d_DX,alpha_over_rho,N);
        soft_thresholding<<<nGridNum,nBlockNum>>>(d_Dy_x_BY_sum,d_DY,alpha_over_rho,N);
        soft_thresholding<<<nGridNum,nBlockNum>>>(d_Dz_x_BZ_sum,d_DZ,alpha_over_rho,N);
        soft_thresholding<<<nGridNum,nBlockNum>>>(d_x_v_sum,d_U,beta_over_mu,N);


        cudaMemset(d_Dx_x_BX_sum,0,sizeof(float)*N);
        cudaMemset(d_Dy_x_BY_sum,0,sizeof(float)*N);
        cudaMemset(d_Dz_x_BZ_sum,0,sizeof(float)*N);
        cudaMemset(d_x_v_sum,0,sizeof(float)*N);

        vector_op_3<<<nGridNum,nBlockNum>>>(d_DX,d_Dx_x,d_BX,d_BX,N,1.0,-1.0,-1.0);
        vector_op_3<<<nGridNum,nBlockNum>>>(d_DY,d_Dy_x,d_BY,d_BY,N,1.0,-1.0,-1.0);
        vector_op_3<<<nGridNum,nBlockNum>>>(d_DZ,d_Dz_x,d_BZ,d_BZ,N,1.0,-1.0,-1.0);
        vector_op_3<<<nGridNum,nBlockNum>>>(d_U,d_x,d_V,d_V,N,1.0,-1.0,-1.0);


        cudaMemset(d_Dx_x,0,sizeof(float)*N);
        cudaMemset(d_Dy_x,0,sizeof(float)*N);
        cudaMemset(d_Dz_x,0,sizeof(float)*N);

        // // loss 저장
        // d_loss[i] = F(d_x, n);  // F 함수 포인터 호출
    }


    cudaMemcpy(pout,d_x,sizeof(float)*N,cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y_cg);

    cudaFree(d_DX);
    cudaFree(d_DY);
    cudaFree(d_DZ);

    cudaFree(d_BX);
    cudaFree(d_BY);
    cudaFree(d_BZ);

    cudaFree(d_U);
    cudaFree(d_V);

    
    cudaFree(d_Dx_x);
    cudaFree(d_Dy_x);
    cudaFree(d_Dz_x);

    cudaFree(d_Dx_x_BX_sum);
    cudaFree(d_Dy_x_BY_sum);
    cudaFree(d_Dz_x_BZ_sum);
    cudaFree(d_x_v_sum);

    cudaFree(d_pin);
    cudaFree(d_SOD);
    cudaFree(d_ang_iso_center);
    cudaFree(d_SID);
    cudaFree(d_angle_offset);
    cudaFree(d_det_tilt);
   



}





DLLEXPORT void A_cg(float *d_A_cg_x, float *d_x, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread,
                float mu, float rho, int N)
{
    

    float *d_Ax = 0;
    cudaMalloc(&d_Ax,sizeof(float)*n_dct*n_view*img_mat[Z]);
    cudaMemset(d_Ax,0,sizeof(float)*n_dct*n_view*img_mat[Z]);


    Projection(d_Ax,d_x,d_ang_iso_center, 
                 d_angle_offset, d_det_tilt,d_SID, d_SOD, full_SOD,
                 dbeta, n_view,  n_dct,  Rd,
                img_mat, voxel,
                sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread  );
    
    float *d_AT_Ax = 0;
    cudaMalloc(&d_AT_Ax,sizeof(float)*N);
    cudaMemset(d_AT_Ax,0,sizeof(float)*N);

    

    ProjectionT(d_AT_Ax,d_Ax,d_ang_iso_center, 
                 d_angle_offset, d_det_tilt,d_SID, d_SOD, full_SOD,
                 dbeta, n_view,  n_dct,  Rd,
                img_mat, voxel,
                sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread  );




    float *d_Dx_term = 0;
    cudaMalloc(&d_Dx_term,sizeof(float)*N);
    cudaMemset(d_Dx_term,0,sizeof(float)*N);

    Dxterm_A_cg(d_Dx_term,d_x,img_mat,N,rho);



    dim3    nBlockNum(n_thread);
    dim3    nGridNum(ceil(N)/(float)n_thread);

    vector_op_3<<<nGridNum,nBlockNum>>>(d_AT_Ax,d_x,d_Dx_term,d_A_cg_x,N,1.0,mu,1.0);



    cudaFree(d_Ax);
    cudaFree(d_AT_Ax);
    cudaFree(d_Dx_term);
}


DLLEXPORT void conjugate_gradient(
                        float *d_x, float *d_y_cg, float *d_ang_iso_center, 
                        float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                        float dbeta, int n_view, int n_dct, float Rd,
                        int *img_mat, float *voxel,
                        float sample, float dct_angle_pitch, 
                        int filter_mode, float d,
                        int n_thread,
                        float mu, float rho, int N,
                        int max_iter) {
    // 필요한 GPU 메모리 할당
 
    dim3    nBlockNum(n_thread);
    dim3    nGridNum(ceil(N)/(float)n_thread);


    float *d_A_cg_x = 0;
    cudaMalloc(&d_A_cg_x,sizeof(float)*N);
    cudaMemset(d_A_cg_x,0,sizeof(float)*N);



    // A_cg(x)
    A_cg(d_A_cg_x, d_x, d_ang_iso_center, 
                d_angle_offset,d_det_tilt,d_SID, d_SOD, full_SOD, 
                 dbeta,  n_view,  n_dct, Rd,
                img_mat, voxel,
                 sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread,
                 mu,  rho,  N);
    



    float *d_r = 0;
    cudaMalloc(&d_r,sizeof(float)*N);
    cudaMemset(d_r,0,sizeof(float)*N);

    // r = y_cg - A_cg(x)
    vector_op_2<<<nGridNum,nBlockNum>>>(d_y_cg,d_A_cg_x,d_r,N,1.0,-1.0);
    
    // y_cg 초기화화
    cudaMemset(d_y_cg,0,sizeof(float)*N);

    // p = np.copy(r)
    float *d_p = 0;
    cudaMalloc(&d_p,sizeof(float)*N);
    cudaMemset(d_p,0,sizeof(float)*N);
    cudaMemcpy(d_p,d_r,sizeof(float)*N,cudaMemcpyDeviceToDevice);



    float rs_old = 0;
    float *d_rs_old = 0;
    cudaMalloc(&d_rs_old, sizeof(float));
    cudaMemset(d_rs_old,0,sizeof(float));

    float rs_new = 0;
    float *d_rs_new = 0;
    cudaMalloc(&d_rs_new, sizeof(float));
    cudaMemset(d_rs_new,0,sizeof(float));



    // rs_old = np.matmul(r.reshape(1,-1),r.reshape(-1,1))
    dot_product<<<nGridNum,nBlockNum>>>(d_r, d_r, d_rs_old, N);
    cudaMemcpy(&rs_old,d_rs_old,sizeof(float),cudaMemcpyDeviceToHost);




    // printf("%f\n",rs_old);
    // printf("%f\n",d_rs_old);
    
    float *d_Ap = 0;
    cudaMalloc(&d_Ap,sizeof(float)*N);
    cudaMemset(d_Ap,0,sizeof(float)*N);

    float alpha = 0;


    float pAp = 0;
    float *d_pAp = 0;
    cudaMalloc(&d_pAp,sizeof(float));
    cudaMemset(d_pAp,0,sizeof(float));

    float beta = 0;

    for (int i = 0; i < max_iter; ++i) {
        
        //Ap = A_cg(p)
        A_cg(d_Ap, d_p, d_ang_iso_center, 
                d_angle_offset,d_det_tilt,d_SID, d_SOD, full_SOD,
                 dbeta,  n_view,  n_dct, Rd,
                img_mat, voxel,
                 sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread,
                 mu,  rho,  N);
    



        // alpha = rs_old / (p.T * Ap)
        dot_product<<<nGridNum,nBlockNum>>>(d_p, d_Ap, d_pAp, N);
        cudaMemcpy(&pAp,d_pAp,sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemset(d_pAp,0,sizeof(float));
        
        alpha = rs_old/pAp;

        // x = x + alpha * p
        vector_op_2<<<nGridNum,nBlockNum>>>(d_x,d_p,d_x,N,1.0,alpha);

        // r = r - alpha * Ap
        vector_op_2<<<nGridNum,nBlockNum>>>(d_r,d_Ap,d_r,N,1.0,-alpha);

        // rs_new = r.T * r
        dot_product<<<nGridNum,nBlockNum>>>(d_r, d_r, d_rs_new, N);
        cudaMemcpy(&rs_new,d_rs_new,sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemset(d_rs_new,0,sizeof(float));


        beta = rs_new / rs_old;
        vector_op_2<<<nGridNum,nBlockNum>>>(d_r,d_p,d_p,N,1.0,beta);


        rs_old = rs_new;

    }

    // y_cg 초기화?


    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    cudaFree(d_rs_old);
    cudaFree(d_rs_new);
    // cudaFree(d_alpha);
    cudaFree(d_pAp);
    cudaFree(d_A_cg_x);


    // 
    
}







// pin : y = A(x)
DLLEXPORT void b(float *d_b, float *d_y, float *d_ang_iso_center, 
                float *d_angle_offset,float *d_det_tilt,float *d_SID, float *d_SOD, float full_SOD,
                float dbeta, int n_view, int n_dct, float Rd,
                int *img_mat, float *voxel,
                float sample, float dct_angle_pitch, 
                int filter_mode, float d,
                int n_thread,
                float mu, float rho, int N,
                float *d_U, float *d_V,
                float *d_DX, float *d_DY, float *d_DZ,
                float *d_BX, float *d_BY, float *d_BZ) //DX,DY,DZ,BX,BY,BZ,U,V 추가 
{
    
    
    
    dim3    nBlockNum(n_thread);
    dim3    nGridNum(ceil(N)/(float)n_thread);


    float *d_AT_y = 0;
    cudaMalloc(&d_AT_y,sizeof(float)*N);
    cudaMemset(d_AT_y,0,sizeof(float)*N);



    ProjectionT(d_AT_y,d_y,d_ang_iso_center, 
                 d_angle_offset, d_det_tilt,d_SID, d_SOD, full_SOD,
                 dbeta, n_view,  n_dct,  Rd,
                img_mat, voxel,
                sample,  dct_angle_pitch, 
                 filter_mode,  d,
                 n_thread  );



    float *d_sub = 0;
    cudaMalloc(&d_sub,sizeof(float)*N);
    cudaMemset(d_sub,0,sizeof(float)*N);



    vector_op_2<<<nGridNum,nBlockNum>>>(d_U,d_V,d_sub,N,1.0,-1.0);



    float *d_Dxterm = 0;
    cudaMalloc(&d_Dxterm,sizeof(float)*N);
    cudaMemset(d_Dxterm,0,sizeof(float)*N);



    Dxterm_b(d_Dxterm, d_DX,d_DY,d_DZ,d_BX,d_BY,d_BZ, img_mat, N,rho);



    vector_op_3<<<nGridNum,nBlockNum>>>(d_AT_y,d_sub,d_Dxterm,d_b,N,1.0,mu,1.0);



    cudaFree(d_AT_y);
    cudaFree(d_sub);
    cudaFree(d_Dxterm);
}


