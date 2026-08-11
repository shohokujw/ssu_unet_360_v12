#include <math.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

using namespace std;

// WINDOWS VERSION
//#define DLLEXPORT extern "C" __declspec(dllexport)

// LINUX VERSION
#define DLLEXPORT extern "C"

/*
 *
 */
 #define PI          3.14159265359f
 #define PI2         6.28318530718f
 #define PIsq        9.86960440109f

#define T           0
#define S           1

#define X           0
#define Y           1
#define Z           2

#define REAL        0
#define IMAG        1

#define min(X, Y)   (X < Y ? X : Y)
#define max(X, Y)   (X > Y ? X : Y)

enum FILTER_MODE { RAM_LAK, SHEPP_LOGAN, COSINE, HAMMING, HANN };

typedef struct {
    float x;
    float y;
} Point;




/*
 *
 public
 *
 */

DLLEXPORT __device__ float   id_to_cor(float id, float d, int n);

DLLEXPORT __device__ float   cor_to_id(float pos, float d, int n);


// projection

DLLEXPORT __global__ void projection( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float sample, 
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID);




// filteration

DLLEXPORT __global__ void multiply_wgt(float *pin,  float *pcos, int img_z, int n_view, int n_dct); 

DLLEXPORT __global__ void multiply_wgt_T(const float *pin,  const float *pcos, float *result, int img_z, int n_view, int n_dct);

DLLEXPORT __global__ void generate_wgt_eqa(float *pcos, int n_dct, int n_view, float dct_ang_pitch, float *SOD);

DLLEXPORT __global__ void generate_filter_fft_eqa(float *filter, float interval, int n) ;

DLLEXPORT __global__ void multiply_filter(cufftComplex *pview_ft, cufftReal *filter, int n, int ext);

DLLEXPORT __global__  void magnitude(cufftReal *pout, cufftComplex *pin, int n);

DLLEXPORT __global__  void filter_select(cufftReal *pdata,int n, int mode,float d);

DLLEXPORT __global__  void zeropad3d(cufftReal *pout, cufftReal *pin,
                              int ext_dct, int n_dct, int n_view, int img_mat_Z);
                              
__global__  void    crop3d(cufftReal *pout, cufftReal *pin,
                           int ext_dct, int n_dct, int n_view, int img_mat_Z);
// backprojection


DLLEXPORT __device__ float calculate_angle(float cx, float cy, float x1, float y1, float x2, float y2);

DLLEXPORT __device__ float get_angle(float cx, float cy, float x, float y);

DLLEXPORT __device__ Point find_intersection(float cx, float cy, float r, 
                      float x1, float y1, float x2, float y2, 
                      float angle_start, float angle_end) ;

DLLEXPORT __global__ void backprojection ( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float dbeta, float half_angle, int air_pos,
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID);


// projectionT
DLLEXPORT __global__ void projectionT( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float dbeta, float half_angle,
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID);

/*
 *
 public
 *
 */
DLLEXPORT __device__ float id_to_cor(float id, float d, int n) {
    float pos = (id - (n - 1.0f)/2.0f)*d;
    return pos ;
}



DLLEXPORT __device__ float cor_to_id(float pos, float d, int n) {
    float id = pos/d + (n - 1.0f)/2.0f;
    return id ;
}



/*
 *
 private
 *
 */


//################################fan_beam_equiangular##################################


//##############projection##############





DLLEXPORT __global__ void projection( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float sample, 
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID)
{   
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    

    int nXY = n_view * n_dct;

    if (idx >= nXY*img_mat_Z ) return;

    int i_dct = int((idx % nXY) % n_dct);
    int i_view = int((idx % nXY) / n_dct);
    int i_slice = int(idx / nXY);


    // float gamma = -id_to_cor(i_dct, dct_ang_pitch, n_dct);
    float gamma = id_to_cor(i_dct, dct_ang_pitch, n_dct);



    float src_ori_T = 0.0;
    float src_ori_S = -SOD[i_view]; 

    //beta rotate
    // src position
    float src_rot_T = cosf(ang_iso_center[i_view]) * src_ori_T  - sinf(ang_iso_center[i_view]) * src_ori_S ; 
    float src_rot_S = sinf(ang_iso_center[i_view]) * src_ori_T  + cosf(ang_iso_center[i_view]) * src_ori_S;
    
    float center_of_circle_rot_T = cosf(ang_iso_center[i_view]) * center_of_circle_T  - sinf(ang_iso_center[i_view]) * center_of_circle_S ; 
    float center_of_circle_rot_S = sinf(ang_iso_center[i_view]) * center_of_circle_T  + cosf(ang_iso_center[i_view]) * center_of_circle_S;
    
    float dct_rot_vector_T = cosf(ang_iso_center[i_view]) * dct_unit_vector_T  - sinf(ang_iso_center[i_view]) * dct_unit_vector_S ;
    float dct_rot_vector_S = sinf(ang_iso_center[i_view]) * dct_unit_vector_T  + cosf(ang_iso_center[i_view]) * dct_unit_vector_S;
    
    // offset rotate
    // float dct_offset_vector_T = cosf(angle_offset[i_view]) * (dct_rot_vector_T - src_rot_T)  - sinf(angle_offset[i_view]) * (dct_rot_vector_S - src_rot_S) + src_rot_T ;
    // float dct_offset_vector_S = sinf(angle_offset[i_view]) * (dct_rot_vector_T - src_rot_T)  + cosf(angle_offset[i_view]) * (dct_rot_vector_S - src_rot_S) + src_rot_S;
    
    
    float dct_offset_vector_T = cosf(angle_offset[i_view]) * (dct_rot_vector_T )  - sinf(angle_offset[i_view]) * (dct_rot_vector_S )  ;
    float dct_offset_vector_S = sinf(angle_offset[i_view]) * (dct_rot_vector_T )  + cosf(angle_offset[i_view]) * (dct_rot_vector_S ) ;




    // dct center position
    float dct_rot_T = src_rot_T + dct_offset_vector_T*SID[i_view]; 
    float dct_rot_S = src_rot_S + dct_offset_vector_S*SID[i_view]; 

    // each dct offset position
    float dct_offset_pos_T = cosf(gamma) * (dct_rot_T - center_of_circle_rot_T) - sinf(gamma) * (dct_rot_S- center_of_circle_rot_S) + center_of_circle_rot_T ;
    float dct_offset_pos_S = sinf(gamma) * (dct_rot_T - center_of_circle_rot_T) + cosf(gamma) * (dct_rot_S- center_of_circle_rot_S) + center_of_circle_rot_S;

    // each dct tilt position
    float dct_tilt_pos_T = cosf(det_tilt[i_view])*(dct_offset_pos_T - dct_rot_T) - sinf(det_tilt[i_view]) * (dct_offset_pos_S - dct_rot_S) + dct_rot_T;
    float dct_tilt_pos_S = sinf(det_tilt[i_view])*(dct_offset_pos_T - dct_rot_T) + cosf(det_tilt[i_view]) * (dct_offset_pos_S - dct_rot_S) + dct_rot_S;

    
    float dist = sqrtf(pow((dct_tilt_pos_T-src_rot_T),2) + pow((dct_tilt_pos_S-src_rot_S),2));


    float dir_ray_T = (dct_tilt_pos_T - src_rot_T)*sample/dist;
    float dir_ray_S = (dct_tilt_pos_S - src_rot_S)*sample/dist;

    int n_sample = int(ceil(dist/sample));


    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_offset_pos_S;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dist;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_rot_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_rot_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_rot_S;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_center_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_center_S;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_pos_tilted_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_pos_tilted_S;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct] =dct_pos_tilted_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_mdl*n_emt + i_emt] =dct_cor_T;
    // pout[i_slice * n_dct * n_view + i_view * n_dct + i_mdl*n_emt + i_emt] =dct_cor_S;


    for (int i_sample = 0; i_sample < n_sample; i_sample++) 
    {
        float ray_id_X = cor_to_id(src_rot_T + i_sample* dir_ray_T, voxel_X, img_mat_X);
        float ray_id_Y = cor_to_id(src_rot_S + i_sample* dir_ray_S, voxel_Y, img_mat_Y);

        pout[i_slice * n_dct * n_view + i_view * n_dct + i_dct]  += sample * tex3D<float>(tex,ray_id_X+0.5f, ray_id_Y+0.5f, i_slice+0.5f);
    }

    return;

}























/*

##############filteration###############

*/


DLLEXPORT __global__ void generate_wgt_eqa(float *pcos, int n_dct, int n_view, float dct_ang_pitch, float *SOD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nXY = n_dct*n_view;
    // if (idx < n_dct) 

    // {
    //     pcos[idx] = DSO * cosf(id_to_cor(idx, dct_ang_pitch, n_dct));
    // }
    // 여기가 잘못되었다 --> wgt는 gamma값 즉, detector 기준으로 들어가야한다.
    // 따라서 view, dct matrix 생성해야한다. 여기서
    if (idx >= nXY) return;
    int i_dct = int(idx % n_dct);
    int i_view = int(idx / n_dct);


    pcos[idx]  = SOD[i_view] * cosf(id_to_cor(i_dct, dct_ang_pitch, n_dct));
    //pcos[idx]  = cosf(id_to_cor(i_dct, dct_ang_pitch, n_dct));
    
    
}


DLLEXPORT __global__ void multiply_wgt(float *pin,  float *pcos, int img_z, int n_view, int n_dct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = img_z * n_view * n_dct;
    
    // if (idx < total_elements)

    if (idx >= total_elements) return;

    int nXY = n_view*n_dct;
    int i_dct = int((idx % nXY) % n_dct);
    int i_view = int((idx % nXY) / n_dct);

    pin[idx] *= pcos[i_view*n_dct+i_dct];


}



DLLEXPORT __global__ void multiply_wgt_T(const float *pin,  const float *pcos, float *result, int img_z, int n_view, int n_dct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = img_z * n_view * n_dct;

    // if (idx < total_elements)

   if (idx >= total_elements) return;

    int nXY = n_view*n_dct;
    int i_dct = int((idx % nXY) % n_dct);
    int i_view = int((idx % nXY) / n_dct);

    result[idx] = pin[idx] * pcos[i_view*n_dct+i_dct];


}





DLLEXPORT __global__  void zeropad3d(cufftReal *pout, cufftReal *pin,
                              int ext_dct, int n_dct, int n_view, int img_mat_Z)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    

    int nXY = n_view * n_dct;

    if (idx >= nXY*img_mat_Z ) return;

    int i_dct = int((idx % nXY) % n_dct);
    int i_view = int((idx % nXY) / n_dct);
    int i_slice = int(idx / nXY);

    pout[i_slice*n_view*ext_dct + i_view*ext_dct + i_dct] = pin[i_slice*n_view*n_dct + i_view*n_dct + i_dct];

    return ;
}


DLLEXPORT __global__ void generate_filter_fft_eqa(float *filter, float interval, int n) {
    
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    
    if (idx >= n) return;

    float sin_fn = 0;
    int i = idx-(n/2);

    if (i==0)
    {
        filter[idx] = 1.0f/(8.0f*interval*interval); 
    }
    else 
    {
        if (i%2)
        {
            sin_fn = sinf(i*interval);
            filter[idx] = -1.0f/(2.0f*((PI*sin_fn)*(PI*sin_fn)));
        }
        else
        {
            filter[idx] = 0;
        }
    }
        
        // filter[idx] *= interval;

}



DLLEXPORT __global__  void filter_select(cufftReal *pdata,int n, int mode,float d)
{
    int     idx   = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= n) return;

    float w = 2.0 * PI * idx / n;

    if (idx > 0) {
        switch (mode) {
        case RAM_LAK:
            break;
        case SHEPP_LOGAN:
            pdata[idx] *=  (sin(w/(2.0*d)) / (w/(2.0*d)));
            break;
        case COSINE:
            pdata[idx] *=  (cos(w/(2.0*d)));
            break;
        case HAMMING:
            pdata[idx] *=  (0.54 + 0.46 * cos(w/d));
            break;
        case HANN:
            pdata[idx] *= (1.0 + cos(w/d))/2.0;
            break;
        }
    }

    if (w > PI * d) {
        pdata[idx] = 0;
    }

    return ;
}



__global__  void    crop3d(cufftReal *pout, cufftReal *pin,
                           int ext_dct, int n_dct, int n_view, int img_mat_Z)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    

    int nXY = n_view * n_dct;

    if (idx >= nXY*img_mat_Z ) return;

    int i_dct = int((idx % nXY) % n_dct);
    int i_view = int((idx % nXY) / n_dct);
    int i_slice = int(idx / nXY);

    pout[i_slice*n_view*n_dct + i_view*n_dct + i_dct] = pin[i_slice*n_view*ext_dct + i_view*ext_dct + i_dct]/ext_dct;

    return ;
}




DLLEXPORT __global__  void magnitude(cufftReal *pout, cufftComplex *pin, int n)
{
    int     idx    = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= n) return;

    pout[idx] = sqrt(pin[idx].x*pin[idx].x + pin[idx].y*pin[idx].y);

    return ;
}


DLLEXPORT __global__ void multiply_filter(cufftComplex *pview_ft, cufftReal *filter, int n, int ext) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        pview_ft[idx].x =  pview_ft[idx].x * filter[idx];
        pview_ft[idx].y =  pview_ft[idx].y * filter[idx];

    }
    else return ;
}






/*

##############backprojection###############

*/







DLLEXPORT __device__ float calculate_angle(float cx, float cy, float x1, float y1, float x2, float y2) {
    // 벡터 v1 = (x1 - cx, y1 - cy), v2 = (x2 - cx, y2 - cy)
    float v1x = x1 - cx;
    float v1y = y1 - cy;
    float v2x = x2 - cx;
    float v2y = y2 - cy;
    
    // 벡터의 내적을 계산
    float dot_product = v1x * v2x + v1y * v2y;
    
    // 벡터의 크기를 계산
    float magnitude_v1 = sqrtf(v1x * v1x + v1y * v1y);
    float magnitude_v2 = sqrtf(v2x * v2x + v2y * v2y);
    

    float cross_product = v1x * v2y - v1y * v2x;


    // 코사인 값 구하기
    float cos_theta = dot_product / (magnitude_v1 * magnitude_v2);

    // 아크 코사인으로 각도를 구하고, 라디안을 도 단위로 변환
    float theta  = acosf(cos_theta);

    if (cross_product < 0) {
        theta  = -theta ; // 시계 방향일 경우 음수
    }

    return theta;
}

DLLEXPORT __device__ float get_angle(float cx, float cy, float x, float y) {
    float dx = x - cx;
    float dy = y - cy;
    float angle = atan2(dy, dx); // 라디안 단위의 각도
    if (angle < 0) {
        angle += 2 * PI; // 0 ~ 2PI 범위로 변환
    }
    return angle;
}



DLLEXPORT __device__ Point find_intersection(float cx, float cy, float r, 
                      float x1, float y1, float x2, float y2, 
                      float angle_start, float angle_end) 
{

    Point p;
    float dx = x2 - x1;
    float dy = y2 - y1;

    float a = dx * dx + dy * dy;
    float b = 2.0 * (dx * (x1 - cx) + dy * (y1 - cy));
    float c = (x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy) - r * r;

    // 판별식
    float discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0) {
        // 교점이 없음
        return ;
    }

    // 두 교점에 해당하는 t 값 계산
    float t1 = (-b + sqrtf(discriminant)) / (2.0 * a);
    float t2 = (-b - sqrtf(discriminant)) / (2.0 * a);

    // 두 교점 구하기
    float ix1 = x1 + t1 * dx;
    float iy1 = y1 + t1 * dy;

    float ix2 = x1 + t2 * dx;
    float iy2 = y1 + t2 * dy;



    // 첫 번째 교점의 각도 구하기
    float angle1 = get_angle(cx, cy, ix1, iy1);
    float angle2 = get_angle(cx, cy, ix2, iy2);


    // 첫 번째 교점이 각도 범위 안에 있는지 확인
    if (angle1 >= angle_start && angle1 <= angle_end) {
        
       p.x = ix1;
       p.y = iy1;

    }

    // 두 번째 교점이 각도 범위 안에 있는지 확인
    if (angle2 >= angle_start && angle2 <= angle_end) {
        
       p.x = ix2;
       p.y = iy2;
    }

    return p;
}


DLLEXPORT __global__ void backprojection( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float dbeta, float half_angle, int air_pos, 
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID)
{   

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int nXY = img_mat_X * img_mat_Y;

    if ( idx >= nXY*img_mat_Z) return;

    int idx_X = int((idx % nXY) % img_mat_X);
    int idx_Y = int((idx % nXY) / img_mat_X);
    int idx_Z = int(idx / nXY);

    // 현재 복셀의 공간상 좌표(mm)
    float pix_ori_Y = id_to_cor(idx_Y, voxel_Y, img_mat_Y);
    float pix_ori_X = id_to_cor(idx_X, voxel_X, img_mat_X);

    float dct_id = 0.0;
    float angle = 0.0;
    float L2 = 0.0;
    float angle_p = 0.0, angle_i = 0.0, src_x=0.0, src_y=0.0, angle_h = (n_dct-1)*dct_ang_pitch/2.0; 
    float alpha = 0.0;
    float dx = 0.0, dy = 0.0;

    for (int i_view = 0; i_view < n_view; i_view++)
    {
        //dct_ang_pitch = Pd/Rd         
        angle_i = ang_iso_center[i_view] + PI/2.0;  //SSU에서 90도 차감해서 쓰고 있어서 복구함
        src_x = -SOD[i_view]*cosf(angle_i);
        src_y = -SOD[i_view]*sinf(angle_i); 
        angle_p = angle_i + angle_offset[i_view];      
        alpha = SID[i_view]/Rd - 1;

        dx = pix_ori_X - src_x;
        dy = pix_ori_Y - src_y;

        angle = atan2f(dy, dx);  //-pi ~ pi
        angle = atanf(tanf(angle - angle_p));  // -pi/2 ~ pi/2 // //2025.1.22일 메모 참조

        dct_id = (angle_h + (1+alpha-0.1652*alpha*angle*angle)*angle)/dct_ang_pitch; //2025.1.22일 메모 참조

        L2 = dx*dx + dy*dy;

        if (idx_Y < air_pos) {
            pout[idx_Z * img_mat_X * img_mat_Y + img_mat_X * idx_Y + idx_X] += dct_ang_pitch* dbeta/L2 * tex3D<float>(tex,dct_id+0.5f, i_view+0.5f, idx_Z+0.5f) ;
        }
        //pout[idx_Z * img_mat_X * img_mat_Y + img_mat_X * idx_Y + idx_X] += dbeta/(PI2*dct_ang_pitch*L2) * tex3D<float>(tex,dct_id+0.5f, i_view+0.5f, idx_Z+0.5f) ;

    }


    return ;

}







/*

##############ProjectionT###############

*/


DLLEXPORT __global__ void projectionT( cudaTextureObject_t tex, float *pout, 
                int n_view, int n_dct, float Rd,
                int img_mat_X,int img_mat_Y,int img_mat_Z, 
                float voxel_X,float voxel_Y,float voxel_Z, 
                float dct_ang_pitch, float dbeta, float half_angle,
                float dct_unit_vector_T, float dct_unit_vector_S,
                float center_of_circle_T, float center_of_circle_S,
                float *ang_iso_center, float *angle_offset, float *det_tilt,
                float *SOD, float *SID)
{   

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int nXY = img_mat_X * img_mat_Y;

    if ( idx >= nXY*img_mat_Z) return;

    int idx_X = int((idx % nXY) % img_mat_X);
    int idx_Y = int((idx % nXY) / img_mat_X);
    int idx_Z = int(idx / nXY);


    float pix_ori_Y = id_to_cor(idx_Y, voxel_Y, img_mat_Y);
    float pix_ori_X = id_to_cor(idx_X, voxel_X, img_mat_X);

    float radius = sqrtf(pix_ori_X * pix_ori_X + pix_ori_Y * pix_ori_Y);
    float phi = atan2f(pix_ori_Y, pix_ori_X);

    for (int i_view = 0; i_view < n_view; i_view++)
    {
        
        float src_ori_T = 0.0;
        float src_ori_S = -SOD[i_view]; 

        //beta rotate
        // src position
        float src_rot_T = cosf(ang_iso_center[i_view]) * src_ori_T - sinf(ang_iso_center[i_view]) * src_ori_S ; 
        float src_rot_S = sinf(ang_iso_center[i_view]) * src_ori_T  + cosf(ang_iso_center[i_view]) * src_ori_S;

        float dct_rot_vector_T = cosf(ang_iso_center[i_view]) * dct_unit_vector_T - sinf(ang_iso_center[i_view]) * dct_unit_vector_S ;
        float dct_rot_vector_S = sinf(ang_iso_center[i_view]) * dct_unit_vector_T  + cosf(ang_iso_center[i_view]) * dct_unit_vector_S;
        
        // offset rotate
        float dct_offset_vector_T = cosf(angle_offset[i_view]) * (dct_rot_vector_T)  - sinf(angle_offset[i_view]) * (dct_rot_vector_S);
        float dct_offset_vector_S = sinf(angle_offset[i_view]) * (dct_rot_vector_T)  + cosf(angle_offset[i_view]) * (dct_rot_vector_S) ;

        // dct center position
        float dct_rot_T = src_rot_T + dct_offset_vector_T*SID[i_view]; 
        float dct_rot_S = src_rot_S + dct_offset_vector_S*SID[i_view]; 


        float L = sqrtf(pow((pix_ori_X - src_rot_T),2) + pow((pix_ori_Y - src_rot_S),2));



        float center_of_circle_rot_T = cosf(ang_iso_center[i_view]+det_tilt[i_view]) * center_of_circle_T - sinf(ang_iso_center[i_view]+det_tilt[i_view]) * center_of_circle_S ; 
        float center_of_circle_rot_S = sinf(ang_iso_center[i_view]+det_tilt[i_view]) * center_of_circle_T  + cosf(ang_iso_center[i_view]+det_tilt[i_view]) * center_of_circle_S;

        

        float center_angle = ang_iso_center[i_view] + det_tilt[i_view] + 90.0*PI/180.0;
        float start_angle =  center_angle -  half_angle;
        float end_angle =  center_angle +  half_angle;        


        float r = sqrtf(pow((dct_rot_T - center_of_circle_rot_T),2) + pow((dct_rot_S - center_of_circle_rot_S),2));
        Point intersected_point = find_intersection(center_of_circle_rot_T,center_of_circle_rot_S,r,
                                              src_rot_T,src_rot_S,pix_ori_X,pix_ori_Y,
                                              start_angle,end_angle);


        // Point intersected_point = find_intersection(center_of_circle_rot_T,center_of_circle_rot_S,Rd,
        //                                       src_rot_T,src_rot_S,pix_ori_X,pix_ori_Y,
        //                                       start_angle,end_angle);



        float gamma = calculate_angle(center_of_circle_rot_T,center_of_circle_rot_S,dct_rot_T,dct_rot_S,intersected_point.x,intersected_point.y);





        float dct_id = cor_to_id(gamma, dct_ang_pitch, n_dct);

        pout[idx_Z * img_mat_X * img_mat_Y + img_mat_X * idx_Y + idx_X] +=  dbeta * n_view /(2.0 * dct_ang_pitch * PI * L * L) * tex3D<float>(tex,dct_id+0.5f, i_view+0.5f, idx_Z+0.5f) ;
        


    }


    return ;

}
