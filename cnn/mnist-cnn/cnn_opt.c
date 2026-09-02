/* =====================================================================
 * cnn_opt.c -- single-file, RVV-optimized MNIST CNN (from the TA's C).
 *   28x28 -> Conv(5x5,8) -> ReLU -> MaxPool(2x2) -> flatten(1152)
 *          -> Dense(1152->10) -> softmax -> argmax
 * Optimizations (RVV 1.0 intrinsics inline, scalar fallback), S4D-style:
 *   - conv: im2col + GEMM vectorized ACROSS patches (no cross-lane reductions)
 *   - ReLU: vectorized vfmax
 *   - dense: vectorized across the 10 outputs (rank-1 fmacc accumulation)
 *   - softmax: range-reduced Horner exp (Taha's hpp) + max-subtraction
 * ===================================================================== */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "profile.h"
#ifdef __riscv
#include <riscv_vector.h>
#endif
#include "config.h"      /* constants, weights, output buffers */
#include "image_data.h"  /* input_data (baked 28x28 image) */

#define PATCH_SIZE  (CL_FILTER_DIM * CL_FILTER_DIM)   /* 25 */
#define NUM_PATCHES (CL_OUT_DIM   * CL_OUT_DIM)       /* 576 */
static void DUMP_(const char*f,const float*p,int n){FILE*o=fopen(f,"w");for(int i=0;i<n;i++)fprintf(o,"%.6f\n",p[i]);fclose(o);}
static float im2col_T[PATCH_SIZE * NUM_PATCHES];       /* [patch_element][patch] */

/* ---- range-reduced Horner exp (ported from Taha's range_reduced_exp_horner.hpp) ---- */
static float pow2i(int n){ if(n<-126)n=-126; if(n>127)n=127; uint32_t b=(uint32_t)(n+127)<<23; float f; memcpy(&f,&b,4); return f; }
static float my_exp(float x){
    const float LN2=0.6931471805599453f;
    int n=(int)(x/LN2); float r=x-(float)n*LN2;
    float e=1.0f/3628800.0f;
    e=e*r+1.0f/362880.0f; e=e*r+1.0f/40320.0f; e=e*r+1.0f/5040.0f; e=e*r+1.0f/720.0f;
    e=e*r+1.0f/120.0f;    e=e*r+1.0f/24.0f;    e=e*r+1.0f/6.0f;    e=e*r+1.0f/2.0f;
    e=e*r+1.0f;           e=e*r+1.0f;
    return e*pow2i(n);
}

/* ---- im2col into transposed layout: im2col_T[k*NUM_PATCHES + p] ---- */
static void im2col(const float* input){
    for(int fi=0;fi<CL_FILTER_DIM;fi++) for(int fj=0;fj<CL_FILTER_DIM;fj++){
        int k=fi*CL_FILTER_DIM+fj; float* row=&im2col_T[k*NUM_PATCHES];
        for(int i=0;i<CL_OUT_DIM;i++){
            const float* src=&input[(i*CL_STRIDE+fi)*CL_IN_DIM + fj];
            float* d=&row[i*CL_OUT_DIM];
            for(int j=0;j<CL_OUT_DIM;j++) d[j]=src[j*CL_STRIDE];
        }
    }
}

/* ---- conv2d: im2col + GEMM, vectorized across patches (accumulator resident) ---- */
static void conv2d_opt(const float* input){
    im2col(input);
    for(int f=0;f<CL_NUM_FILTERS;f++){
        const float* filt=&conv_filters[f*PATCH_SIZE];
        float bias=conv_biases[f]; float* out=&conv_out[f*NUM_PATCHES];
#if defined(__riscv) && !defined(FORCE_SCALAR)
        for(int p=0;p<NUM_PATCHES;){
            size_t vl=__riscv_vsetvl_e32m8(NUM_PATCHES-p);
            vfloat32m8_t acc=__riscv_vfmv_v_f_f32m8(bias,vl);
            for(int k=0;k<PATCH_SIZE;k++)
                acc=__riscv_vfmacc_vf_f32m8(acc,filt[k],__riscv_vle32_v_f32m8(&im2col_T[k*NUM_PATCHES+p],vl),vl);
            __riscv_vse32_v_f32m8(&out[p],acc,vl); p+=(int)vl;
        }
#else
        for(int p=0;p<NUM_PATCHES;p++){ float s=bias;
            for(int k=0;k<PATCH_SIZE;k++) s+=filt[k]*im2col_T[k*NUM_PATCHES+p]; out[p]=s; }
#endif
    }
}

/* ---- ReLU (vectorized) ---- */
static void relu_opt(float* x,int n){
#if defined(__riscv) && !defined(FORCE_SCALAR)
    for(int i=0;i<n;){ size_t vl=__riscv_vsetvl_e32m8(n-i);
        __riscv_vse32_v_f32m8(&x[i],__riscv_vfmax_vf_f32m8(__riscv_vle32_v_f32m8(&x[i],vl),0.0f,vl),vl); i+=(int)vl; }
#else
    for(int i=0;i<n;i++) if(x[i]<0.0f) x[i]=0.0f;
#endif
}

/* ---- maxPool 2x2 (scalar; cheap) ---- */
static void maxpool_opt(const float* input){
    int ia=MP_IN_DIM*MP_IN_DIM, oa=MP_OUT_DIM*MP_OUT_DIM;
    for(int c=0;c<MP_OUT_CHANNELS;c++){ int io=c*ia, oo=c*oa;
        for(int i=0;i<MP_OUT_DIM;i++) for(int j=0;j<MP_OUT_DIM;j++){
            float mx=-1e30f;
            for(int di=0;di<MP_KERNEL_DIM;di++) for(int dj=0;dj<MP_KERNEL_DIM;dj++){
                float v=input[io+(i*MP_STRIDE+di)*MP_IN_DIM+(j*MP_STRIDE+dj)]; if(v>mx)mx=v; }
            max_pool_out[oo+i*MP_OUT_DIM+j]=mx;
        }
    }
}

/* ---- flatten (channel-interleave, scalar) ---- */
static void flatten_opt(const float* input){
    int idx=0, sp=FL_IN_DIM*FL_IN_DIM;
    for(int i=0;i<FL_IN_DIM;i++) for(int j=0;j<FL_IN_DIM;j++){ int pi=i*FL_IN_DIM+j;
        for(int c=0;c<FL_IN_CHANNELS;c++) flattened_out[idx++]=input[c*sp+pi]; }
}

/* ---- dense (S4D-style linear: vectorized across the 10 outputs) ---- */
static void dense_opt(const float* input){
#if defined(__riscv) && !defined(FORCE_SCALAR)
    size_t vl=__riscv_vsetvl_e32m2(D_OUT_DIM);           /* 10 <= 16 lanes (m2) */
    vfloat32m2_t acc=__riscv_vle32_v_f32m2(dense_biases,vl);
    for(int j=0;j<D_IN_DIM;j++)
        acc=__riscv_vfmacc_vf_f32m2(acc,input[j],__riscv_vle32_v_f32m2(&dense_weights[j*D_OUT_DIM],vl),vl);
    __riscv_vse32_v_f32m2(dense_out,acc,vl);
#else
    for(int i=0;i<D_OUT_DIM;i++){ float s=dense_biases[i];
        for(int j=0;j<D_IN_DIM;j++) s+=input[j]*dense_weights[j*D_OUT_DIM+i]; dense_out[i]=s; }
#endif
}

/* ---- softmax (Horner exp + max-subtraction for stability) ---- */
static void softmax_opt(float* x,int n){
    float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    float s=0.0f; for(int i=0;i<n;i++){ x[i]=my_exp(x[i]-m); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=s;
}

static void argmax(float* p,int n){ int b=0; for(int i=1;i<n;i++) if(p[i]>p[b])b=i; prediction=b; }

static const char* STAGE[6]={"conv2d","relu","maxpool","dense","softmax","flatten"};
static uint64_t g_inst[6];

int main(void){
    init_inst_counter();
    uint64_t t0=get_inst_count(),t1;
    #define TICK(i) do{ t1=get_inst_count(); g_inst[i]=t1-t0; t0=t1; }while(0)
    conv2d_opt(input_data);                                        TICK(0);
#ifdef DUMP
    DUMP_("o_conv.txt",conv_out,CL_OUT_DIM*CL_OUT_DIM*CL_NUM_FILTERS);
#endif
    relu_opt(conv_out, CL_OUT_DIM*CL_OUT_DIM*CL_NUM_FILTERS);        TICK(1);
#ifdef DUMP
    DUMP_("o_relu.txt",conv_out,CL_OUT_DIM*CL_OUT_DIM*CL_NUM_FILTERS);
#endif
    maxpool_opt(conv_out);                                          TICK(2);
#ifdef DUMP
    DUMP_("o_maxpool.txt",max_pool_out,MP_OUT_DIM*MP_OUT_DIM*MP_OUT_CHANNELS);
#endif
    flatten_opt(max_pool_out);                                      TICK(5);
#ifdef DUMP
    DUMP_("o_flat.txt",flattened_out,FL_OUT_DIM);
#endif
    dense_opt(flattened_out);                                       TICK(3);
#ifdef DUMP
    DUMP_("o_dense.txt",dense_out,D_OUT_DIM);
#endif
    softmax_opt(dense_out, D_OUT_DIM);                              TICK(4);
    argmax(dense_out, D_OUT_DIM);
    #undef TICK
#ifdef DUMP
    FILE*o;
    #define DMP(f,p,n) do{o=fopen(f,"w");for(int i=0;i<n;i++)fprintf(o,"%.6f\n",(p)[i]);fclose(o);}while(0)
    /* recompute stages to dump pre-softmax dense too */
#endif
    printf("Predicted label: %d\n", prediction);
    printf("softmax:"); for(int i=0;i<D_OUT_DIM;i++) printf(" %.5f",dense_out[i]); printf("\n");
    uint64_t tot=0;
    printf("Per-stage instruction counts\n");
    for(int i=0;i<6;i++){ printf("  %-9s : %llu\n",STAGE[i],(unsigned long long)g_inst[i]); tot+=g_inst[i]; }
    printf("  %-9s : %llu\n","TOTAL",(unsigned long long)tot);
    return 0;
}
