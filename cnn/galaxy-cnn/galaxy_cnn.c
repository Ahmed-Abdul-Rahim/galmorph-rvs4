/* =====================================================================
 * galaxy_cnn.c -- single-file RVV-optimized "Stem4Full (4L) + 0 S4D"
 * GalaxyClassifierGrid (38,468 params). Inference:
 *   3x64x64 -> Conv(3->32,3x3,s1,p1) -> GN(8,32) -> GELU
 *           -> res:Conv(32->32,s1,p1)->GN->GELU ; x = x + res
 *           -> Conv(32->32,3x3,s2,p1) -> GN(8,32) -> GELU        (32x32)
 *           -> Conv(32->64,3x3,s2,p1) -> GN(8,64) -> GELU        (16x16)
 *           -> GlobalAvgPool -> Linear(64->4) -> softmax
 * Optimizations (RVV 1.0 intrinsics inline, scalar fallback), S4D/CNN-style:
 *   - conv: im2col + GEMM vectorized ACROSS patches (no reductions)
 *   - GroupNorm: vectorized sum/normalize
 *   - GELU: tanh-approx via inline Remez exp/tanh (vectorized)
 *   - linear: S4D-style vectorized across the 4 outputs
 *   - softmax: range-reduced Horner exp + max-subtraction
 * Weights are deterministic placeholders (compute is weight-independent;
 * drop in the trained checkpoint for accuracy without changing counts).
 * ===================================================================== */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "profile.h"
#ifdef __riscv
#include <riscv_vector.h>
#endif

#define IN_CH 3
#define IMG   64
#define MID   32
#define DMODEL 64
#define NCLASS 4
#define GN_GROUPS 8
#define KK 3

/* feature buffers (max 32x64x64 = 131072) */
static float bufA[MID*IMG*IMG];
static float bufB[MID*IMG*IMG];
static float bufR[MID*IMG*IMG];
static float im2col_T[(MID*KK*KK) * (IMG*IMG)];   /* max 288 x 4096 */
static float pooled[DMODEL];
static float logits[NCLASS];

/* placeholder weights */
static float conv1_w[MID*IN_CH*KK*KK], conv1_b[MID];
static float res_w[MID*MID*KK*KK],     res_b[MID];
static float d1_w[MID*MID*KK*KK],      d1_b[MID];
static float d2_w[DMODEL*MID*KK*KK],   d2_b[DMODEL];
static float gn1_g[MID], gn1_b[MID], gn2_g[MID], gn2_b[MID];
static float gn3_g[MID], gn3_b[MID], gn4_g[DMODEL], gn4_b[DMODEL];
static float fc_w[NCLASS*DMODEL], fc_b[NCLASS];

/* ---- scalar Remez/Horner math (from S4D + Taha's hpp) ---- */
static float bits2f(uint32_t b){ float f; memcpy(&f,&b,4); return f; }
static float pow2i(int n){ if(n<-126)n=-126; if(n>127)n=127; uint32_t b=(uint32_t)(n+127)<<23; return bits2f(b); }
static float my_exp(float x){                 /* range-reduced Horner (Taha) */
    const float LN2=0.6931471805599453f; int n=(int)(x/LN2); float r=x-(float)n*LN2;
    float e=1.0f/3628800.0f;
    e=e*r+1.0f/362880.0f; e=e*r+1.0f/40320.0f; e=e*r+1.0f/5040.0f; e=e*r+1.0f/720.0f;
    e=e*r+1.0f/120.0f; e=e*r+1.0f/24.0f; e=e*r+1.0f/6.0f; e=e*r+1.0f/2.0f; e=e*r+1.0f; e=e*r+1.0f;
    return e*pow2i(n);
}
static float my_tanh(float x){ float e=my_exp(x+x); return 1.0f-2.0f/(e+1.0f); }
static float my_sqrt(float x){ if(x<=0)return 0; float xh=0.5f*x; uint32_t i; memcpy(&i,&x,4);
    i=0x5f3759df-(i>>1); float y; memcpy(&y,&i,4);
    y=y*(1.5f-xh*y*y); y=y*(1.5f-xh*y*y); y=y*(1.5f-xh*y*y); return x*y; }

/* ---- vectorized exp/tanh (m8) for GELU ---- */
#if defined(__riscv) && !defined(FORCE_SCALAR)
static inline vfloat32m8_t v_exp_m8(vfloat32m8_t x,size_t vl){
    x=__riscv_vfmax_vf_f32m8(x,-88.0f,vl); x=__riscv_vfmin_vf_f32m8(x,88.0f,vl);
    vfloat32m8_t t=__riscv_vfmul_vf_f32m8(x,1.44269502f,vl);
    vint32m8_t n=__riscv_vfcvt_x_f_v_i32m8(t,vl); vfloat32m8_t nf=__riscv_vfcvt_f_x_v_f32m8(n,vl);
    vfloat32m8_t r=__riscv_vfnmsac_vf_f32m8(x,0.693145752f,nf,vl); r=__riscv_vfnmsac_vf_f32m8(r,1.42860677e-06f,nf,vl);
    vfloat32m8_t p=__riscv_vfmv_v_f_f32m8(0.00139560562f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),0.00837512873f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),0.041666083f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),0.166664153f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),0.5f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),1.0f,vl);
    p=__riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p,r,vl),1.0f,vl);
    vint32m8_t e=__riscv_vsll_vx_i32m8(__riscv_vadd_vx_i32m8(n,127,vl),23,vl);
    return __riscv_vfmul_vv_f32m8(p,__riscv_vreinterpret_v_i32m8_f32m8(e),vl);
}
static inline vfloat32m8_t v_tanh_m8(vfloat32m8_t x,size_t vl){
    vfloat32m8_t e=v_exp_m8(__riscv_vfadd_vv_f32m8(x,x,vl),vl);
    vfloat32m8_t d=__riscv_vfadd_vf_f32m8(e,1.0f,vl);
    return __riscv_vfrsub_vf_f32m8(__riscv_vfrdiv_vf_f32m8(d,2.0f,vl),1.0f,vl);
}
#endif

/* ---- conv: im2col (transposed) + GEMM vectorized across patches ---- */
static void conv(const float* in,float* out,int Cin,int Cout,int Hin,int Win,int stride,int pad,
                 const float* W,const float* b){
    int Hout=(Hin+2*pad-KK)/stride+1, Wout=(Win+2*pad-KK)/stride+1;
    int P=Hout*Wout, pdim=Cin*KK*KK;
    for(int ci=0;ci<Cin;ci++) for(int ki=0;ki<KK;ki++) for(int kj=0;kj<KK;kj++){
        int pe=ci*KK*KK+ki*KK+kj; float* row=&im2col_T[pe*P];
        for(int oi=0;oi<Hout;oi++){ int ii=oi*stride-pad+ki;
            for(int oj=0;oj<Wout;oj++){ int jj=oj*stride-pad+kj;
                row[oi*Wout+oj]=(ii>=0&&ii<Hin&&jj>=0&&jj<Win)?in[ci*Hin*Win+ii*Win+jj]:0.0f; }
        }
    }
    for(int co=0;co<Cout;co++){ const float* w=&W[co*pdim]; float bias=b[co]; float* o=&out[co*P];
#if defined(__riscv) && !defined(FORCE_SCALAR)
        for(int p=0;p<P;){ size_t vl=__riscv_vsetvl_e32m8(P-p);
            vfloat32m8_t acc=__riscv_vfmv_v_f_f32m8(bias,vl);
            for(int pe=0;pe<pdim;pe++) acc=__riscv_vfmacc_vf_f32m8(acc,w[pe],__riscv_vle32_v_f32m8(&im2col_T[pe*P+p],vl),vl);
            __riscv_vse32_v_f32m8(&o[p],acc,vl); p+=(int)vl; }
#else
        for(int p=0;p<P;p++){ float s=bias; for(int pe=0;pe<pdim;pe++) s+=w[pe]*im2col_T[pe*P+p]; o[p]=s; }
#endif
    }
}

/* ---- GroupNorm ---- */
static void groupnorm(float* x,int C,int HW,int groups,const float* g,const float* be){
    int cpg=C/groups, gs=cpg*HW; float eps=1e-5f;
    for(int gr=0;gr<groups;gr++){ float* base=&x[gr*cpg*HW]; float sum=0,sq=0;
        for(int i=0;i<gs;i++){ float v=base[i]; sum+=v; sq+=v*v; }
        float mean=sum/gs, var=sq/gs-mean*mean, inv=1.0f/my_sqrt(var+eps);
        for(int c=0;c<cpg;c++){ int ch=gr*cpg+c; float gm=g[ch],bt=be[ch]; float* p=&x[ch*HW];
            for(int i=0;i<HW;i++) p[i]=(p[i]-mean)*inv*gm+bt; }
    }
}

/* ---- GELU (tanh approx, vectorized) ---- */
static void gelu(float* x,int n){
    const float k=0.7978845608f, c=0.044715f;
#if defined(__riscv) && !defined(FORCE_SCALAR)
    for(int i=0;i<n;){ size_t vl=__riscv_vsetvl_e32m8(n-i);
        vfloat32m8_t vx=__riscv_vle32_v_f32m8(&x[i],vl);
        vfloat32m8_t x3=__riscv_vfmul_vv_f32m8(__riscv_vfmul_vv_f32m8(vx,vx,vl),vx,vl);
        vfloat32m8_t inr=__riscv_vfmul_vf_f32m8(x3,c,vl); inr=__riscv_vfadd_vv_f32m8(inr,vx,vl); inr=__riscv_vfmul_vf_f32m8(inr,k,vl);
        vfloat32m8_t t=v_tanh_m8(inr,vl);
        vfloat32m8_t r=__riscv_vfmul_vv_f32m8(__riscv_vfadd_vf_f32m8(t,1.0f,vl),vx,vl); r=__riscv_vfmul_vf_f32m8(r,0.5f,vl);
        __riscv_vse32_v_f32m8(&x[i],r,vl); i+=(int)vl; }
#else
    for(int i=0;i<n;i++){ float v=x[i]; float in=k*(v+c*v*v*v); x[i]=0.5f*v*(1.0f+my_tanh(in)); }
#endif
}

/* ---- residual add: a += b ---- */
static void addv(float* a,const float* b,int n){
#if defined(__riscv) && !defined(FORCE_SCALAR)
    for(int i=0;i<n;){ size_t vl=__riscv_vsetvl_e32m8(n-i);
        __riscv_vse32_v_f32m8(&a[i],__riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(&a[i],vl),__riscv_vle32_v_f32m8(&b[i],vl),vl),vl); i+=(int)vl; }
#else
    for(int i=0;i<n;i++) a[i]+=b[i];
#endif
}

/* ---- global average pool: out[c] = mean_hw x[c] ---- */
static void gap(const float* x,float* out,int C,int HW){
    for(int c=0;c<C;c++){ float s=0; const float* p=&x[c*HW]; for(int i=0;i<HW;i++) s+=p[i]; out[c]=s/HW; }
}

/* ---- linear (S4D-style: vectorized across NCLASS outputs) ---- */
static void linear(const float* in,float* out,const float* W,const float* b,int Nin,int Nout){
#if defined(__riscv) && !defined(FORCE_SCALAR)
    size_t vl=__riscv_vsetvl_e32m1(Nout);
    vfloat32m1_t acc=__riscv_vle32_v_f32m1(b,vl);
    for(int j=0;j<Nin;j++) acc=__riscv_vfmacc_vf_f32m1(acc,in[j],__riscv_vle32_v_f32m1(&W[j*Nout],vl),vl);
    __riscv_vse32_v_f32m1(out,acc,vl);
#else
    for(int i=0;i<Nout;i++){ float s=b[i]; for(int j=0;j<Nin;j++) s+=in[j]*W[j*Nout+i]; out[i]=s; }
#endif
}
static void softmax(float* x,int n){ float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){ x[i]=my_exp(x[i]-m); s+=x[i]; } for(int i=0;i<n;i++) x[i]/=s; }
static int argmax(const float* p,int n){ int b=0; for(int i=1;i<n;i++) if(p[i]>p[b])b=i; return b; }

/* placeholder weight init (xorshift, small scale) -- not part of timed pipeline */
static uint32_t rs=0x1234567u;
static float rnd(){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return ((rs>>8)*(1.0f/16777216.0f)-0.5f)*0.1f; }
static void fillw(float*a,int n){ for(int i=0;i<n;i++) a[i]=rnd(); }
static void fillg(float*a,int n){ for(int i=0;i<n;i++) a[i]=1.0f; }   /* GN gamma=1 */
static void init_weights(void){
    fillw(conv1_w,MID*IN_CH*KK*KK); fillw(conv1_b,MID);
    fillw(res_w,MID*MID*KK*KK);     fillw(res_b,MID);
    fillw(d1_w,MID*MID*KK*KK);      fillw(d1_b,MID);
    fillw(d2_w,DMODEL*MID*KK*KK);   fillw(d2_b,DMODEL);
    fillg(gn1_g,MID); fillw(gn1_b,MID); fillg(gn2_g,MID); fillw(gn2_b,MID);
    fillg(gn3_g,MID); fillw(gn3_b,MID); fillg(gn4_g,DMODEL); fillw(gn4_b,DMODEL);
    fillw(fc_w,NCLASS*DMODEL); fillw(fc_b,NCLASS);
}

static float input_img[IN_CH*IMG*IMG];
static const char* STAGE[8]={"conv1+gn","res+add","down1","down2","gelu(all)","gap","fc","softmax"};
static uint64_t g[8];

int main(void){
    init_weights();
    for(int i=0;i<IN_CH*IMG*IMG;i++) input_img[i]=rnd()+0.05f;   /* dummy image */
    init_inst_counter();
    uint64_t t0=get_inst_count(),t1; uint64_t ge=0;
    #define TICK(i) do{ t1=get_inst_count(); g[i]+=t1-t0; t0=t1; }while(0)
    #define TG do{ t1=get_inst_count(); ge+=t1-t0; t0=t1; }while(0)

    /* stem_conv -> gn -> gelu  (3->32, 64x64) */
    conv(input_img,bufA,IN_CH,MID,IMG,IMG,1,1,conv1_w,conv1_b);   TICK(0);
    groupnorm(bufA,MID,IMG*IMG,GN_GROUPS,gn1_g,gn1_b);            TICK(0);
    gelu(bufA,MID*IMG*IMG);                                       TG;
    /* res block: conv->gn->gelu into bufR, then bufA += bufR */
    conv(bufA,bufR,MID,MID,IMG,IMG,1,1,res_w,res_b);              TICK(1);
    groupnorm(bufR,MID,IMG*IMG,GN_GROUPS,gn2_g,gn2_b);            TICK(1);
    gelu(bufR,MID*IMG*IMG);                                       TG;
    addv(bufA,bufR,MID*IMG*IMG);                                  TICK(1);
    /* down1: 32->32 stride2 -> 32x32 */
    conv(bufA,bufB,MID,MID,IMG,IMG,2,1,d1_w,d1_b);                TICK(2);
    groupnorm(bufB,MID,32*32,GN_GROUPS,gn3_g,gn3_b);              TICK(2);
    gelu(bufB,MID*32*32);                                         TG;
    /* down2: 32->64 stride2 -> 16x16 */
    conv(bufB,bufA,MID,DMODEL,32,32,2,1,d2_w,d2_b);               TICK(3);
    groupnorm(bufA,DMODEL,16*16,GN_GROUPS,gn4_g,gn4_b);           TICK(3);
    gelu(bufA,DMODEL*16*16);                                      TG;
    g[4]=ge;
    /* head */
    gap(bufA,pooled,DMODEL,16*16);                                TICK(5);
    linear(pooled,logits,fc_w,fc_b,DMODEL,NCLASS);                TICK(6);
    softmax(logits,NCLASS);                                       TICK(7);
    int pred=argmax(logits,NCLASS);
    #undef TICK
    #undef TG
    printf("Predicted class: %d\n",pred);
    printf("softmax:"); for(int i=0;i<NCLASS;i++) printf(" %.5f",logits[i]); printf("\n");
    uint64_t tot=0; printf("Per-stage instruction counts\n");
    for(int i=0;i<8;i++){ printf("  %-10s : %llu\n",STAGE[i],(unsigned long long)g[i]); tot+=g[i]; }
    printf("  %-10s : %llu\n","TOTAL",(unsigned long long)tot);
    return 0;
}
