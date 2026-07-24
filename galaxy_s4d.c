/* =====================================================================
 * galaxy_s4d.c  --  Single-file, optimized S4D galaxy-morphology classifier
 * ---------------------------------------------------------------------
 * 64x64 grayscale image -> Hilbert scan -> linear up-projection ->
 * S4D -> GELU -> S4D -> GELU -> take-last -> linear head -> softmax.
 *
 * This is the best C variant from the optimization study ("opt #14"):
 *   - O(L*N) recurrent S4D scan (not O(L^2) convolution)
 *   - B_bar folded into C_bar  (x = B*x' change of variables; scan drops B)
 *   - per-channel discretization constants kept resident in vector regs
 *   - RVV 1.0 vectorized scan + vectorized GELU (with scalar fallbacks)
 *   - custom Remez-minimax math (exp/sin/cos/tanh/sqrt) inlined in this TU
 * All math is from scratch: no libm dependency in the forward pass.
 *
 * Build/run:  see README.md.  Profiling: prints per-layer retired-
 * instruction counts via profile.h (RISC-V `instret` CSR, or perf on x86).
 * ===================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "profile.h"
#ifdef __riscv
#include <riscv_vector.h>
#endif

/* ---- model dimensions ---- */
#define IMG_SIZE     64      /* input is 64x64 */
#define SEQ_LEN      4096    /* 64*64 positions after the Hilbert scan */
#define D_MODEL      64      /* hidden dimension */
#define D_STATE      64      /* S4D state dim (32 complex pairs) */
#define N_CLASSES    4       /* galaxy classes */
#define IN_CHANNELS  1       /* grayscale */
#define WEIGHTS_SIZE_FLOATS 21124

static float bits2f(uint32_t b){ float f; memcpy(&f, &b, 4); return f; }
static uint32_t f2bits(float f){ uint32_t b; memcpy(&b, &f, 4); return b; }

/* -----------------------------------------------------------------------
 * my_exp(x) = e^x
 * x = n*ln2 + r, r in [-ln2/2, ln2/2]  =>  e^x = 2^n * e^r
 * ln2 split hi/lo (Cody-Waite) so n*ln2_hi is exact in f32 across the
 * whole supported domain. e^r: Remez-fit minimax quintic (6 coefficients).
 * 2^n: written directly into an f32's exponent bits.
 * ----------------------------------------------------------------------- */
static float my_exp(float x){
    if (x < -88.0f) return 0.0f;
    if (x >  88.0f) return bits2f(0x7F800000);  /* +inf */

    const float inv_ln2 = bits2f(0x3FB8AA3B);
    const float ln2_hi  = bits2f(0x3F317200);
    const float ln2_lo  = bits2f(0x35BFBE8E);

    float t  = x * inv_ln2;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);  /* round to nearest */
    float nf = (float)n;

    float r = x - nf * ln2_hi;                  // FMA
    r = r - nf * ln2_lo;                          // FMA

    float p = bits2f(0x3AB6ECC1);                 /* c6 */
    p = p * r + bits2f(0x3C0937D6);                 // FMA (c5)
    p = p * r + bits2f(0x3D2AAA0E);                 // FMA (c4)
    p = p * r + bits2f(0x3E2AAA02);                 // FMA (c3)
    p = p * r + bits2f(0x3F000000);                 // FMA (c2)
    p = p * r + bits2f(0x3F800000);                 // FMA (c1)
    p = p * r + bits2f(0x3F800000);                 // FMA (c0)

    uint32_t bits = (uint32_t)(n + 127) << 23;
    return p * bits2f(bits);
}

/* -----------------------------------------------------------------------
 * my_sin(x), my_cos(x)
 * Reduce to r in [-pi,pi] via n = round(x/2pi), r = x - n*2pi, using a
 * Cody-Waite hi/lo split of 2pi. Rounding to nearest lands r in [-pi,pi]
 * directly -- no quadrant-correction branch needed (unlike the original's
 * truncate-then-correct approach).
 * sin/cos of r: fixed degree-13/12 Remez minimax polynomial in r^2 (7
 * terms) -- no loop, no division, same instruction count every call.
 * ----------------------------------------------------------------------- */
static float my_sin(float x){
    const float twopi_hi  = bits2f(0x40C90F80);
    const float twopi_lo  = bits2f(0x38354443);
    const float recip_2pi = bits2f(0x3E22F983);

    float t = x * recip_2pi;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);
    float nf = (float)n;

    float r = x - nf * twopi_hi;                     // FMA
    r = r - nf * twopi_lo;                             // FMA

    float x2 = r * r;
    float p = bits2f(0x2F15B1B2);                       /* c6 */
    p = p * x2 + bits2f(0xB2D46C36);                      // FMA (c5)
    p = p * x2 + bits2f(0x3638CA51);                      // FMA (c4)
    p = p * x2 + bits2f(0xB9500B09);                      // FMA (c3)
    p = p * x2 + bits2f(0x3C08887C);                      // FMA (c2)
    p = p * x2 + bits2f(0xBE2AAAAA);                      // FMA (c1)
    p = p * x2 + bits2f(0x3F800000);                      // FMA (c0)

    return r * p;
}

static float my_cos(float x){
    const float twopi_hi  = bits2f(0x40C90F80);
    const float twopi_lo  = bits2f(0x38354443);
    const float recip_2pi = bits2f(0x3E22F983);

    float t = x * recip_2pi;
    int32_t n = (int32_t)(t >= 0.0f ? t + 0.5f : t - 0.5f);
    float nf = (float)n;

    float r = x - nf * twopi_hi;                      // FMA
    r = r - nf * twopi_lo;                               // FMA

    float x2 = r * r;
    /* v2: degree-14 (8-coefficient) Remez minimax, one degree higher than
     * the original optimized cut's degree-12 fit -- see "v2 CHANGELOG" at
     * the top of this file. */
    float p = bits2f(0xAD2B0E51);                          /* c7 */
    p = p * x2 + bits2f(0x310D96C7);                         // FMA (c6)
    p = p * x2 + bits2f(0xB493D39B);                         // FMA (c5)
    p = p * x2 + bits2f(0x37D00ACA);                         // FMA (c4)
    p = p * x2 + bits2f(0xBAB60B4B);                         // FMA (c3)
    p = p * x2 + bits2f(0x3D2AAAAA);                         // FMA (c2)
    p = p * x2 + bits2f(0xBF000000);                         // FMA (c1)
    p = p * x2 + bits2f(0x3F800000);                         // FMA (c0)

    return p;
}

/* -----------------------------------------------------------------------
 * my_tanh(x) = 1 - 2/(e^(2x)+1)  -- same identity as the original, built
 * on the fast my_exp() above. No clamp needed: my_exp saturates to 0 / inf
 * at the extremes, which makes this formula saturate to -1 / +1 correctly.
 * ----------------------------------------------------------------------- */
static float my_tanh(float x){
    float e2x = my_exp(x + x);
    return 1.0f - 2.0f / (e2x + 1.0f);
}

/* -----------------------------------------------------------------------
 * my_sqrt(x): bit-hack initial guess for 1/sqrt(x) ("fast inverse square
 * root"), refined by three Newton iterations (multiply-only, no
 * division), then sqrt(x) = x * (1/sqrt(x)).
 * This is only ever called twice per inference (a constant inside gelu()),
 * so it isn't performance-critical -- three iterations were chosen
 * (instead of two) purely to match the original's ~1e-7 accuracy rather
 * than to save instructions.
 * ----------------------------------------------------------------------- */
static float my_sqrt(float x){
    if (x <= 0.0f) return 0.0f;

    float xhalf = 0.5f * x;
    uint32_t i = f2bits(x);
    i = 0x5f3759df - (i >> 1);
    float y = bits2f(i);

    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);

    return x * y;
}

#ifdef __riscv
#endif

// Helper function for complex multiplication
static void complex_mul(float a_real, float a_imag, float b_real, float b_imag, float* out_real, float* out_imag) {
    *out_real = a_real * b_real - a_imag * b_imag;
    *out_imag = a_real * b_imag + a_imag * b_real;
}

// Helper function for complex exponential
static void complex_exp(float a_real, float a_imag, float* out_real, float* out_imag) {
    float exp_a = my_exp(a_real);
    *out_real = exp_a * my_cos(a_imag);
    *out_imag = exp_a * my_sin(a_imag);
}

void hilbert_scan(float input[IN_CHANNELS][IMG_SIZE][IMG_SIZE], float output[SEQ_LEN][IN_CHANNELS], const int* hilbert_indices) {

//Flatten input manually and reorder
    for (int d = 0 ; d < SEQ_LEN ; d++) {
        int flat_idx = hilbert_indices[d];
        
        // SAFETY BOUNDS CHECK to prevent Segfaults if binary file formatting is weird
        if (flat_idx < 0 || flat_idx >= IMG_SIZE * IMG_SIZE) {
            flat_idx = 0; 
        }
        
        int y = flat_idx / IMG_SIZE;
        int x = flat_idx % IMG_SIZE;
        for (int c = 0 ; c < IN_CHANNELS ; c++) {
            output[d][c] = input[c][y][x];
        }
    }
}

//Generic linear layer implementation (linear uproject & fc kept for compatibility with main.c and test.c)
void linear(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int in_features,
    int out_features
) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {
            float acc = bias[j];
            for (int k = 0; k < in_features; k++) {
                acc += input[i * in_features + k] * weight[j * in_features + k];
            }
            output[i * out_features + j] = acc;
        }
    }
}

void s4d_layer(
    float input[SEQ_LEN][D_MODEL],
    float output[SEQ_LEN][D_MODEL],
    const float* log_dt,
    const float* log_A_real,
    const float* A_imag,
    const float* C_real,
    const float* C_imag,
    const float* D
) {
    int half_state = D_STATE / 2;  // 32

    for (int h = 0; h < D_MODEL; h++) {
        float dt = my_exp(log_dt[h]);

        float A_bar_real_arr[32], A_bar_imag_arr[32];
        float B_bar_real_arr[32], B_bar_imag_arr[32];
        float C_real_arr[32], C_imag_arr[32];  // hoisted C loads (loop-invariant across t)

        // Discretize once per channel, these stay constant for all 4096 timesteps.
        for (int n = 0; n < half_state; n++) {
            float lambda_real = -my_exp(log_A_real[h * half_state + n]);
            float lambda_imag = A_imag[h * half_state + n];

            // A_bar = exp(lambda * dt)
            complex_exp(lambda_real * dt, lambda_imag * dt, &A_bar_real_arr[n], &A_bar_imag_arr[n]);

            // B_bar = (A_bar - 1) / lambda. B is fixed at 1 in this parameterization,
            // so there's nothing extra to multiply in, B_bar is the whole input gain.
            float N_real = A_bar_real_arr[n] - 1.0f;
            float N_imag = A_bar_imag_arr[n];
            float denom = lambda_real * lambda_real + lambda_imag * lambda_imag;

            B_bar_real_arr[n] = (N_real * lambda_real + N_imag * lambda_imag) / denom;
            B_bar_imag_arr[n] = (N_imag * lambda_real - N_real * lambda_imag) / denom;

            // Fold B_bar into C and fold the output factor 2:  Cbar = 2 * C * B_bar.
            // With x = B_bar*x', the scan drops B entirely (x' = A*x' + u, u real).
            float _cr = C_real[(h * half_state + n) * 2];
            float _ci = C_imag[(h * half_state + n) * 2];
            float _br = B_bar_real_arr[n], _bi = B_bar_imag_arr[n];
            C_real_arr[n] = 2.0f * (_cr * _br - _ci * _bi);
            C_imag_arr[n] = 2.0f * (_cr * _bi + _ci * _br);
        }

        // State vector for this channel, zeroed at t=0.
        float x_real[32] = {0};
        float x_imag[32] = {0};

#ifdef __riscv
        {
            size_t vl = __riscv_vsetvl_e32m4(half_state);           // 32 states, one vector
            vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            // Load per-channel CONSTANTS once; kept resident across all timesteps.
            vfloat32m4_t var = __riscv_vle32_v_f32m4(A_bar_real_arr, vl);
            vfloat32m4_t vai = __riscv_vle32_v_f32m4(A_bar_imag_arr, vl);
            vfloat32m4_t vcr = __riscv_vle32_v_f32m4(C_real_arr, vl);
            vfloat32m4_t vci = __riscv_vle32_v_f32m4(C_imag_arr, vl);
            // Recurrent STATE stays in vector registers across the whole t-loop.
            vfloat32m4_t vxr = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vxi = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            for (int t = 0; t < SEQ_LEN; t++) {
                float u_t = input[t][h];
                vfloat32m4_t vdr = __riscv_vfmul_vv_f32m4(var, vxr, vl);
                vdr = __riscv_vfnmsac_vv_f32m4(vdr, vai, vxi, vl);   // ar*xr - ai*xi
                vfloat32m4_t vdi = __riscv_vfmul_vv_f32m4(var, vxi, vl);
                vdi = __riscv_vfmacc_vv_f32m4(vdi, vai, vxr, vl);    // ar*xi + ai*xr  = x'_i(t)
                vxr = __riscv_vfadd_vf_f32m4(vdr, u_t, vl);          // dr + u  (B folded away)
                vxi = vdi;                                          // x'_i(t) (register coalesce)
                vfloat32m4_t vt = __riscv_vfmul_vv_f32m4(vcr, vxr, vl);
                vt = __riscv_vfnmsac_vv_f32m4(vt, vci, vxi, vl);     // Cbar includes the *2
                vfloat32m1_t vred = __riscv_vfredosum_vs_f32m4_f32m1(vt, vzero, vl);
                output[t][h] = D[h]*u_t + __riscv_vfmv_f_s_f32m1_f32(vred);
            }
        }
#else
        for (int t = 0; t < SEQ_LEN; t++) {
            float u_t = input[t][h];
            float y = D[h] * u_t;
            for (int n = 0; n < half_state; n++) {
                float decayed_real, decayed_imag;
                complex_mul(A_bar_real_arr[n], A_bar_imag_arr[n], x_real[n], x_imag[n], &decayed_real, &decayed_imag);
                x_real[n] = decayed_real + u_t;      // B folded away (x' = A*x' + u)
                x_imag[n] = decayed_imag;
                y += (C_real_arr[n] * x_real[n] - C_imag_arr[n] * x_imag[n]);  // Cbar has the *2
            }
            output[t][h] = y;
        }
#endif
    }
}

#ifdef __riscv
static inline vfloat32m4_t v_exp_m4(vfloat32m4_t x, size_t vl){
    x = __riscv_vfmax_vf_f32m4(x, -88.0f, vl);
    x = __riscv_vfmin_vf_f32m4(x,  88.0f, vl);
    vfloat32m4_t t = __riscv_vfmul_vf_f32m4(x, 1.44269502f, vl);        // x/ln2
    vint32m4_t   n = __riscv_vfcvt_x_f_v_i32m4(t, vl);                  // round to nearest
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsac_vf_f32m4(x, 0.693145752f, nf, vl); // x - nf*ln2_hi
    r = __riscv_vfnmsac_vf_f32m4(r, 1.42860677e-06f, nf, vl);          // - nf*ln2_lo
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(0.00139560562f, vl);       // c6
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.00837512873f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.041666083f,  vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.166664153f,  vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 0.5f,          vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f,          vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), 1.0f,          vl);
    vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}
static inline vfloat32m4_t v_tanh_m4(vfloat32m4_t x, size_t vl){
    vfloat32m4_t e2x = v_exp_m4(__riscv_vfadd_vv_f32m4(x, x, vl), vl);
    vfloat32m4_t d = __riscv_vfadd_vf_f32m4(e2x, 1.0f, vl);
    return __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(d, 2.0f, vl), 1.0f, vl); // 1 - 2/(e2x+1)
}
#endif

void gelu(float* x, int size) {
    const float pi = 3.141592653589793f;
    const float k = my_sqrt(2.0f / pi);
    const float coeff = 0.044715f;

#ifdef __riscv
    size_t vl;
    for (int i = 0; i < size; i += (int)vl) {
        vl = __riscv_vsetvl_e32m4(size - i);
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(&x[i], vl);
        vfloat32m4_t vx3 = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(vx, vx, vl), vx, vl);
        vfloat32m4_t inner = __riscv_vfmul_vf_f32m4(vx3, coeff, vl);       // coeff*x^3
        inner = __riscv_vfadd_vv_f32m4(inner, vx, vl);                    // x + coeff*x^3
        inner = __riscv_vfmul_vf_f32m4(inner, k, vl);                     // * k
        vfloat32m4_t vt = v_tanh_m4(inner, vl);
        vfloat32m4_t r = __riscv_vfmul_vv_f32m4(__riscv_vfadd_vf_f32m4(vt, 1.0f, vl), vx, vl);
        r = __riscv_vfmul_vf_f32m4(r, 0.5f, vl);                          // 0.5*x*(1+tanh)
        __riscv_vse32_v_f32m4(&x[i], r, vl);
    }
#else
    for (int i = 0; i < size; i++) {
        float x_cubed = x[i] * x[i] * x[i];
        float inner = k * (x[i] + coeff * x_cubed);
        float tanh_val = my_tanh(inner);
        x[i] = 0.5f * x[i] * (1.0f + tanh_val);
    }
#endif
}

void softmax(float* logits, int size) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = my_exp(logits[i] - max_val);
        sum += logits[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        logits[i] /= sum;
    }
}

void take_last_timestamp(float input[SEQ_LEN][D_MODEL], float output[D_MODEL]) {
    for (int j = 0; j < D_MODEL; j++) {
        output[j] = input[SEQ_LEN - 1][j];
    }
}

/* =====================================================================
 * Forward pass. Weight layout in the flat model_weights.bin buffer:
 *   [0]      hilbert_scan.indices  (4096 int32)
 *   [..]     uproject.weight (64x1), uproject.bias (64)
 *   [..]     s4_1: log_dt(64), log_A_real(64x32), A_imag(64x32),
 *                  C(64x32x2 interleaved re/im), D(64)
 *   [..]     s4_2: same shape as s4_1
 *   [..]     fc.weight (4x64), fc.bias (4)
 * The forward pass brackets every layer with the instruction counter and
 * prints a per-layer breakdown (the whole point of this study).
 * ===================================================================== */
static const char *LAYER_NAMES[9] = {
    "hilbert","input_proj","s4_1","gelu_1","s4_2","gelu_2","ttls","output_proj","softmax"
};
static uint64_t g_layer_insts[9];

void model_forward(
    float image[IN_CHANNELS][IMG_SIZE][IMG_SIZE],
    float probabilities[N_CLASSES],
    const float *model_weights,
    const int   *hilbert_indices
) {
    int off = 0;
    off += 4096 * (int)sizeof(int);
    const float *up_w   = (const float *)((const char *)model_weights + off); off += 64*1*(int)sizeof(float);
    const float *up_b   = (const float *)((const char *)model_weights + off); off += 64*(int)sizeof(float);
    const float *s1_ldt = (const float *)((const char *)model_weights + off); off += 64*(int)sizeof(float);
    const float *s1_lar = (const float *)((const char *)model_weights + off); off += 64*32*(int)sizeof(float);
    const float *s1_ai  = (const float *)((const char *)model_weights + off); off += 64*32*(int)sizeof(float);
    const float *s1_cr  = (const float *)((const char *)model_weights + off);
    const float *s1_ci  = s1_cr + 1;                                          off += 64*32*2*(int)sizeof(float);
    const float *s1_D   = (const float *)((const char *)model_weights + off); off += 64*(int)sizeof(float);
    const float *s2_ldt = (const float *)((const char *)model_weights + off); off += 64*(int)sizeof(float);
    const float *s2_lar = (const float *)((const char *)model_weights + off); off += 64*32*(int)sizeof(float);
    const float *s2_ai  = (const float *)((const char *)model_weights + off); off += 64*32*(int)sizeof(float);
    const float *s2_cr  = (const float *)((const char *)model_weights + off);
    const float *s2_ci  = s2_cr + 1;                                          off += 64*32*2*(int)sizeof(float);
    const float *s2_D   = (const float *)((const char *)model_weights + off); off += 64*(int)sizeof(float);
    const float *fc_w   = (const float *)((const char *)model_weights + off); off += 4*64*(int)sizeof(float);
    const float *fc_b   = (const float *)((const char *)model_weights + off);

    static float hilbert_out[SEQ_LEN][IN_CHANNELS];
    static float proj_out[SEQ_LEN][D_MODEL];
    static float s4d1_out[SEQ_LEN][D_MODEL];
    static float s4d2_out[SEQ_LEN][D_MODEL];
    static float pooled[D_MODEL];
    static float logits[N_CLASSES];

    uint64_t t0 = get_inst_count(), t1;
    #define TICK(i) do { t1 = get_inst_count(); g_layer_insts[i] = t1 - t0; t0 = t1; } while (0)

    hilbert_scan(image, hilbert_out, hilbert_indices);                                     TICK(0);
    linear((float*)hilbert_out, (float*)proj_out, up_w, up_b, SEQ_LEN, IN_CHANNELS, D_MODEL); TICK(1);
    s4d_layer(proj_out, s4d1_out, s1_ldt, s1_lar, s1_ai, s1_cr, s1_ci, s1_D);               TICK(2);
    gelu(&s4d1_out[0][0], SEQ_LEN * D_MODEL);                                               TICK(3);
    s4d_layer(s4d1_out, s4d2_out, s2_ldt, s2_lar, s2_ai, s2_cr, s2_ci, s2_D);               TICK(4);
    gelu(&s4d2_out[0][0], SEQ_LEN * D_MODEL);                                               TICK(5);
    take_last_timestamp(s4d2_out, pooled);                                                 TICK(6);
    linear(pooled, logits, fc_w, fc_b, 1, D_MODEL, N_CLASSES);                              TICK(7);
    for (int i = 0; i < N_CLASSES; i++) probabilities[i] = logits[i];
    softmax(probabilities, N_CLASSES);                                                     TICK(8);
    #undef TICK
}


#ifndef BAKED
/* Built-in validation: run samples 0-4 and compare to the PyTorch reference
 * softmax (test_data/sample_N_softmax.bin). No python needed. */
static int argmax4(const float *p){ int m=0; for(int i=1;i<N_CLASSES;i++) if(p[i]>p[m]) m=i; return m; }

static int run_validation(void) {
    static float weights[WEIGHTS_SIZE_FLOATS];
    FILE *fw = fopen("model_params/model_weights.bin", "rb");
    if (!fw) { fprintf(stderr, "Error: run from the repo root (model_params/model_weights.bin not found)\n"); return 1; }
    if (fread(weights, sizeof(float), WEIGHTS_SIZE_FLOATS, fw) != WEIGHTS_SIZE_FLOATS)
        fprintf(stderr, "Warning: unexpected weight file size\n");
    fclose(fw);
    const int   *hidx = (const int   *)weights;
    const float *mw   = (const float *)weights;
    const char *names[N_CLASSES] = { "Round Elliptical","In-between Elliptical","Cigar-shaped Elliptical","Edge-on Disk" };

    int pass = 0, total = 0;
    printf("Validating against PyTorch reference (test_data/):\n");
    for (int sN = 0; sN < 5; sN++) {
        char path[256];
        static float image[IN_CHANNELS][IMG_SIZE][IMG_SIZE], probs[N_CLASSES], ref[N_CLASSES];
        snprintf(path, sizeof(path), "test_data/sample_%d_img.bin", sN);
        FILE *fi = fopen(path, "rb"); if (!fi) continue;
        if (fread(image, sizeof(float), IN_CHANNELS*IMG_SIZE*IMG_SIZE, fi) == 0) { fclose(fi); continue; }
        fclose(fi);
        snprintf(path, sizeof(path), "test_data/sample_%d_softmax.bin", sN);
        FILE *fr = fopen(path, "rb"); if (!fr) continue;
        if (fread(ref, sizeof(float), N_CLASSES, fr) != N_CLASSES) { fclose(fr); continue; }
        fclose(fr);

        model_forward(image, probs, mw, hidx);
        int cp = argmax4(probs), cr = argmax4(ref);
        float md = 0.0f; for (int i=0;i<N_CLASSES;i++){ float d = probs[i]-ref[i]; if (d<0) d=-d; if (d>md) md=d; }
        total++; if (cp==cr) pass++;
        printf("  sample %d: %-22s vs ref %-22s | max prob diff %.2e | %s\n",
               sN, names[cp], names[cr], md, cp==cr ? "MATCH" : "MISMATCH");
    }
    printf("\n%d/%d classes match the PyTorch reference%s\n",
           pass, total, (pass==total && total>0) ? "  (probabilities match to the fp32 floor)" : "");
    return (pass==total && total>0) ? 0 : 1;
}
#endif

#ifdef BAKED
#include "bench_data.h"   /* baked weights + sample image for RISC-V/QEMU (no file I/O) */
#endif

int main(int argc, char *argv[]) {
    static float weights[WEIGHTS_SIZE_FLOATS];
    static float image[IN_CHANNELS][IMG_SIZE][IMG_SIZE];
    static float probs[N_CLASSES];

#ifdef BAKED
    /* Weights + sample-0 image are compiled in (QEMU's newlib can't fopen files).
     * Build with -DBAKED for the RISC-V instruction-count benchmark. */
    memcpy(weights, BENCH_WEIGHTS, sizeof(weights));
    memcpy(image,   BENCH_IMAGE,   sizeof(image));
    (void)argc; (void)argv;
#else
    if (argc == 2 && strcmp(argv[1], "--validate") == 0) return run_validation();
    if (argc != 2) {
        printf("Usage: %s <input_image.bin>   (or: %s --validate)\n", argv[0], argv[0]);
        return 1;
    }
    FILE *fw = fopen("model_params/model_weights.bin", "rb");
    if (!fw) { fprintf(stderr, "Error: model_params/model_weights.bin not found (run from repo root)\n"); return 1; }
    if (fread(weights, sizeof(float), WEIGHTS_SIZE_FLOATS, fw) != WEIGHTS_SIZE_FLOATS)
        fprintf(stderr, "Warning: unexpected weight file size\n");
    fclose(fw);

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { fprintf(stderr, "Error: could not open image %s\n", argv[1]); return 1; }
    if (fread(image, sizeof(float), IN_CHANNELS*IMG_SIZE*IMG_SIZE, fi) == 0)
        fprintf(stderr, "Warning: failed to read image\n");
    fclose(fi);
#endif

    const int   *hilbert_indices = (const int   *)weights;
    const float *model_weights   = (const float *)weights;

    init_inst_counter();
    model_forward(image, probs, model_weights, hilbert_indices);

    const char *names[N_CLASSES] = {
        "Round Elliptical","In-between Elliptical","Cigar-shaped Elliptical","Edge-on Disk"
    };
    int best = 0;
    printf("\nClass probabilities\n");
    for (int i = 0; i < N_CLASSES; i++) {
        printf("  %d  %-24s : %6.2f%%\n", i, names[i], probs[i]*100.0f);
        if (probs[i] > probs[best]) best = i;
    }
    printf("\nPrediction: %s\n", names[best]);

    uint64_t total = 0;
    printf("\nPer-layer dynamic instruction counts\n");
    for (int i = 0; i < 9; i++) { printf("  %-12s : %12llu\n", LAYER_NAMES[i], (unsigned long long)g_layer_insts[i]); total += g_layer_insts[i]; }
    printf("  %-12s : %12llu\n", "TOTAL", (unsigned long long)total);
    fflush(stdout);
    return 0;
}
