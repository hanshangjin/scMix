#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               const float *__restrict__ _delta_t,
                               F *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F delta_t = _delta_t[i];

        F no = max(o, u + k[ii]);
        F A = exp(o - no);
        F B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        F w_tmp = -exp(w * delta_t);

        no = max(w_tmp + o, k[ii]);
        A = exp(w_tmp + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _delta_t, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = max(o, k[ii] + u);
        F A = exp(o - no);
        F B = exp(k[ii] + u - no);

        F num = A * p + B * v[ii];
        F iden = 1 / (A * q + B);

        F delta_t = _delta_t[i];

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gy[ii] * (v[ii] - y[i]) * B * iden;

        F w_tmp = -exp(w * delta_t);

        no = max(w_tmp + o, k[ii]);
        A = exp(w_tmp + o - no);
        B = exp(k[ii] - no);
        // dpdw = A * (p + dpdw);
        // dqdw = A * (q + dqdw);
        dpdw = A * (w_tmp * delta_t * p + dpdw);
        dqdw = A * (w_tmp * delta_t * q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    F gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;

        F delta_t = _delta_t[i];

        F A = gy[ii] * z[i] * exp(zexp[i]);
        F B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        F w_tmp = -exp(w * delta_t);

        F no = max(w_tmp + o, zexp[i] - k[ii] - u);
        A = exp(w_tmp + o - no);
        B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *delta_t, float *y) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, delta_t, y);
}

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *delta_t, float *gy, float *gw, float *gu, float *gk, float *gv) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, delta_t, gy, gw, gu, gk, gv);
}
