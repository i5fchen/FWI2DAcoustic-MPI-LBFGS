enum {

    LBFGS_SUCCESS,
    LBFGSERR_MAXIMUMLINESEARCH,
    LBFGSERR_MAXIMUMITERATION
};
static void vecncpy(float *y, const float *x, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] = -x[i];
    }
}
static void vecadd(float *y, const float *x, const float c, const int n)
{
    int i;
    for (i = 0;i < n;++i) {
        y[i] += c * x[i];
    }
}
static void vecdot(float* s, const float *x, const float *y, const int n)
{
    int i;
    *s = 0.;
    for (i = 0;i < n;++i) {
        *s += x[i] * y[i];
    }
}
static void vec2norm(float* s, const float *x, const int n)
{
    vecdot(s, x, x, n);
    *s = (float)sqrt(*s);
}
static void vec2norminv(float* s, const float *x, const int n)
{
    vec2norm(s, x, n);
    *s = (float)(1.0 / *s);
}
static void veccpy(float *y, const float *x, const int n)
{
    int i;

    for (i = 0;i < n;++i) {
        y[i] = x[i];
    }
}
static void vecdiff(float *z, const float *x, const float *y, const int n)
{
    int i;

    for (i = 0;i < n;++i) {
        z[i] = x[i] - y[i];
    }
}
static void vecscale(float *y, const float c, const int n)
{
    int i;

    for (i = 0;i < n;++i) {
        y[i] *= c;
    }
}
typedef struct {
    int m;
    float epsilon;
    int max_iterations;
    int max_linesearch;
    float ftol;
    float wolfe;
} lbfgs_parameter_t;
typedef struct tag_iteration_data {
    float alpha;
    float *s;     /* [n] */
    float *y;     /* [n] */
    float ys;     /* vecdot(y, s) */
} iteration_data_t;
static const lbfgs_parameter_t _defparam = {
    12, 1e-9, 8, 6, 1e-7, 0.98 //5
};
#define send_data_tag3 3001
#define send_data_tag2 2001
#define send_data_tag1 1001
#define send_data_tag0 9999
#define N (NZ*NX)

