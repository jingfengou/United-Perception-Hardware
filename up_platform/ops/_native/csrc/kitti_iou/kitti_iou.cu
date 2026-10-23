// Port of UP's KITTI Numba evaluator, retaining its corner/intersection algorithm.
// Original: RRPN-revise, yanyan (scrin@foxmail.com), MIT license.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
// Preserve the reference Numba contraction order, including its asymmetric
// dot-product rounding. Reassociation changes exact-edge inclusion predicates.
__device__ float mul(float a, float b) { return __fmul_rn(a,b); }
__device__ void corners(const float* b, float* p) {
    float c = cosf(b[4]), s = sinf(b[4]);
    float x[4] = {-b[2]/2, -b[2]/2, b[2]/2, b[2]/2};
    float y[4] = {-b[3]/2, b[3]/2, b[3]/2, -b[3]/2};
    for (int i=0; i<4; ++i) {
        p[2*i] = __fmaf_rn(c,x[i],mul(s,y[i]))+b[0];
        p[2*i+1] = (mul(c,y[i])-mul(s,x[i]))+b[1];
    }
}

__device__ bool inside(float x, float y, const float* p) {
    float ab0=p[2]-p[0], ab1=p[3]-p[1], ad0=p[6]-p[0], ad1=p[7]-p[1];
    float ap0=x-p[0], ap1=y-p[1];
    float abab=__fmaf_rn(ab0,ab0,mul(ab1,ab1)), abap=__fmaf_rn(ab1,ap1,mul(ab0,ap0));
    float adad=__fmaf_rn(ad0,ad0,mul(ad1,ad1)), adap=__fmaf_rn(ad1,ap1,mul(ad0,ap0));
    return abab>=abap && abap>=0 && adad>=adap && adap>=0;
}

__device__ bool segment(const float* p, const float* q, int i, int j, float* t) {
    const float *a=p+2*i, *b=p+2*((i+1)%4), *c=q+2*j, *d=q+2*((j+1)%4);
    float ba0=b[0]-a[0], ba1=b[1]-a[1], da0=d[0]-a[0], ca0=c[0]-a[0];
    float da1=d[1]-a[1], ca1=c[1]-a[1];
    bool acd=mul(da1,ca0)>mul(ca1,da0), bcd=mul(d[1]-b[1],c[0]-b[0])>mul(c[1]-b[1],d[0]-b[0]);
    if (acd!=bcd && (mul(ca1,ba0)>mul(ba1,ca0))!=(mul(da1,ba0)>mul(ba1,da0))) {
        float dc0=d[0]-c[0], dc1=d[1]-c[1], abba=mul(a[0],b[1])-mul(b[0],a[1]), cddc=mul(c[0],d[1])-mul(d[0],c[1]);
        float dh=mul(ba1,dc0)-mul(ba0,dc1);
        t[0]=(mul(abba,dc0)-mul(ba0,cddc))/dh;
        t[1]=(mul(abba,dc1)-mul(ba1,cddc))/dh;
        return true;
    }
    return false;
}

__device__ double intersection(const float* a, const float* b) {
    float p[8], q[8], v[48], angles[24], t[2];
    corners(a,p); corners(b,q);
    int count=0;
    for (int i=0; i<4; ++i) {
        if (inside(p[2*i],p[2*i+1],q)) { v[2*count]=p[2*i]; v[2*count+1]=p[2*i+1]; ++count; }
        if (inside(q[2*i],q[2*i+1],p)) { v[2*count]=q[2*i]; v[2*count+1]=q[2*i+1]; ++count; }
    }
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) {
        if (segment(p,q,i,j,t)) { v[2*count]=t[0]; v[2*count+1]=t[1]; ++count; }
    }
    if (!count) return 0.0;
    float cx=0,cy=0;
    for (int i=0; i<count; ++i) { cx+=v[2*i]; cy+=v[2*i+1]; }
    cx/=count; cy/=count;
    for (int i=0; i<count; ++i) {
        float x=v[2*i]-cx, y=v[2*i+1]-cy, norm=sqrtf(__fmaf_rn(x,x,mul(y,y)));
        x/=norm; y/=norm;
        angles[i]=y<0 ? -2-x : x;
    }
    for (int i=1; i<count; ++i) {
        if (angles[i-1]>angles[i]) {
            float angle=angles[i], x=v[2*i], y=v[2*i+1];
            int j=i;
            while (j>0 && angles[j-1]>angle) {
                angles[j]=angles[j-1]; v[2*j]=v[2*j-2]; v[2*j+1]=v[2*j-1]; --j;
            }
            angles[j]=angle; v[2*j]=x; v[2*j+1]=y;
        }
    }
    double area=0;
    for (int i=0; i<count-2; ++i) {
        const float *b=v+2*i+2, *c=v+2*i+4;
        float cross=mul(v[0]-c[0],b[1]-c[1])-mul(v[1]-c[1],b[0]-c[0]);
        area+=fabs(static_cast<double>(cross)/2.0);
    }
    return area;
}

__global__ void evaluate(const float* boxes, const float* queries, float* out, int64_t n, int64_t k, int criterion) {
    int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=n*k) return;
    // Original kernel passes query first; criterion 0 is query area, not box area.
    const float *a=queries+5*(index%k), *b=boxes+5*(index/k);
    float area1=a[2]*a[3], area2=b[2]*b[3];
    double overlap=intersection(a,b);
    out[index]=criterion==-1 ? overlap/(area1+area2-overlap) :
        criterion==0 ? overlap/area1 : criterion==1 ? overlap/area2 : overlap;
}
}

torch::Tensor kitti_rotate_iou(torch::Tensor boxes, torch::Tensor queries, int64_t criterion) {
    TORCH_CHECK(boxes.is_cuda() && queries.is_cuda(), "KITTI IoU requires accelerator tensors");
    TORCH_CHECK(boxes.device()==queries.device(), "KITTI inputs must share a device");
    TORCH_CHECK(boxes.scalar_type()==torch::kFloat32 && queries.scalar_type()==torch::kFloat32, "KITTI IoU requires float32");
    TORCH_CHECK(boxes.is_contiguous() && queries.is_contiguous(), "KITTI inputs must be contiguous");
    TORCH_CHECK(boxes.dim()==2 && queries.dim()==2 && boxes.size(1)==5 && queries.size(1)==5, "Expected [N,5] and [K,5]");
    c10::cuda::CUDAGuard guard(boxes.device());
    auto out=torch::zeros({boxes.size(0), queries.size(0)}, boxes.options());
    if (out.numel()) {
        evaluate<<<(out.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
            boxes.data_ptr<float>(),queries.data_ptr<float>(),out.data_ptr<float>(),boxes.size(0),queries.size(0),criterion);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
