// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/poly2mask_kernel.h"

#include <thrust/fill.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define WARP_SIZE 32

namespace phi {

__global__ void ExtendIndex(const double *seg,
                            int num,
                            double scale,
                            int *d_xy) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  d_xy[tid] = static_cast<int>(scale * seg[tid] + .5);
  __syncthreads();
  if (tid == 0) {
    d_xy[num] = seg[0];
    d_xy[num + 1] = seg[1];
  }
}

__global__ void GetBound(int *xy, int k, int *bound, int *m_out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 0) m_out[0] = 0;
  __syncthreads();

  int dx = std::abs(xy[2 * tid] - xy[2 * (tid + 1)]);
  int dy = std::abs(xy[2 * tid + 1] - xy[2 * (tid + 1) + 1]);
  int now_bound = std::max(dx, dy) + 1;
  m_out[0] = m_out[0] + now_bound;
  bound[tid] = now_bound;

  __syncthreads();
  if (tid == 0) {
    int t = bound[0];
    bound[0] = 0;
    for (int i = 1; i < k; i++) {
      int tt = bound[i];
      bound[i] = bound[i - 1] + t;
      t = tt;
    }
  }
}

__global__ void LinearInsert(
    int *d_xy, int k, int *d_bound, int *d_u, int *d_v) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int xs = d_xy[tid * 2];
  int ys = d_xy[tid * 2 + 1];
  int xe = d_xy[(tid + 1) * 2];
  int ye = d_xy[(tid + 1) * 2 + 1];

  int dx = std::abs(xe - xs);
  int dy = std::abs(ye - ys);
  int t;
  bool flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
  if (flip) {
    t = xs;
    xs = xe;
    xe = t;
    t = ys;
    ys = ye;
    ye = t;
  }
  double s = dx >= dy ? static_cast<double>(ye - ys) / dx
                      : static_cast<double>(xe - xs) / dy;
  if (dx >= dy) {
    for (int i = 0; i <= dx; i++) {
      t = flip ? dx - i : i;
      d_u[d_bound[tid] + i] = xs + t;
      d_v[d_bound[tid] + i] = static_cast<int>(ys + s * t + 0.5);
    }
  } else {
    for (int i = 0; i <= dy; i++) {
      t = flip ? dy - i : i;
      d_v[d_bound[tid] + i] = t + ys;
      d_u[d_bound[tid] + i] = static_cast<int>(xs + s * t + .5);
    }
  }
}

__global__ void Simplify(int *d_u,
                         int *d_v,
                         double scale,
                         int h,
                         int w,
                         int *d_m,
                         int *d_x,
                         int *d_y) {
  double xd, yd;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) d_m[0] = -1;

  if (tid > 0 && d_u[tid] != d_u[tid - 1]) {
    xd = static_cast<double>(d_u[tid] < d_u[tid - 1] ? d_u[tid] : d_u[tid] - 1);
    xd = (xd + .5) / scale - .5;
    if (!(std::floor(xd) != xd || xd < 0 || xd > w - 1)) {
      yd = static_cast<double>(d_v[tid] < d_v[tid - 1] ? d_v[tid]
                                                       : d_v[tid - 1]);
      yd = (yd + .5) / scale - .5;
      if (yd < 0)
        yd = 0;
      else if (yd > h)
        yd = h;
      yd = std::ceil(yd);
      int new_index = phi::CudaAtomicAdd(d_m, 1);
      d_x[new_index] = static_cast<int> xd;
      d_y[new_index] = static_cast<int> yd;
    }
  }
}

__global__ void Transform(
    int *d_x, int *d_y, int h, int w, int m, uint32_t *d_a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) d_a[m] = (uint32_t)(h * w);
  d_a[tid] = (uint32_t)(d_x[tid] * h + d_y[tid]);
}

template <typename T, typename Context>
void Poly2MaskKernel(const Context &dev_ctx,
                     const std::vector<DenseTensor *> &segs,
                     int h,
                     int w,
                     DenseTensor *final_out) {
  // Encode
  int n = segs.size();
  std::vector<DenseTensor> out(n);
  double scale = 5;
  std::vector<int> segs_m;
  for (int j = 0; j < n; j++) {
    auto *seg = segs[j]->data<double>();
    int num = segs[j]->dims()[0];
    int k = (num + .5) / 2;
    int threads = std::min(num, WARP_SIZE);
    int blocks = (num + .5) / threads;

    DenseTensor xy;
    xy.Resize({num + 2});
    int *d_xy = dev_ctx.template Alloc<int>(&xy);
    ExtendIndex<<<blocks, threads, 0, dev_ctx.stream()>>>(
        seg, num, scale, d_xy);

    DenseTensor bound, m;
    bound.Resize({k});
    m.Resize({1});
    auto *d_bound = dev_ctx.template Alloc<int>(&bound);
    auto *d_m = dev_ctx.template Alloc<int>(&m);
    threads = std::min(k, WARP_SIZE);
    blocks = (k + .5) / threads;
    // determine the refine bound

    GetBound<<<blocks, threads, 0, dev_ctx.stream()>>>(d_xy, k, d_bound, d_m);

    DenseTensor h_m;
    phi::Copy(dev_ctx, m, phi::CPUPlace(), false, &h_m);
    int new_m = h_m.data<int>()[0];
    DenseTensor u, v;
    u.Resize({new_m});
    v.Resize({new_m});
    auto *d_u = dev_ctx.template Alloc<int>(&u);
    auto *d_v = dev_ctx.template Alloc<int>(&v);
    LinearInsert<<<blocks, threads, 0, dev_ctx.stream()>>>(
        d_xy, k, d_bound, d_u, d_v);

    DenseTensor x, y;
    x.Resize({new_m});
    y.Resize({new_m});
    int *d_x = dev_ctx.template Alloc<int>(&x);
    int *d_y = dev_ctx.template Alloc<int>(&y);
    threads = std::min(new_m, threads);
    blocks = (new_m + .5) / threads;

    Simplify<<<blocks, threads, 0, dev_ctx.stream()>>>(
        d_u, d_v, scale, h, w, d_m, d_x, d_y);

    phi::Copy(dev_ctx, m, phi::CPUPlace(), true, &h_m);
    new_m = h_m.data<int>()[0];
    out[j].Resize({new_m + 1});
    auto *out_temp = dev_ctx.template Alloc<uint32_t>(&out[j]);
    threads = std::min(new_m, WARP_SIZE);
    blocks = (new_m + .5) / threads;
    Transform<<<blocks, threads, 0, dev_ctx.stream()>>>(
        d_x, d_y, h, w, new_m, out_temp);

    thrust::sort(thrust::device, out_temp, out_temp + new_m);
    uint32_t *out_end_position =
        thrust::unique(thrust::device, out_temp, out_temp + new_m);
    segs_m.push_back(out_end_position - out_temp);
  }

  /* Merge */
  DenseTensor cnt;
  int final_m;
  if (n == 1) {
    cnt = out[0];
    final_m = segs_m[0];
  } else if (n > 1) {
    cnt.Resize({h * w + 1});
    auto *cnts = dev_ctx.template Alloc<uint32_t>(&cnt);
    uint32_t *end = cnts;
    for (int i = 1; i < n; i++) {
      auto *temp_rle = out[i].data<uint32_t>();
      int temp_len = segs_m[i];
      end = thrust::merge(
          thrust::device, cnts, end, temp_rle, temp_rle + temp_len, cnts);
      uint32_t *merge_end = thrust::unique(thrust::device, cnts, end);
      final_m = merge_end - cnts;
    }
  }

  // Decode
  final_out->Resize({h * w});
  auto *d_out = dev_ctx.template Alloc<bool>(final_out);
  bool v = 0;
  DenseTensor h_cnt;
  phi::Copy(dev_ctx, cnt, phi::CPUPlace(), true, &h_cnt);
  auto *res = h_cnt.data<uint32_t>();
  bool *start = d_out;
  bool *end = start;
  for (int i = 0; i < final_m; i++) {
    end = end + res[i];
    thrust::fill(thrust::device, start, end, v);
    v = !v;
    start = end;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(poly2mask, GPU, ALL_LAYOUT, phi::Poly2MaskKernel, bool) {}
