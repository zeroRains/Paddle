// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

/**
 * @brief This kernel is used to transform the poly label to the mask.
 * @param  ctx          device context
 * @param  segs         the list of the input tensor of polygamma
 * @param  h            the height of output mask
 * @param  w            the width of output mask
 * @param  final_out    the output tensor of mask
 */
template <typename T, typename Context>
void Poly2MaskKernel(const Context& dev_ctx,
                     const std::vector<DenseTensor*>& segs,
                     int h,
                     int w,
                     DenseTensor* final_out);

}  // namespace phi
