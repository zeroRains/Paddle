#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle

np.random.seed(100)
paddle.seed(100)


class TestPoly2MaskOpAPI(unittest.TestCase):
    """Test Poly2Mask api."""

    def test_dygraph(self):
        inputs = [
            [
                8.46675669,
                0.71374409,
                9.08619699,
                -0.58200715,
                10.35329272,
                -0.65174745,
                11.45141363,
                -0.25661963,
                11.76114814,
                0.13268563,
                11.93006551,
                0.91129614,
                11.76114814,
                1.56208304,
                13.9714688,
                2.2128818,
                14.12633605,
                3.8979383,
                14.45012068,
                5.78636941,
                11.93006551,
                5.85027529,
                8.14294334,
                5.85027529,
                5.94670153,
                5.78636941,
                4.36992874,
                5.72245168,
                3.73638088,
                4.87411272,
                3.58151362,
                4.42088973,
                3.42667509,
                3.83400871,
                3.42667509,
                3.11932779,
                4.04608666,
                2.34070542,
                4.83450178,
                2.01531789,
                6.4113033,
                1.36451913,
                7.35455695,
                1.23668366,
                7.83320883,
                1.10304935,
                8.62162395,
                0.91129614,
                8.62162395,
                0.58590862,
            ],
            [
                8.2978106,
                7.21576683,
                13.33792093,
                7.01821478,
                13.33792093,
                7.93047146,
                13.49278819,
                8.64516424,
                13.33792093,
                9.16231683,
                9.87461211,
                9.09838723,
                9.71974486,
                11.50397093,
                9.71974486,
                12.34651104,
                10.19842547,
                13.06701452,
                9.71974486,
                13.45631978,
                7.6783703,
                12.67189856,
                7.35455695,
                10.66143082,
                7.6783703,
                9.09838723,
                7.98807608,
                8.64516424,
                8.2978106,
                8.44761219,
                8.2978106,
                7.80264785,
                8.2978106,
                7.27969642,
            ],
        ]
        inputs_numpy = [np.array(i, dtype=np.float32) for i in inputs]
        inputs_tensor = [paddle.to_tensor(i, dtype='float32') for i in inputs]
        h = 14
        w = 14
        actual = paddle.poly2mask(inputs_tensor, h, w)

        import pycocotools.mask as mask_util

        rles = mask_util.frPyObjects(inputs_numpy, h, w)
        rle = mask_util.merge(rles)
        expected = mask_util.decode(rle).astype(np.bool_)

        self.assertTrue(
            (actual.numpy() == expected).all(),
            msg='poly2mask output is wrong, out =' + str(actual.numpy()),
        )


if __name__ == "__main__":
    unittest.main()
