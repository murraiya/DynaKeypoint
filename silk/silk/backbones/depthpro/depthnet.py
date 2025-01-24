# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, Iterable, List, Union

import torch
import skimage.io as io
import silk.backbones.depthpro.ml_depth_pro.src.depth_pro as depth_pro

from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from silk.flow import AutoForward, Flow


class DepthNet(AutoForward, torch.nn.Module):
    def __init__(
        self,
        # input_name: Union[str, Tuple[str]]= ("images", "intrinsics"),
        # default_outputs: Union[str, Iterable[str]]= "depthpro",
    ):
        torch.nn.Module.__init__(self)
        self.model, self.transform = depth_pro.create_model_and_transforms(device="cuda:1")
        self.model.eval()
        
        # AutoForward.__init__(self, Flow(input_name), default_outputs=default_outputs)

        # self.flow.define_transition(
        #     default_outputs,
        #     self.model,
        #     input_name,
        # )
        
        # backbone input, output shapes
        # in vgg.py forward  torch.Size([2, 1, 164, 164])
        # in vgg.py forward  torch.Size([2, 128, 148, 148])
        dim_decoder = 512
        last_dims=(32, 1)
        self.pose_head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.depth_head = nn.Sequential(
            # nn.Conv2d(
            #     dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            # ),
            nn.Linear(1226, 1226, 1),
            nn.ReLU(True),
        )

        # Set the final convolution layer's bias to be 0.
        # self.pose_head[4].bias.data.fill_(0)
        # self.depth_head[4].bias.data.fill_(0)
        
        # self._pose_pred = nn.Conv2d(last_dims[1], 6, kernel_size=1, padding=0)
        


    def forward(self, images: torch.Tensor, intrinsics) -> torch.Tensor:
        """
        Goes through the layers of the VGG model as the forward pass.
        Computes the output.
        Args:
            images (tensor): image pytorch tensor with
                shape N x num_channels x H x W
        Returns:
            output (tensor): the output point pytorch tensor with
            shape N x cell_size^2+1 x H/8 x W/8.
        """
        # print(intrinsics)
        # tensor([[707.0912,   0.0000, 601.8873],
        # [  0.0000, 707.0912, 183.1104],
        # [  0.0000,   0.0000,   1.0000]], device='cuda:1')

        # fx = (intrinsics[0][0]+intrinsics[1][1])/2
        # print(images.shape)
        # print(min(images.reshape(-1)), max(images.reshape(-1)))
        # tensor(0., device='cuda:1') tensor(255., device='cuda:1')
        # print(images.shape)
        # torch.Size([2, 3, 370, 1226])

        images[0] = self.transform(images[0])
        images[1] = self.transform(images[1])
        # print(min(images.reshape(-1)), max(images.reshape(-1)))
        # tensor(-1., device='cuda:1') tensor(1., device='cuda:1')
        
        # print(images.shape)
        # from original ml-depth-pro: <class 'torch.Tensor'> torch.Size([3, 375, 1242])
        # here: torch.Size([2, 3, 370, 1226])
       


        prediction = self.model.infer(images, f_px=intrinsics[0][0])
        depth = prediction["depth"]  # Depth in [m].
        # print(depth.shape)
        # torch.Size([2, 370, 1226])
        # depth = self.depth_head(depth)
        # print(depth.shape)
        # torch.Size([2, 370, 1226])

        features = prediction["features"].clone()
        # print(features.shape)
        # torch.Size([2, 256, 768, 768])

        features = torch.cat([features[0], features[1]], dim=0).unsqueeze(0)
        # print(features.shape)
        # torch.Size([1, 512, 768, 768])
        pose = self.pose_head(features)
        # print(pose.shape)
        # torch.Size([1, 1, 1536, 1536]
        pose = self._pose_pred(pose)
        # print(pose.shape)
        # torch.Size([1, 6, 1536, 1536])
        pose = 0.01*pose.mean(3).mean(2)
        # print(pose.shape)
        # torch.Size([1, 6])
        # pose = 0.01 * pose.view(pose.size(0), 1, 6)
        # print("final pose shape", pose.shape)
        # final pose shape torch.Size([1, 1, 6])
        
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("+====================================================")
        # print(pose.requires_grad)
        # True
        return depth, pose


        
        # prediction = self.model.infer(images[0], f_px=intrinsics[0][0])
        # depth_1 = prediction["depth"]  # Depth in [m].
        # prediction = self.model.infer(images[1], f_px = intrinsics[0][0])
        # depth_2 = prediction["depth"]  # Depth in [m].
        # # io.imsave("./depth.png", depth_1.cpu().numpy())

        # # print(depth_1.shape) torch.Size([370, 1226])
        # print(min(depth_1.reshape(-1)), max(depth_1.reshape(-1)))
        # # tensor(0.0071, device='cuda:1') tensor(0.5267, device='cuda:1')
        # # tensor(4.2536, device='cuda:1') tensor(10000., device='cuda:1')

        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        # # print(focallength_px, intrinsics[0][0])
        # # tensor(707.0912, device='cuda:1') tensor(707.0912, device='cuda:1')
        
        
        # depth = torch.stack([depth_1, depth_2], dim=0)
        # print(depth.shape)
        # torch.Size([2, 370, 1226])
        
        # the last version did the same. 
        # return depth
