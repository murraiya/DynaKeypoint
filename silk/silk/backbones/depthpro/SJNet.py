# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from silk.backbones.silk.coords import (
    CoordinateMappingComposer,
    CoordinateMappingProvider,
)
import silk.backbones.depthpro.ml_depth_pro.src.depth_pro as depth_pro
from silk.backbones.superpoint.magicpoint import (
    # Backbone as VGGBackbone,
    DetectorHead as VGGDetectorHead,
    MagicPoint,
)
from silk.backbones.superpoint.vgg import ParametricVGG as VGGBackbone
from silk.backbones.superpoint.superpoint import (
    DescriptorHead as VGGDescriptorHead,
    SuperPoint,
)
from silk.backbones.sfmlearner.sfmlearner import(
    PoseHead as VGGPoseHead,
    DispHead as VGGDispHead,
)
# from silk.backbones.depthpro.depthnet import DepthNet

from silk.flow import AutoForward, Flow
from silk.models.superpoint_utils import get_dense_positions



class SJNet(AutoForward, torch.nn.Module):
    def __init__(
        self,
        input_name: Union[str, Tuple[str]]= ("images", "intrinsics"),
        default_outputs: Union[str, Iterable[str]]= ("depths", "logits", "normalized_descriptors"),
        detection_threshold: float = 0.00,
        detection_top_k: int = 3000,
        nms_dist=0,
        border_dist=8,
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
        normalize_descriptors: bool = True,
    ):
        torch.nn.Module.__init__(self)  
        AutoForward.__init__(self, Flow(*input_name), default_outputs=default_outputs)
        
        self.model, self.transform = depth_pro.create_pretrained_depth_pro(device="cuda:1")
            
        
        self.flow.define_transition(
            "depthpro_input",
            self.transform,
            "images",
        )        
        self.flow.define_transition(
            "f_px",
            lambda x: x[0][0],
            "intrinsics",
        )
        self.flow.define_transition(
            ("depths", "logits", "raw_descriptors"),
            self.model.infer_sj,
            "depthpro_input",
            "f_px",
        )

        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )
        self.normalize_descriptors = normalize_descriptors

        MagicPoint.add_detector_head_post_processing(
            self.flow,
            "logits",
            cell_size=1,
            detection_threshold=detection_threshold,
            detection_top_k=detection_top_k,
            nms_dist=nms_dist,
            border_dist=border_dist,
        )
        SJNet.add_descriptor_head_post_processing(
            self.flow,
            input_name="images",
            descriptor_head_output_name="raw_descriptors",
            prefix="",
            scale_factor=self.descriptor_scale_factor,
            normalize_descriptors=normalize_descriptors,
        )
   
    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str, # = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        positions_name: str = "positions",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        flow.define_transition(
            f"{prefix}normalized_descriptors",
            partial(
                SuperPoint.normalize_descriptors,
                scale_factor=scale_factor,
                normalize=normalize_descriptors,
            ),
            descriptor_head_output_name,
        )
        flow.define_transition(
            f"{prefix}dense_descriptors",
            SJNet.get_dense_descriptors,
            f"{prefix}normalized_descriptors",
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(
                SJNet.sparsify_descriptors,
                scale_factor=scale_factor,
                normalize_descriptors=normalize_descriptors,
            ),
            descriptor_head_output_name,
            f"positions",
        )
        flow.define_transition(
            f"{prefix}sparse_positions",
            lambda x: x,
            f"positions",
        )
        flow.define_transition(
            f"{prefix}dense_positions",
            SJNet.get_dense_positions,
            f"probability",
        )

    @staticmethod
    def get_dense_positions(probability):
        batch_size = probability.shape[0]
        device = probability.device
        dense_positions = get_dense_positions(
            probability.shape[2],
            probability.shape[3],
            device,
            batch_size=batch_size,
        )

        dense_probability = probability.reshape(probability.shape[0], -1, 1)
        dense_positions = torch.cat((dense_positions, dense_probability), dim=2)

        return dense_positions

    @staticmethod
    def get_dense_descriptors(normalized_descriptors):
        dense_descriptors = normalized_descriptors.reshape(
            normalized_descriptors.shape[0],
            normalized_descriptors.shape[1],
            -1,
        )
        dense_descriptors = dense_descriptors.permute(0, 2, 1)
        return dense_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        sparse_descriptors = []
        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            pos = pos.floor().long()

            descriptors = raw_descriptors[i, :, pos[:, 0], pos[:, 1]].T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(
                descriptors,
                scale_factor,
                normalize_descriptors,
            )

            sparse_descriptors.append(descriptors)
        return tuple(sparse_descriptors)

