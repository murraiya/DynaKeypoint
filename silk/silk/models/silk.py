# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from silk.backbones.depthpro.SJNet import SJNet
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.flow import AutoForward, Flow
from silk.models.sift import SIFT
from silk.losses.sfmlearner.sfm_loss import ones_like_loss, photometric_reconstruction_loss
from silk.matching.mnn import (
    compute_dist,
    double_softmax_distance,
    match_descriptors,
    mutual_nearest_neighbor,
)
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.transforms.abstract import MixedModuleDict, NamedContext, Transform


def matcher(
    postprocessing="none",
    threshold=1.0,
    temperature=0.1,
    return_distances=False,
):
    if postprocessing == "none" or postprocessing == "mnn":
        return partial(mutual_nearest_neighbor, return_distances=return_distances)
    elif postprocessing == "ratio-test":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_ratio=threshold),
            distance_fn=partial(compute_dist, dist_type="cosine"),
            return_distances=return_distances,
        )
    elif postprocessing == "double-softmax":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_distance=threshold),
            distance_fn=partial(double_softmax_distance, temperature=temperature),
            return_distances=return_distances,
        )

    raise RuntimeError(f"postprocessing {postprocessing} is invalid")


class SiLKBase(
    OptimizersHandler,
    AutoForward,
    StateDictRedirect,
    pl.LightningModule,
):
    def __init__(
        self,
        optimizer_spec: Optional[Spec] = None,
        image_aug_transform: Optional[Transform] = None,
        **kwargs,
    ):
        pl.LightningModule.__init__(self, **kwargs)
        OptimizersHandler.__init__(self, optimizer_spec)  # see below

        self.predicted_pose = None

        self.model = SJNet()
        state = MixedModuleDict(
            {
                "model": self.model,
            }
        )
        self.sift = SIFT(device="cuda:1")
        StateDictRedirect.__init__(self, state)
        AutoForward.__init__(self, Flow("batch", "use_image_aug"), "loss")

        for name, param in self.model.named_parameters():
            # if name in ['model.head.4.weight', 'model.head.4.bias'] or name.split('_')[0]=='model.kpt' or name.split('_')[0]=='model.desc':
            if name.split('_')[0]=='model.kpt' or name.split('_')[0]=='model.desc':
                
                if name.split('.')[-1]=='weight':
                    torch.nn.init.xavier_normal_(param, gain=1.0)
                pass

            else: param.requires_grad = False

        self._image_aug_transform = image_aug_transform

# for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def _check_batch(self, batch):
        ensure_is_instance(batch, NamedContext)
        
        batch.ensure_exists("images")
        batch.ensure_exists("images_gray")
        batch.ensure_exists("rel_pose")
        batch.ensure_exists("intrinsics")
        images_ori = batch["images"][0].to(self.device)
        images_gray = batch["images_gray"][0].to(self.device)
        # print(images_ori.shape, images_gray.shape)
        # torch.Size([2, 3, 370, 1226]) torch.Size([2, 1, 370, 1226])

        intrinsics = torch.from_numpy(batch["intrinsics"][0][0]).to(self.device)
        # print(batch["rel_pose"])
        # print(batch["rel_pose"][0])
        # print(batch["rel_pose"][0][0])
        rel_pose = torch.stack([torch.from_numpy(batch["rel_pose"][0][0]), torch.from_numpy(batch["rel_pose"][0][1])])
        
        # print(rel_pose.shape)
        # 2,4,4
        
        return images_ori, images_gray, rel_pose, intrinsics # 0~1 3channel image and torch.intrinsic
    

    def _init_loss_flow(
        self,
        images,
        images_gray,
        rel_pose,
        intrinsics,
    ):
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            images,
            "use_image_aug",
        )
        self.flow.define_transition(
            ("depthpro", "top_k_logits", "descs", "probability", "normalized_descriptors"),
            self.model.forward_flow,
            outputs=Flow.Constant(("depths", "positions", "sparse_descriptors", "probability", "normalized_descriptors")),
            images = "augmented_images",
            intrinsics = intrinsics,
        )
        self.flow.define_transition(
            ("sift_kpt", "sift_desc"),
            self.sift,
            "images_gray",
        )

        self.flow.define_transition(
            "sift_pose",
            SIFT.getpose,
            "sift_kpt",
            "sift_desc",
            intrinsics,
        )
        
        
        self.flow.define_transition(
            "pose",
            SIFT.getpose,
            "top_k_logits",
            "descs",
            intrinsics,
        )
        self.flow.define_transition(
            "logit_loss",
            ones_like_loss,
            "probability"
        )
        self.flow.define_transition(
            ("photo_loss", "desc_loss", "loss__"),
            photometric_reconstruction_loss,
            "sift_pose",
            images_gray,
            "depthpro",
            intrinsics,
            "probability",
            "normalized_descriptors",
            rel_pose,
        )
        
        
        self._loss_fn = self.flow.with_outputs(
            (
                "photo_loss",
                "desc_loss",
                "logit_loss",
                "loss__"
            )
        )

    # @property
    # def model(self):
    #     return self.silk_model

    # def model_forward_flow(self, *args, **kwargs):
    #     return self.silk_model.forward_flow(*args, **kwargs)

    def _apply_pe(self, descriptors_0, descriptors_1, descriptors_shape):
        if not self._pe:
            return descriptors_0, descriptors_1
        _0 = torch.zeros((1,) + descriptors_shape[1:], device=descriptors_0.device)
        pe = self._pe(_0)
        pe = self._img_to_flat(pe)
        pe = pe * self.model.descriptor_scale_factor

        return descriptors_0 + pe, descriptors_1 + pe

    def _contextualize(self, descriptors_0, descriptors_1, descriptors_shape=None):
        if self._contextualizer is None:
            return descriptors_0, descriptors_1

        spatial_shape = False
        if not descriptors_shape:
            spatial_shape = True
            assert descriptors_0.ndim == 4
            assert descriptors_1.ndim == 4

            descriptors_shape = descriptors_0.shape
            descriptors_0 = self._img_to_flat(descriptors_0)
            descriptors_1 = self._img_to_flat(descriptors_1)

        assert descriptors_0.ndim == 3
        assert descriptors_1.ndim == 3

        descriptors_0 = descriptors_0.detach()
        descriptors_1 = descriptors_1.detach()

        descriptors_0, descriptors_1 = self._apply_pe(
            descriptors_0, descriptors_1, descriptors_shape
        )

        descriptors_0, descriptors_1 = self._contextualizer(
            descriptors_0, descriptors_1
        )

        if spatial_shape:
            descriptors_0 = self._flat_to_img(descriptors_0, descriptors_shape)
            descriptors_1 = self._flat_to_img(descriptors_1, descriptors_shape)

        return descriptors_0, descriptors_1

    def _aug_images(self, images, use_image_aug):
        if use_image_aug:
            images = self._image_aug_transform(images)
        return images

    def _split_descriptors(self, descriptors):
        desc_0 = SiLKBase._img_to_flat(descriptors[0::2])
        desc_1 = SiLKBase._img_to_flat(descriptors[1::2])
        return desc_0, desc_1

    def _split_logits(self, logits):
        logits_0 = SiLKBase._img_to_flat(logits[0::2]).squeeze(-1)
        logits_1 = SiLKBase._img_to_flat(logits[1::2]).squeeze(-1)
        return logits_0, logits_1

    @staticmethod
    def _img_to_flat(x):
        # x : BxCxHxW
        batch_size = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batch_size, channels, -1)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def _flat_to_img(x, shape):
        # x : BxNxC
        assert len(shape) == 4
        assert shape[0] == x.shape[0]
        assert shape[1] == x.shape[2]

        x = x.permute(0, 2, 1)
        x = x.reshape(shape)
        return x

    def _total_loss(self, mode, batch, use_image_aug: bool):
        
        # recon_loss_ = []
        # actx_desc_loss_ = []
        # keypt_loss_ = []
        # precision_ = []
        # recalll_ = []
        
        loss_1, loss_2, loss_3, loss_4 = self._loss_fn(batch, use_image_aug)
        print(loss_1, loss_2, loss_3)
        # tensor(nan, device='cuda:1') tensor(-1.1683, device='cuda:1') tensor(0.6931, device='cuda:1')

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(sj.shape)
        # logit: torch.Size([2, 1, 370, 1226])

        # recon_loss, actx_desc_loss, keypt_loss, precision, recall = \
        # self._loss_fn(
        #     batch, use_image_aug
        # )
        # recon_loss_.append(recon_loss)
        # actx_desc_loss_.append(actx_desc_loss)
        # keypt_loss_.append(keypt_loss)
        # precision_.append(precision)
        # recalll_.append(recall)
        
        # recon_loss_ = sum(recon_loss_) / self.crop_iter
        # actx_desc_loss_ = sum(actx_desc_loss_) / self.crop_iter
        # keypt_loss_ = sum(keypt_loss_) / self.crop_iter
        # precision_ = sum(precision_) / self.crop_iter
        # recalll_ = sum(recalll_) / self.crop_iter
        
        # print("nan check")
        # if math.isnan(actx_desc_loss):
        #     print("actx_desc_loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # if math.isnan(actx_desc_loss):
        #     print("ctx_desc_loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # f1 = (2 * precision * recall) / (precision + recall)

        # loss = ctx_desc_loss + actx_desc_loss + 10*recon_loss + 10*pose_loss #+ 10*recon_loss
        
        
        # loss = recon_loss_+ actx_desc_loss_ + keypt_loss_        

        # # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # print(loss_2.shape)
        loss = 3*loss_1+loss_2+loss_3
        self.log(f"{mode}.total.loss", loss)
        self.log(f"{mode}.photo.loss", 3*loss_1)
        self.log(f"{mode}.desc.loss", loss_2)
        self.log(f"{mode}.logit.loss", loss_3)

        # self.log(f"{mode}.total.loss", loss)
        # # self.log(f"{mode}.pose.loss", 10*pose_loss)
        # self.log(f"{mode}.recon.loss", recon_loss_)
        # self.log(f"{mode}.acontextual.descriptors.loss", actx_desc_loss_)
        # self.log(f"{mode}.keypoints.loss", keypt_loss_)
        # self.log(f"{mode}.precision", precision_)
        # self.log(f"{mode}.recall", recalll_)
        # if (self._ghost_sim is not None) and (mode == "train"):
        #     self.log("ghost.sim", self._ghost_sim)
        # return loss

        return loss
    
    def training_step(self, batch, batch_idx):
        
        return self._total_loss(
            "train",
            batch,
            use_image_aug=True,
            # use_image_aug=False,
        )

    def validation_step(self, batch, batch_idx):
        return self._total_loss(
            "val",
            batch,
            use_image_aug=False,
        )


class SiLKRandomHomographies(SiLKBase):
    def __init__(
        self,
        optimizer_spec: Union[Spec, None] = None,
        image_aug_transform: Union[Transform, None] = None,
        training_random_homography_kwargs: Union[Dict[str, Any], None] = None,
        **kwargs,
    ):
        SiLKBase.__init__(
            self,
            optimizer_spec,
            image_aug_transform,
            **kwargs,
        )
        # homographic sampler arguments
        self._training_random_homography_kwargs = (
            {}
            if training_random_homography_kwargs is None
            else training_random_homography_kwargs
        )

        self.flow.define_transition(
            ("images", "images_gray", "rel_pose", "intrinsics"),
            # ("crop_images", "crop_point", "intrinsics", "depth_map_1", "depth_map_2", "original_shape", "original_images"),
            self._check_batch, 
            "batch"
        )
        self._init_loss_flow(
            "images",
            "images_gray", 
            "rel_pose",
            "intrinsics",
        )



