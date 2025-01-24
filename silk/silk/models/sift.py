# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import cv2 as cv
import torch
import numpy as np
from silk.matching.mnn import compute_dist, mutual_nearest_neighbor

matcher = partial(
    mutual_nearest_neighbor, distance_fn=partial(compute_dist, dist_type="cosine")
)


class SIFT:
    def __init__(self, device) -> None:
        self._sift = cv.SIFT_create()
        self._device = device

    def __call__(self, images: torch.Tensor):
        # convert image to numpy
        images = images * 255
        images = images.permute(0, 2, 3, 1)
        images = images.to(torch.uint8)
        images = images.cpu().numpy()

        keypoints = []
        descriptors = []

        for image in images:
            kp, desc = self._sift.detectAndCompute(image, None)

            # normalize
            kp = torch.tensor(tuple(k.pt for k in kp), device=self._device)
            desc = torch.tensor(desc, device=self._device)

            # xy to yx
            kp = kp[:, [1, 0]]

            keypoints.append(kp)
            descriptors.append(desc)

        return tuple(keypoints), tuple(descriptors)
    
    @staticmethod
    def getpose(logits, descriptors, intrinsics):
        # logits: y,x order
        # descriptors (with image shape)
        intrinsics = intrinsics.float().cpu().numpy()
        focal = (intrinsics[0][0]+intrinsics[1][1])/2
        pp = (intrinsics[0][2], intrinsics[1][2])
        
        matching = matcher(descriptors[0], descriptors[1])
        # print(len(matching))
        # 32
        # print(len(logits), logits[0].shape)
        # 2 torch.Size([101, 3])
        # print(max(logits[0][:,2]))
        # tensor(0.8331, device='cuda:1')
        # print(max(logits[0][:,0]), max(logits[0][:,1]))
        # tensor(287.5000, device='cuda:1') tensor(1162.5000, device='cuda:1')
        # print(descriptors[0].shape)
        # torch.Size([101, 128])
		
        E, mask = cv.findEssentialMat(
            logits[1][:,:2][matching[:,1]].detach().cpu().numpy()[:,[1,0]], 
            logits[0][:,:2][matching[:,0]].detach().cpu().numpy()[:,[1,0]],
            focal=focal, pp=pp, method=cv.RANSAC, threshold=1, prob=0.999
        )
        # if E==None: exit(0)
        if E.shape[0] != 3:
            E = E[:3,:3]
        
        _, R, t, mask = cv.recoverPose(E, #return R1->2, t1->2
            logits[1][:,:2][matching[:,1]].detach().cpu().numpy()[:,[1,0]], 
            logits[0][:,:2][matching[:,0]].detach().cpu().numpy()[:,[1,0]],
            focal=focal, pp=pp
        )
        
        # (3, 3) (3, 1)
        pose = torch.from_numpy(np.concatenate([R,t], axis=1))
        pose = torch.cat([pose, torch.tensor([[0,0,0,1]])])
        # print(pose)
        return pose