import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
import torch
from util import get_model, SILK_MATCHER
from silk.models.sift import SIFT


STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

import sys
sys.path.append(r'/root/silk')

class VisualOdometry:
	def __init__(self):
		self.frame_stage = 1
		self.R = None
		self.t = np.array((0,0,0))
		self.model = get_model(default_outputs=("positions", "sparse_descriptors"))
		# self.model = get_model(default_outputs=("pose6d", "sparse_positions", "sparse_descriptors"))
		self.frames_num = 0
		self.silk_macthes_len = 0
		self.silk_macthes_len = 0
		
	def getAbsoluteScale(self, abs_gt):  #specialized for KITTI odometry dataset
		last_gtX = abs_gt[1][0][3]
		last_gtY = abs_gt[1][1][3]
		last_gtZ = abs_gt[1][2][3]
		gtX = abs_gt[0][0][3]
		gtY = abs_gt[0][1][3]
		gtZ = abs_gt[0][2][3]

		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	@torch.no_grad
	def processSecondFrame(self, img, rel_gt, abs_gt):
		positions, descriptors = self.model(img, self.intrinsics)
		print(len(positions), len(descriptors))
		print(positions[0].shape, descriptors[0].shape)
				
		exit(0)
		self.frames_num+=1
		pose = SIFT.getpose(descriptors=descriptors, logits=positions, intrinsics=self.intrinsics)
		
		self.rel_gt = rel_gt #4x4 ndarray
		self.frame_stage = STAGE_DEFAULT_FRAME 
		absolute_scale = self.getAbsoluteScale(abs_gt)
		# print(pose.shape)
		# torch.Size([4, 4])

		R_ = pose[:3,:3]
		t_ = absolute_scale*(pose[:3,3])

		
		return R_, np.expand_dims(t_, axis=1) 


	@torch.no_grad
	def processFrame(self, img, rel_gt, abs_gt):
		positions, descriptors = self.model(img, self.intrinsics)
		self.frames_num+=1

		pose = SIFT.getpose(descriptors=descriptors, logits=positions, intrinsics=self.intrinsics)
		absolute_scale = self.getAbsoluteScale(abs_gt)

		R_=pose[:3,:3]
		t_=absolute_scale*R_@(pose[:3,3])
		
		return R_, np.expand_dims(t_, axis=1) 

	def update(self, img, abs_gt, rel_gt, intrinsics):
		self.intrinsics = intrinsics
		self.focal = (float(intrinsics[0,0]) + float(intrinsics[1,1])) / 2
		self.pp = (float(intrinsics[0,2]), float(intrinsics[1,2]))
		
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			R_, t_ = self.processFrame(img, rel_gt, abs_gt)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			R_, t_ = self.processSecondFrame(img, rel_gt, abs_gt)
		return R_, t_
   


