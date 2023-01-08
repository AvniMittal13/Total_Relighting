#Need to convert this cell to .py file
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from SemanticGuidedHumanMatting.model.model import HumanSegment, HumanMatting
import SemanticGuidedHumanMatting.utils
import SemanticGuidedHumanMatting.inference as inference


def Matting(image_dir,result_dir):

  model = HumanMatting(backbone='resnet50')
  model = nn.DataParallel(model).cuda().eval()
  model.load_state_dict(torch.load("\SemanticGuidedHumanMatting\pretrained\SGHM-ResNet50.pth"))

  image_list = sorted([*glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True),
                      *glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)])

  num_image = len(image_list)

  for i in range(num_image):
      image_path = image_list[i]
      image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]

      with Image.open(image_path) as img:
          img = img.convert("RGB")

      pred_alpha, pred_mask = inference.single_inference(model, img)

      output_dir = result_dir + image_path[len(image_dir):image_path.rfind('/')]
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
      save_path = output_dir + '/' + image_name + '.png'
      Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)
      return Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L')



def MattingSingleImage(upload):

    model = HumanMatting(backbone='resnet50')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model).eval()
    model.load_state_dict(torch.load("SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth"))

    image = Image.open(upload)

    with Image.open(upload) as img:
        img = img.convert("RGB")

    pred_alpha, pred_mask = inference.single_inference(model, img)

    return Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L')