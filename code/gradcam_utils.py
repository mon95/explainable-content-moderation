'''
File: gradCAM_util.py
Project: code
File Created: November 2019
Author: Shalini Chaudhuri (you@you.you)
'''
import torch
import argparse
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision.models import densenet201
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class VGG(nn.Module):
    def __init__(self, model):
        # changing because we are using vgg11_bn
        # need to register to the 27t layer RELU before maxpool2d
        super(VGG, self).__init__()
        self.vgg = model
        self.features_conv = self.vgg.features[:28]
        # Use the same params as the pretrained model
        # to verify arch of pretrained print(model)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, 
                                     padding=0, dilation=1, ceil_mode=False)
        self.classifier = self.vgg.classifier
        self.gradients = None
    

    def activations_hook(self, grad):
        self.gradients = grad
        

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    

    def get_activations_gradient(self):
        return self.gradients
    

    def get_activations(self, x):
        return self.features_conv(x)


def run_inference(model, dataloader):
    vgg = VGG(model)
    vgg.eval()
    img, _ = next(iter(dataloader))
    scores = vgg(img)
    label = torch.argmax(scores)
    return vgg, img, scores, label


def get_grad_cam(vgg, img, scores, label):
  # labels to check the length of tensor
    scores[:, label].backward(retain_graph=True)
    gradients = vgg.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = vgg.get_activations(img).detach()
    # 512 is size of conv layer preceding it
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    #plt.matshow(heatmap.squeeze())
    return heatmap

def render_superimposition(root_dir, heatmap, image, output_directory=''):
    print(os.path.join(root_dir, '', image))
    img = cv2.imread(os.path.join(root_dir, '', image))
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(output_directory + '/superimposed_' + image, superimposed_img)
    cv2.imshow('output', superimposed_img)


def replace_area(root_dir, heatmap, image, output_directory='', denominator=4, kernel=(25,25)):
    print(os.path.join(root_dir, '', image))
    img = cv2.imread(os.path.join(root_dir, '', image))
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    maximum = np.max(heatmap)
    threshold = maximum/denominator

    # Use this heatmap after thresholding as a mask
    blurred_image = np.copy(img)
    blurred_image[heatmap > threshold] = img[heatmap > threshold]
    blurred_image = cv2.blur(blurred_image, ksize=kernel) # blur only that part

    blurred_image[heatmap <= threshold] = img[heatmap <= threshold]

    cv2.imwrite(output_directory + '/blurred_' + image, blurred_image)