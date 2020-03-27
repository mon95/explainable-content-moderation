import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision.models import vgg11_bn
from torchvision.models import densenet201
from torchvision.models import densenet121
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import gradcam_utils
import os

def fetch_and_fit_model():
    device = torch.device('cpu')
    model_ft = vgg11_bn(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, 2) 
    state = torch.load('./trainedModels/vggFeatureExtraction.pt', map_location=device)
    model_ft.load_state_dict(state['model_state_dict'])
    return model_ft


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', dest='input_directory', action='store', default=os.getcwd(), required=False, help='The root directory containing the image. Use "./" for current directory')
    # args = parser.parse_args()

    # input_directory = args.input_directory
    input_directory = './data/test/violent'
    output_directory = './gradCAM_augmented/violent'
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(root=os.getcwd(), transform=transform)
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
    images = []
    for filename in os.listdir(os.path.join(input_directory, '')):
        if filename.endswith('.jpg'):
            images.append(filename)

    model_gradcam = fetch_and_fit_model()
    ds, img, scores, label = gradcam_utils.run_inference(model_gradcam, dataloader)
    heatmap = gradcam_utils.get_grad_cam(ds, img, scores, label)
    for image in images:
        gradcam_utils.replace_area(input_directory, heatmap, image, output_directory)
    print(scores)
    print(label)
