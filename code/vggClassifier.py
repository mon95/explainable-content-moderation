from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from optimizer import Optimizer
from classifier import Classifier
from dataset import DataSet

from pathlib import Path

print("PyTorch Version: ",torch.__version__)


if __name__ == '__main__':
    data_dir = "C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\data"
    model_name = "vgg"
    num_classes = 2
    batch_size = 8
    num_epochs = 15
    feature_extract = True
    
    vggClassifier = Classifier(model_name, num_classes)
    model = vggClassifier.initPretrainedModel(224)
    
    dataloaders, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batch_size)
    data_transforms = DataSet.setUpDataLoaderTransformers(inputSize = 224)
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # sgdOptimizer = Optimizer(device)
    # optimizer_ft = sgdOptimizer.optimize(model, feature_extract, 0.001, 0.9)
    
    # criterion = nn.CrossEntropyLoss()
    
    # model, hist = vggClassifier.train_model(model, 
    #     criterion, 
    #     optimizer_ft, 
    #     dataloaders_dict, 
    #     dataset_sizes,is_inception=False)
    
    # torch.save({
    # 'name': 'vggFeatureExtraction',
    # 'epoch': 15,
    # 'model_state_dict': model.state_dict(),
    # 'optimizer_state_dict': optimizer_ft.state_dict(),
    # }, 'C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\vggFeatureExtraction.pt')
    
    # ohist = [h.cpu().numpy() for h in hist]
    
    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    # plt.legend()
    # plt.show()
    
    # testing
    state = torch.load('C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\vggFeatureExtraction.pt')
    model.load_state_dict(state['model_state_dict'])
    model.cuda()
    predictions = vggClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batch_size=8)
    
    # save predicted values
    np.savetxt('C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\predictedLabelsVgg.csv', predictions, fmt='%s')
    