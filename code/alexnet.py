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
import numpy as np


from optimizer import Optimizer
from classifier import Classifier
from dataset import DataSet

'''
File: alexnet.py
Project: cs6476-computervision-project
File Created: October 2019
Author: Shalini Chaudhuri (you@you.you)
'''

if __name__ == '__main__':

    data_dir = 'data'
    model_name = 'alexnet'
    output_classes = 2
    feature_extract = True
    batch_size = 8
    num_epochs = 15
    
    alexnetClassifier = Classifier(model_name, output_classes)
    model = alexnetClassifier.initPretrainedModel(224)

#     print(model)

    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batch_size)
    data_transforms = DataSet.setUpDataLoaderTransformers()
    

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # sgdOptimizer = Optimizer(device)
    # optimizer_ft = sgdOptimizer.optimize(model, feature_extract, 0.001, 0.9)

    # criterion = nn.CrossEntropyLoss()

    # model, hist = alexnetClassifier.train_model(model, 
    #         criterion, 
    #         optimizer_ft, 
    #         dataloaders_dict, 
    #         dataset_sizes)


    # torch.save({
    #         'name': 'alexnetFeatureExtraction',
    #         'epoch': 15,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer_ft.state_dict(),
    #         }, './trainedModels/alexnetFeatureExtraction.pt')

#     ohist = [h.cpu().numpy() for h in hist]
#     plt.title("Validation Accuracy vs. Number of Training Epochs")
#     plt.xlabel("Training Epochs")
#     plt.ylabel("Validation Accuracy")
#     plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
#     plt.ylim((0,1.))
#     plt.xticks(np.arange(1, num_epochs+1, 1.0))
#     plt.legend()
#     plt.show()


    # testing
    state = torch.load('./trainedModels/alexnetFeatureExtraction.pt')
    model.load_state_dict(state['model_state_dict'])
    predictions = alexnetClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batch_size = 8)
#     print(len(predictions))

    # save predicted values
    np.savetxt('./trainedModels/predictedLabelsAlexNet.csv', predictions, fmt='%s')


        
