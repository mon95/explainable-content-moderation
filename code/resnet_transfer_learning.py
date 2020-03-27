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

from classifier import Classifier
from optimizer import Optimizer
from dataset import DataSet

if __name__ == '__main__':

    data_dir = '../data'
    model_name = 'resnet'
    output_classes = 2
    feature_extract = True
    batch_size = 4
    num_epochs = 15

    resnetClassifier = Classifier(model_name, output_classes, batch_size, num_epochs)
    model = resnetClassifier.initPretrainedModel(224)

    dataloaders, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batch_size)
    data_transforms = DataSet.setUpDataLoaderTransformers()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sgdOptimizer = Optimizer(device)
    optimizer_ft = sgdOptimizer.optimize(model, feature_extract, 0.001, 0.9)

    criterion = nn.CrossEntropyLoss()

    model, hist = resnetClassifier.train_model(model,
                                                criterion,
                                                optimizer_ft,
                                                dataloaders_dict,
                                                dataset_sizes)

    torch.save({
        'name': 'resnet_feature_extraction_4',
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    }, '../trainedModels/resnetFeatureExtraction4.pt')

    # testing
    state = torch.load('../trainedModels/resnetFeatureExtraction4.pt')
    model.load_state_dict(state['model_state_dict'])
    predictions = resnetClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batch_size=4)

    # save predicted values
    np.savetxt('../trainedModels/predictedLabelsResNet4.csv', predictions, fmt='%s')





