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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)




if __name__ == '__main__':
	# Top level data directory. Here we assume the format of the directory conforms
	#   to the ImageFolder structure
	data_dir = "C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\data"

	# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
	model_name = "inception"

	# Number of classes in the dataset
	num_classes = 2

	# Batch size for training (change depending on how much memory you have)
	batch_size = 8

	# Number of epochs to train for
	num_epochs = 15

	# Flag for feature extracting. When False, we finetune the whole model,
	#   when True we only update the reshaped layer params
	feature_extract = True

	inceptionClassifier = Classifier(model_name, num_classes)
	model = inceptionClassifier.initPretrainedModel(299)

	# print(model)

	dataloaders, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batch_size)
	data_transforms = DataSet.setUpDataLoaderTransformers(inputSize = 299)

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

	# Detect if we have a GPU available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

	# sgdOptimizer = Optimizer(device)
	# optimizer_ft = sgdOptimizer.optimize(model, feature_extract, 0.001, 0.9)

	# criterion = nn.CrossEntropyLoss()

	# model, hist = inceptionClassifier.train_model(model, 
	#         criterion, 
	#         optimizer_ft, 
	#         dataloaders_dict, 
	#         dataset_sizes,
	#         True)


	# torch.save({
	#         'name': 'inceptionV3FeatureExtraction',
	#         'epoch': 15,
	#         'model_state_dict': model.state_dict(),
	#         'optimizer_state_dict': optimizer_ft.state_dict(),
	#         }, 'C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\inceptionv3FeatureExtraction.pt')

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
	state = torch.load('C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\inceptionV3FeatureExtraction.pt')
	model.load_state_dict(state['model_state_dict'])
	model.cuda()
	predictions = inceptionClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batch_size = 8)

	# save predicted values
	np.savetxt('C:\\Users\\amit\\Documents\\GitHub\\cs6476-computervision-project\\trainedModels\\predictedLabelsInceptionV3.csv', predictions, fmt='%s')
