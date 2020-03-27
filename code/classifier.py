from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import sklearn.metrics

'''
File: classfier.py
Project: cs6476-computervision-project
File Created: October 2019
Author: Shalini Chaudhuri
'''
class Classifier:

    """

        Model Name: will be used to define which pretrained model we want to use
        Feature Extract: Are we using the CNN as a feature extractor(changing only the final layer)
                        or retraining for our problem
        Num Epochs:
        Batch Size:
        OutPut Class: Binary classification, so 2
    """

    model_name = None
    output_classes = 2

    def __init__(self, model_name, output_classes = 2, batch_size = 8, num_epochs=15, feature_extract=True):
        self.model_name = model_name
        self.output_classes = output_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_extract = feature_extract
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    def train_model(self, model, criterion, optimizer, dataloaders, dataset_sizes, is_inception=False):
        since = time.time()

        best_model_weights = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        val_acc_history = []

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # outputs = model(inputs)
                        # loss = criterion(outputs, labels)
                        
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # if phase == 'train':
                    #     scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_accuracy))

        # load best model weights
        model.load_state_dict(best_model_weights)
        return model, val_acc_history


    def set_requires_grad(self, model):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def initPretrainedModel(self, inputSize):
        model = None
        input_size = 0
        if self.model_name == 'alexnet' and self.feature_extract:
            model = torchvision.models.alexnet(pretrained=True)
            self.set_requires_grad(model)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.output_classes)
            input_size = inputSize

        if self.model_name == 'inception' and self.feature_extract:
            print("Initializing model: Inception_V3")
            model = models.inception_v3(pretrained=True)
            self.set_requires_grad(model)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, self.output_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.output_classes)
            input_size = inputSize
        
        if self.model_name == "vgg" and self.feature_extract:
            model = models.vgg11_bn(pretrained=True)
            self.set_requires_grad(model)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,self.output_classes)
            input_size = inputSize
            
        if self.model_name == "resnet":
            """Resnet 18
            """
            print("Initializing to use pre-trained Resnet 18 for feature extraction...")
            model = models.resnet18(pretrained=True)
            self.set_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.output_classes)
            input_size = inputSize

        return model

    def testModel(self, dataloaders, model, classes, dataset_sizes, batch_size):
        correct = 0
        total = dataset_sizes['test']
        predictions = []

        y_actual = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(dataloaders['test'], 0):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                                
                samples = dataloaders['test'].dataset.samples[index*batch_size : index*batch_size + batch_size]
                predicted_classes = [classes[predicted[j]] for j in range(predicted.size()[0])]
                sample_names = [s[0] for s in samples]
                
                predictions.extend(list(zip(sample_names, predicted_classes)))
                # labels = labels.cpu()
                # predicted = predicted.cpu()
                y_actual.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        try:
            print(f"Accuracy (Sklearn): {sklearn.metrics.accuracy_score(y_actual, y_pred)}")
            print(f"F1-Score (Sklearn): {sklearn.metrics.f1_score(y_actual, y_pred)}")
            print(f"Precision Score: {sklearn.metrics.precision_score(y_actual, y_pred)}")
            print(f"Recall Score: {sklearn.metrics.recall_score(y_actual, y_pred)}")
            print(f"\nConfusion Matrix:\n{sklearn.metrics.confusion_matrix(y_actual, y_pred)}")
            print(f"\nClassification Report:\n{sklearn.metrics.classification_report(y_actual, y_pred)}")
        except RuntimeError:
            print("Error computing metrics: \n", RuntimeError)

        print('\n\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))


        return predictions
