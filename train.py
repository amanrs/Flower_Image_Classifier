# Imports here
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
import json 
import time
import argparse
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict


def validation():
    print("validating parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('missing: test, train or valid sub-directories')
        
ap = argparse.ArgumentParser(description='Train Neural Network')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_directory", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)

pa = ap.parse_args()
data_dir = pa.data_dir
checkpoint = pa.save_directory
l_rate = pa.learning_rate
dout = pa.dropout
power = pa.gpu
epochs = pa.epochs
arch = pa.arch

print("validating parameters")
if (power and not torch.cuda.is_available()):
    raise Exception("--gpu option enabled...but no GPU detected")
if(not os.path.isdir(data_dir)):
    raise Exception('directory does not exist!')

print("retreiving data")
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
valid_dir = data_dir + '/valid'

data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Two options for models 
if arch=="vgg16":
    model = models.vgg16(pretrained=True)
    print(model)
else:
    model = models.densenet121(pretrained=True)
    print(model)

def classify(dropout=0.5, learn_r = 0.001, arch = "vgg16"):
    
    input_data = 25088
    if arch=="vgg16":
        input_data = 25088,
    else:
        input_data = 1024,
    
    for param in model.parameters():
        param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('input', nn.Linear(input_data[0], 120)),
            ('relu1', nn.ReLU()),
            ('h1', nn.Linear(120, 90)),
            ('relu2', nn.ReLU()),
            ('h2', nn.Linear(90, 80)),
            ('relu3', nn.ReLU()),
            ('h3', nn.Linear(80, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
    return classifier

model.classifier =  classify(dout, l_rate, arch)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=l_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def neural_net(model, dataloaders, validloaders, epochs, criterion, optimizer):
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
    
    
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(dataloaders):
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                crct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for inpts, lbls in validloaders:
                        inpts, lbls = inpts.to(device), lbls.to(device)
                        logps = model.forward(inpts)
                        ps = torch.exp(logps).data
                        crct += (lbls.data == ps.max(1)[1]).sum().item()
                        total += lbls.size(0)
                        
                r_loss = running_loss/print_every 
                v_accuracy = crct/total
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(r_loss),
                      "Validation Accuracy: {}".format(v_accuracy))
                
                running_loss = 0
    print("Training done")
    
print("Training data")
neural_net(model, dataloaders, validloaders, epochs, criterion, optimizer)

print("Checking data")
def check_accuracy_on_test(testloaders):    
    correct = 0
    total = 0
    model.to('cuda:0')
    with torch.no_grad():
        for inputs, labels in testloaders:
            imgs, lbls = inputs.to('cuda'), labels.to('cuda')
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

    print('Network accuracy on test images is: %d %%' % (100 * correct / total))
    
check_accuracy_on_test(testloaders)
# TODO: Save the checkpoint 
model.class_to_idx = image_datasets.class_to_idx
model.cpu
print("saving model")
torch.save({'structure' :'vgg16',
            'epochs':epochs,
            'learning_rate': l_rate,
            'input_size': 25088,
            'output_size': 102,
            'structure': arch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx},
            'checkpoint.pth')

print("model finished!")