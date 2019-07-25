    
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
# import train.py
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='/home/workspace/ImageClassifier/cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
data_dir = "./flowers//"

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint
category_names = pa.category_names

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

model = models.vgg16(pretrained=True)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    lr=checkpoint['learning_rate']
    if structure=="vgg16":
        model = models.vgg16(pretrained=False)
#         print(model)
    else:
        model = models.densenet121(pretrained=False)
#         print(model)
    model.classifier =  classify(checkpoint['learning_rate'])

    model.load_state_dict(checkpoint['state_dict'])
    
    return model



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
model = load_checkpoint(path)
print(model)

if (power and not torch.cuda.is_available()):
    raise Exception("--gpu option enabled...but no GPU detected")
if (number_of_outputs is None):
    top_k = 5
else:
    top_k = number_of_outputs


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image)
    
    transform_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    t_img = transform_img(img)
    
    return t_img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    t_img = process_image(image_path)
    t_img = t_img.unsqueeze_(0)
    t_img = t_img.float()
    with torch.no_grad():
        output = model.forward(t_img)
    prob = F.softmax(output.data,dim=1)
    prob = prob.topk(topk)
    print(prob)
    return prob

def classify_image(path = 'flowers/test/1/image_06743.jpg', model='checkpoint.pth'):
    plt.figure(figsize= [10,5])
    plt.subplot(1,2,1)
    index = 1
    image = process_image(path)
    axes = imshow(image, ax=plt)
    axes.title(cat_to_name[str(index)])   
    axes.show()
    probs, clss = predict(path,model)
    y = np.array(probs[0])
    x =  [cat_to_name[str(cl + 1)] for cl in np.array(clss[0])]
    N = float(len(x))
    tickLocations = np.arange(N)
    fig, ax = plt.subplots()
    ax.bar(tickLocations, y, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(x)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()
    
print("predicting image")
prediction = predict(path_image,model,top_k)
p = np.array(prediction[0])
c = np.array(prediction[1])
# hey = (prb, clsss)
cat_file = json.loads(open("cat_to_name.json").read())
# print(p[0])
i = 0
for j in range(len(p[0])):
    i = i+1
    a = str(p[0][j] * 100.) + '%'
    if (cat_file):
        b = cat_file.get(str(c[0][j]),'None')
    else:
        b = ' class {}'.format(str(c[j]))
    print("{}.{} ({})".format(i, b,a))
# for p, c in hey:
#     i = i + 1
#     p = str(round(p,4) * 100.) + '%'
    
            
