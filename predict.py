import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

from args import get_input_args
import json

from PIL import Image

from utilities import load_checkpoint, process_image, category_names

def predict(image_path, model, gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    inputs = image.unsqueeze_(0).to(device)

    with torch.no_grad():
        output = model.forward(inputs)
        tensor_probs, tensor_labels = torch.topk(output, topk)
        top_probs = tensor_probs.exp()
    
    idx_to_class = {model.class_to_idx[key]: key for key in model.class_to_idx}
    classes = list()
    
    probs = top_probs.tolist()[0]
    labels = tensor_labels.tolist()[0]
    
    for label in labels:
        classes.append(idx_to_class[label])

    return probs, classes


def main():
    in_args = get_input_args()
    model = load_checkpoint(in_args)
    
    gpu = in_args.gpu
    image_path = in_args.image
    topk = in_args.top_k
    labels_path = in_args.labels
    
    cat_to_name = category_names(labels_path)
    probs, classes = predict(image_path, model, gpu, topk)
    labels = [cat_to_name[str(index)] for index in classes]

    output = []
    for i in range(0, len(labels)):
        output.append([labels[i], classes[i], probs[i]])
    
    l1, l2 = len(output), len(output[0])
    print(pd.DataFrame(output, index=['']*l1, columns=['']*l2).T)
    
if __name__== "__main__":
    main()