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

from utilities import save_checkpoint

def main():
    in_args = get_input_args()
    
    data_directory = in_args.data_directory
    train_directory = data_directory + '/train'
    valid_directory = data_directory + '/valid'
    test_directory = data_directory + '/test'
    
    save_checkpoint_path = in_args.save_checkpoint_path
    
    train_transforms = transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    validation_test_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_directory, transform=train_transforms),
        "validation": datasets.ImageFolder(valid_directory, transform=validation_test_transforms),
        "test": datasets.ImageFolder(test_directory, transform=validation_test_transforms)
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=128, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=128, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=128, shuffle=True)
    }
    
    model = getattr(models, in_args.arch)(pretrained=True)
    class_to_idx = image_datasets['train'].class_to_idx
    
    for param in model.parameters():
        param.requires_grad = False

    fc1_input = model.classifier[0].in_features if hasattr(model.classifier, "__getitem__") else model.classifier.in_features    
    fc1_output = in_args.hidden_units
    fc2_input = in_args.hidden_units
    fc2_output = len(class_to_idx)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(fc1_input, fc1_output)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(fc2_input, fc2_output)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.classifier = classifier
    model.to(device);

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    epochs = in_args.epochs
    steps = 0
    running_loss = 0
    print_every = 20
    for epoch in range(epochs):
        for inputs, labels in dataloaders["train"]:
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
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders["test"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")
                running_loss = 0
                model.train()

    print("Training finished : ---------------------------")
    
    save_checkpoint(save_checkpoint_path, model, optimizer, epochs, running_loss, class_to_idx)
    
if __name__== "__main__":
    main()
    