import torch
import json
from PIL import Image
from torchvision import transforms, models

##############################################
def load_checkpoint(args):
    filepath = args.checkpoint
    gpu = args.gpu
    
    if torch.cuda.is_available() and gpu:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    model = checkpoint['model']
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

##############################################
def save_checkpoint(path, model, optimizer, epochs, loss, class_to_idx):

    checkpoint = {
        'model': model,
        'input_size': 25088,
        'output_size': 102,
        'classifier': model.classifier,
        'class_to_idx': class_to_idx,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, path)
    
##############################################
def category_names(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

##############################################
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_transformed = img_transforms(img)
    
    print("Image selected: ", image)

    return img_transformed
##############################################


    