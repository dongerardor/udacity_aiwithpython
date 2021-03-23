import argparse

# train.py arguments
#   --data_directory 
#   --save_checkpoint_path
#   --arch "vgg16" or "DenseNet121"
#   --learning_rate (0.01)
#   --hidden_units (512)
#   --epochs (5)
#   --gpu

# predict.py arguments
#   --/path/to/image checkpoint
#   --top_k 3
#   --category_names cat_to_name.json
#   --gpu

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', default='./flowers', help = 'Path to data. Default: ./flowers')
    parser.add_argument('--save_checkpoint_path', default='./saved/checkpoint.pth', help = 'Path to save checkpoint directory, default: ./saved/checkpoint.pth')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'densenet121'], help = 'Architecture to use. Options: vgg16 or densenet121, default vgg16')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'Set learning rate (type: float, default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default = 512, help = 'Set hidden units (type: int, default: 512)')
    parser.add_argument('--epochs', type=int, default = 5, help = 'Set epochs amount (type: int, default: 2)')
    parser.add_argument('--checkpoint', default='./saved/checkpoint.pth', help = 'Path to load checkpoint')
    parser.add_argument('--gpu', type=bool, default = True, help = 'Use gpu')
    parser.add_argument('--top_k', type=int, default=5, help = 'Set how many of the top probabilities to show')
    parser.add_argument('--image', default='flowers/test/1/image_06743.jpg', help = 'Specifies which image to test')
    parser.add_argument('--labels', default='cat_to_name.json', help = 'Set JSON file with labels')
    
    return parser.parse_args()