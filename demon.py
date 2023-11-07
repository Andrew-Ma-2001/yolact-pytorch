import os
import os.path as osp
import torch
import torch.nn as nn

from nets.yolact import Yolact

from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# old_model = Yolact(num_classes=81, pretrained=False, train_mode=False)

# old_model_path = 'model_data/yolact_weights_coco.pth'
# new_model_path= 'logs/resnet50/best_epoch_weights.pth'



# # Save the new state dict
# torch.save(new_state_dict, 'new_state_dict.pth')

from yolact import YOLACT

# model = YOLACT(model_path = 'new_state_dict.pth', class_path = 'model_data/person_classes.txt', train_mode=False)



old_model_path = 'model_data/yolact_weights_coco.pth'

# Load the model
old_model = torch.load(old_model_path, map_location=device)

new_state_dict = OrderedDict()
# Modify the keys in the state dict
for k, v in old_model.items():
    if 'prediction_layers' in k:
        for i in range(3, 8):
            new_k = k.replace('prediction_layers', f'prediction_layer_P{i}')
            new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v

# Add in prediction layers seperated in state dict
# Save the new state dict
import argparse
import yaml


parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--config', type=str, default='config/resnet50.yaml', help='Path to the configuration file')
parser.add_argument('--map_mode', type=int, default=3, help='Map mode for evaluation')
args = parser.parse_args()
# Load hyperparameters from yaml file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

classes_path = config['general']['classes_path']
model_path = config['saving']['save_dir'] + '/best_epoch_weights.pth'
Image_dir = config['data']['val_image_path']
Json_path = config['data']['val_annotation_path']
map_out_path    = config['saving']['save_dir']
save_path = osp.join(map_out_path, 'original_val_result')

resnet50 = Yolact(num_classes=2, pretrained=False, train_mode=False)
resnet50.load_state_dict(new_state_dict, strict=False)

torch.save(resnet50.state_dict(), 'demon.pth')




