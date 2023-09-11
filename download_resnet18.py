import torch
import torchvision.models as models
import os

# Download the pretrained weights for ResNet18
resnet18 = models.resnet18(pretrained=True)

# Save the weights to the specified directory
weights_dir = 'model_data/'
os.makedirs(weights_dir, exist_ok=True)
weights_path = os.path.join(weights_dir, 'resnet18_backbone_weights.pth')

# Save the state_dict of the model (i.e., the weights)
torch.save(resnet18.state_dict(), weights_path)
