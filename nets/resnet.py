import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNetVerSmall(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        out = []
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        out.append(x)
        x = self.layer3(x)
        out.append(x)          # 14x14
        x = self.layer4(x)
        out.append(x)          # 7x7

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1    = norm_layer(planes)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2    = norm_layer(planes)
        self.conv3  = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3    = norm_layer(planes * 4)
        self.relu   = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channels   = []
        self.inplanes   = 64
        self.norm_layer = norm_layer

        # 544, 544, 3 -> 272, 272, 64
        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = norm_layer(64)
        self.relu       = nn.ReLU(inplace=True)
        # 272, 272, 64 -> 136, 136, 64
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers     = nn.ModuleList()
        # 136, 136, 64 -> 136, 136, 256
        self._make_layer(block, 64, layers[0])
        # 136, 136, 256 -> 68, 68, 512
        self._make_layer(block, 128, layers[1], stride=2)
        # 68, 68, 512 -> 34, 34, 1024
        self._make_layer(block, 256, layers[2], stride=2)
        # 34, 34, 1024 -> 17, 17, 2048
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion)
            )

        layers          = [block(self.inplanes, planes, stride, downsample, self.norm_layer)]
        self.inplanes   = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)[-3:]

    def init_backbone(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)



if __name__ == "__main__":
    import numpy as np
    # Instantiate the ResNet with 4 layers each having 2 blocks
    model = ResNetVerSmall(BasicBlock,[3, 4, 6, 3])
    # model = ResNet([3, 4, 6, 3])
    
    # # Load the pre-trained weights from torch resnet18
    pretrained_weights = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).state_dict()
    
    # # Prepare the state_dict to load into our model
    # state_dict = {}
    # for k, v in pretrained_weights.items():
    #     if 'fc' not in k:  # Exclude the fully connected layer
    #         state_dict[k] = v
    
    # # Load the weights into our model
    # model.load_state_dict(state_dict, strict=False)
    
    # Save the two model state dict size and name to two txt files
    # with open('model_state_dict.txt', 'w') as f:
    #     print("Model's state_dict:", file=f)
    #     for param_tensor in model.state_dict():
    #         print(param_tensor, "\t", model.state_dict()[param_tensor].size(), file=f)
    
    # with open('pretrained_state_dict.txt', 'w') as f:
    #     print("Pretrained model's state_dict:", file=f)
    #     for param_tensor in pretrained_weights:
    #         print(param_tensor, "\t", pretrained_weights[param_tensor].size(), file=f)

    # Use torch summary to print the model summary
    from torchsummary import summary
    summary(model, (3, 544, 544), device='cpu')

    # Test for an image
    image = np.random.rand(1, 3, 544, 544)
    image = torch.from_numpy(image).float()
    output = model(image)
    
    # Print the output size
    for item in output:
        print(item.shape)

    # Save the model
    # torch.save(model.state_dict(), 'model_data/resnet34_backbone_weights.pth')