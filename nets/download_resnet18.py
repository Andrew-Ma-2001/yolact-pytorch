import torch
import torchvision.models as models
import os

def down_resnet18():
    # Download the pretrained weights for ResNet18
    resnet18 = models.resnet18(pretrained=True)

    # Save the weights to the specified directory
    weights_dir = 'model_data/'
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'resnet18_backbone_weights.pth')

    # Save the state_dict of the model (i.e., the weights)
    torch.save(resnet18.state_dict(), weights_path)

    import numpy as np
    # Create a test image and see the output of the model
    test_image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)

    # Convert the image to a PyTorch tensor
    test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float()

    # Add a batch dimension
    test_tensor = test_tensor.unsqueeze(0)

    # Load the model
    resnet18 = models.resnet18(pretrained=False)

    # Load the weights
    resnet18.load_state_dict(torch.load(weights_path))

    # Set the model to evaluation mode
    resnet18.eval()

    # Run the test image through the model
    output = resnet18(test_tensor)

    # Print the output shape
    print(output.shape)


    from torchsummary import summary

    # Load the resnet50 model
    resnet50 = models.resnet50(pretrained=True)

    # Print out the summary of the resnet50 model
    summary(resnet50, (3, 544, 544), device='cpu')

