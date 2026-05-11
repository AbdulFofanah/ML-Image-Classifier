# Convolutional Neural Network model

import torch
import torch.nn as nn
from torchinfo import summary

class OxfordPetCNN(nn.Module):  # Sets the neural network class with 37 class scores
    def __init__(self, num_classes=37):
        super().__init__() # initialise the PyTorch nn.Module parent class

        # Visual Geometry Group style architecture
        self.features = nn.Sequential(  # run in this specific order
            # 2 convolution layers before pooling for more details 1/3
            nn.Conv2d(
                in_channels=3,  # RGB
                out_channels=32,  # number of feature maps/filters learning low-level patterns
                kernel_size=3,  # 3x3 patch of pixels
                padding=1  # keeps the output the same height and width as the input
            ),
            nn.BatchNorm2d(32),  # normalises the 32 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0

            nn.Conv2d(
                in_channels=32,  # takes the 32 feature maps from the previous convolution
                out_channels=32,  # learns another 32 refined feature maps before pooling
                kernel_size=3,  # 3x3 patch of features
                padding=1  # keeps the output the same height and width
            ),
            nn.BatchNorm2d(32),  # normalises the 32 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0
            nn.MaxPool2d(kernel_size=2),  # keeps the highest value in each 2x2 block and halves the image size

            # 2 convolution layers before pooling for more details 2/3
            nn.Conv2d(
                in_channels=32,  # takes the 32 feature maps from block 1
                out_channels=64,  # increases to 64 feature maps to learn more complex patterns
                kernel_size=3,  # 3x3 patch of features
                padding=1  # keeps the output the same height and width
            ),
            nn.BatchNorm2d(64),  # normalises the 64 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0

            nn.Conv2d(
                in_channels=64,  # takes the 64 feature maps from the previous convolution
                out_channels=64,  # learns another 64 refined feature maps before pooling
                kernel_size=3,  # 3x3 patch of features
                padding=1  # keeps the output the same height and width
            ),
            nn.BatchNorm2d(64),  # normalises the 64 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0
            nn.MaxPool2d(kernel_size=2),  # keeps the highest value in each 2x2 block and halves the image size

            # 2 convolution layers before pooling for more details 3/3
            nn.Conv2d(
                in_channels=64,  # takes the 64 feature maps from block 2
                out_channels=128,  # increases to 128 feature maps to learn higher-level patterns
                kernel_size=3,  # 3x3 patch of features
                padding=1  # keeps the output the same height and width
            ),
            nn.BatchNorm2d(128),  # normalises the 128 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0

            nn.Conv2d(
                in_channels=128,  # takes the 128 feature maps from the previous convolution
                out_channels=128,  # learns another 128 refined feature maps before pooling
                kernel_size=3,  # 3x3 patch of features
                padding=1  # keeps the output the same height and width
            ),
            nn.BatchNorm2d(128),  # normalises the 128 feature maps to make training more stable
            nn.ReLU(),  # turns all negative values to 0
            nn.MaxPool2d(kernel_size=2),  # keeps the highest value in each 2x2 block and halves the image size

            nn.AdaptiveAvgPool2d((4, 4))  # forces the final feature map to [batch, 128, 4, 4]
        )

        self.classifier = nn.Sequential( # run in this specific order
            nn.Flatten(), # 3d to 1d by multiplying the last 3 dimensions of the tensor together
            nn.Linear(128 * 4 * 4, 128), # turns the flattened image features into 128 learned scores for the classifier
            nn.ReLU(),
            nn.Linear(128, num_classes) # maps compressed signals to pet breeds and generates logits
        )

    def forward(self, input_tensor): # process the input tensor with the features and classifier
        input_tensor = self.features(input_tensor)
        input_tensor = self.classifier(input_tensor)
        return input_tensor

def main():
    basic_model = OxfordPetCNN(num_classes=37)
    temp_batch = torch.randn(32, 3, 96, 96)

    basic_model.eval() # evaluation mode
    with torch.no_grad(): # disable gradient tracking
        temp_batch_output = basic_model(temp_batch)

    print(f"Input batch shape: {temp_batch.shape}\n")
    print(f"Output batch shape: {temp_batch_output.shape}\n")

    summary(basic_model, input_size=(32, 3, 96, 96))


if __name__ == "__main__":
    main()