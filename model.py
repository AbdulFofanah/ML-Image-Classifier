# Convolutional Neural Network model

import torch
import torch.nn as nn
from torchinfo import summary

class OxfordPetCNN(nn.Module):  # Sets the neural network class with 37 class scores
    def __init__(self, num_classes=37):
        super().__init__() # initialise the PyTorch nn.Module parent class

        self.features = nn.Sequential( #run in this specific order
            nn.Conv2d(
                in_channels=3, #RGB
                out_channels=16, # number of feature maps (filters learning from patterns)
                kernel_size=3, # 3x3 patch of pixels
                padding=1 # keeps the output the same size as the input
            ),
            nn.BatchNorm2d(16),  # normalises the 16 feature maps to make training more stable
            nn.ReLU(), # all negatives to 0
            nn.MaxPool2d(kernel_size=2), # highest value of 2x2 blocks

            nn.Conv2d(
                in_channels=16, # number of feature maps (filters learning from patterns)
                out_channels=32, # look for twice as many patterns as before
                kernel_size=3, # 3x3 patch of pixels
                padding=1 # keeps the output the same size as the input
            ),
            nn.BatchNorm2d(32),  # normalises the 32 feature maps to make training more stable
            nn.ReLU(), # all negatives to 0
            nn.MaxPool2d(kernel_size=2),  # highest value of 2x2 blocks

            nn.Conv2d(
                in_channels=32, # number of feature maps (filters learning from patterns)
                out_channels=64, # look for twice as many patterns as before
                kernel_size=3,  # 3x3 patch of pixels
                padding=1 # keeps the output the same size as the input
            ),
            nn.BatchNorm2d(64),  # normalises the 64 feature maps to make training more stable
            nn.ReLU(), # all negatives to 0
            nn.MaxPool2d(kernel_size=2)  # highest value of 2x2 blocks
        )

        self.classifier = nn.Sequential( #run in this specific order
            nn.Flatten(), # 3d to 1d by multiplying the last 3 dimensions of the tensor together
            nn.Linear(64 * 8 * 8, 128), # combines the learned inputs to neurons with heir own weights
            nn.ReLU(),
            nn.Linear(128, num_classes) # maps compressed signals to pet breeds and generate logits
        )

    def forward(self, input_tensor): # process the input tensor with the features and classifier
        input_tensor = self.features(input_tensor)
        input_tensor = self.classifier(input_tensor)
        return input_tensor

def main():
    basic_model = OxfordPetCNN(num_classes=37)
    temp_batch = torch.randn(32, 3, 64, 64)

    basic_model.eval() # evaluation mode
    with torch.no_grad(): # disable gradient tracking
        temp_batch_output = basic_model(temp_batch)

    print(f"Input batch shape: {temp_batch.shape}\n")
    print(f"Output batch shape: {temp_batch_output.shape}\n")

    summary(basic_model, input_size=(32, 3, 64, 64))


if __name__ == "__main__":
    main()