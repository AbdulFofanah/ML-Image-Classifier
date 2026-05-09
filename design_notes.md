**My notes and reasoning**
Numbers = significant milestone

Project decisions: 
1) I created a base data pipline using the official Oxford-IIIT Pet `trainval` and `test` splits, keeping the test split separate from training and validation.
2) I created a baseline CNN in my model.py file that accepted the data in teh correct shap adn output is in the correct logits for the 37 pet classes
Architecture choices:
1) No model architecture yet
2) used three Conv2d --> ReLU --> MaxPool2d blocks followed by Flatten, Linear(4096 --> 128), ReLU, and Linear(128 --> 37) as a simple baseline CNN architecture.

Conv2d learns visual features from the image, ReLU adds non-linearity. and MaxPool2d reduces the image size.

hyperparameters:
1) I used an 80/20 train-validation split, batch size `32`, fixed seed `67`, and `num_workers=0` as simple baseline settings.
3) The baseline model uses convolution channel sizes 16, 32, and 64, 3×3 kernels with padding 1, max-pooling with kernel size 2, and a hidden classifier size of 12

transforms:
1) I resized images to `64×64`, converted them to tensors, and normalised them with mean `[0.479638, 0.448003, 0.397518]` and standard deviation `[0.254801, 0.250005, 0.258661]`. 
These values represent the mean and standard deviation of the entire training split with seed 67

2) No transform changes were made

Experiments:
1) I used `64×64` images and batch size `32` as fast baseline choices
2) first implemented a baseline one to adapt based on architectural choices as i go

Current environment check:
- Python version: 3.11.15
- PyTorch version: 2.11.0+cu130
- Torchvision version: 0.26.0+cu130
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU - My laptop GPU -
I upgraded the environment from CPU-only PyTorch to CUDA-enabled PyTorch so
that the local training can use my RTX 4060 Laptop GPU