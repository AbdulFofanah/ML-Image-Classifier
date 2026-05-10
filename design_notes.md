**My notes and reasoning**
Numbers = significant milestone

Project decisions: 
1) I created a base data pipline using the official Oxford-IIIT Pet `trainval` and `test` splits, keeping the test split separate from training and validation.
2) I created a baseline CNN in my model.py file that accepted the data in teh correct shap adn output is in the correct logits for the 37 pet classes
3) I replaced the basic random 80/20 train-validation split with a stratified 80/20 split so the class distribution across the 37 pet breeds is better preserved in both training and validation.
4) Refactored my data pipline so that i could decouple the code for teh data loaded with the rest so that it could be imported to other python files down the line.
5) I created  my first training file `train.py` with separate training and validation functions. 
I kept the test set unused during training so model development is based only on training and validation performance.
Architecture choices:
1) No model architecture yet
2) used three Conv2d --> ReLU --> MaxPool2d blocks followed by Flatten, Linear(4096 --> 128), ReLU, and Linear(128 --> 37) as a simple baseline CNN architecture.
5) No architecture changes were made in Phase 3. The training script uses the existing baseline CNN from `model.py`.

Conv2d learns visual features from the image, ReLU adds non-linearity. and MaxPool2d reduces the image size.

hyperparameters:
1) I used an 80/20 train-validation split, batch size `32`, fixed seed `67`, and `num_workers=0` as simple baseline settings.
3) The baseline model uses convolution channel sizes 16, 32, and 64, 3×3 kernels with padding 1, max-pooling with kernel size 2, and a hidden classifier size of 128
3) I kept the train-validation split at 80/20 and kept the split seed as 67, but changed the split method to stratified splitting using the class labels.

transforms:
1) I resized images to `64×64`, converted them to tensors, and normalised them with mean `[0.479638, 0.448003, 0.397518]` and standard deviation `[0.254801, 0.250005, 0.258661]`. 
These values represent the mean and standard deviation of the entire training split with seed 67
2) No transform changes were made
3) Because the training split changed from random to stratified, the training mean and standard deviation should be recomputed using the new stratified training split only.
I also recomputed the mean `[0.479322, 0.446042, 0.395331]` and standard deviation `[0.255698, 0.250303, 0.258181]` with this new split.

Experiments:
1) I used `64×64` images and batch size `32` as fast baseline choices
2) first implemented a baseline one to adapt based on architectural choices as i go
5) My first test completed successfully on CUDA. It achieved training loss `3.5418`, training accuracy `5.06%`, validation loss `3.3598`, and validation accuracy `9.78%`.
it successfully loaded batches, trained for one epoch, evaluated on validation data, saved `model.pth`, and exited with code `0`.

Current environment check:
- Python version: 3.11.15
- PyTorch version: 2.11.0+cu130
- Torchvision version: 0.26.0+cu130
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU - My laptop GPU -
I upgraded the environment from CPU-only PyTorch to CUDA-enabled PyTorch so
that the local training can use my RTX 4060 Laptop GPU