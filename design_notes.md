**My notes and reasoning**
Numbers = significant milestone

Project decisions: 
1) I created a base data pipline using the official Oxford-IIIT Pet `trainval` and `test` splits, keeping the test split separate from training and validation.

Architecture choices:
1) No model architectur yet

hyperparameters:
1) I used an 80/20 train-validation split, batch size `32`, fixed seed `67`, and `num_workers=0` as simple baseline settings.

transforms:
1) I resized images to `64×64`, converted them to tensors, and normalised them with mean `[0.479638, 0.448003, 0.397518]` and standard deviation `[0.254801, 0.250005, 0.258661]`. 
These values represent the mean and standard deviation of the entire training split with seed 67

Experiments I accpeted or rejeceted:
1) I used `64×64` images and batch size `32` as fast baseline choices

Current environment check:
- Python version: 3.11.15
- PyTorch version: 2.11.0+cu130
- Torchvision version: 0.26.0+cu130
- CUDA available: True
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU - My laptop GPU -
I upgraded the environment from CPU-only PyTorch to CUDA-enabled PyTorch so
that the local training can use my RTX 4060 Laptop GPU