**Experiment Log**

The file holds all of my meaningful experimenets/training

Shared:

**Data:**  
Official Oxford-IIIT Pet `trainval` split, using stratified 80/20 train-validation split.

**Model:**  
Baseline `OxfordPetCNN`.

## Expereiment 1: Check that my full training loop works before i run a longer baseline experiment.

**Date:**  
19/05/2026

**Transforms:**  
Resize to `64×64`, `ToTensor`, normalisation with training-set mean `[0.479322, 0.446042, 0.395331]` and standard deviation `[0.255698, 0.250303, 0.258181]`.

**Training settings:**  
- Epochs: `1`
- Batch size: `32`
- Optimiser: Adam
- Learning rate: `0.001`
- Loss function: CrossEntropyLoss
- Seed: `67`
- Device: CUDA
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

**Results:**  
- Training loss: `3.5418`
- Training accuracy: `5.06%`
- Validation loss: `3.3598`
- Validation accuracy: `9.78%`
- Epoch time: `16.77 seconds`

## Experiment 3 — Baseline CNN 10-epoch run

**Goal:**  
Train the same baseline CNN for 10 epochs to see whether the baseline continues improving or starts overfitting.

**Training settings:**  
- Epochs: `10`
- Batch size: `32`
- Optimiser: Adam
- Learning rate: `0.001`
- Loss function: CrossEntropyLoss
- Seed: `67`
- Device: CUDA
- Image size: `64×64`

**Results summary:**

- Epoch 1: train loss 3.5487, train acc 4.76%, val loss 3.3592, val acc 8.70%.
- Epoch 2: train loss 3.3292, train acc 9.44%, val loss 3.2583, val acc 11.01%.
- Epoch 3: train loss 3.0971, train acc 15.08%, val loss 3.0973, val acc 13.32%.
- Epoch 4: train loss 2.8408, train acc 20.58%, val loss 3.0296, val acc 15.35%.
- Epoch 5: train loss 2.5750, train acc 27.65%, val loss 3.0467, val acc 16.17%.
- Epoch 6: train loss 2.2317, train acc 36.11%, val loss 3.1724, val acc 16.85%.
- Epoch 7: train loss 1.8283, train acc 46.57%, val loss 3.3645, val acc 14.95%.
- Epoch 8: train loss 1.3986, train acc 59.48%, val loss 3.6820, val acc 15.62%.
- Epoch 9: train loss 0.9699, train acc 71.60%, val loss 4.3277, val acc 18.75%.
- Epoch 10: train loss 0.6015, train acc 82.54%, val loss 5.0390, val acc 19.16%.

**Outcome:**  
Accepted as the longer baseline run. The best validation accuracy improved to `19.16%`.

**Interpretation:**  
The model learned the training set strongly, with training accuracy increasing from `4.76%` to `82.54%`. 
However, validation accuracy only increased to `19.16%`, and validation loss began increasing after around epoch 4. 
This suggests that the baseline CNN is overfitting, memorising the training data much faster than it generalises to unseen validation images.