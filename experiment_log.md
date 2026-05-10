**Experiment Log**

The file holds all of my meaningful experimenets/training

Shared:

**Data:**  
Official Oxford-IIIT Pet `trainval` split, using stratified 80/20 train-validation split.

**Model:**  
Baseline `OxfordPetCNN`.

**Goal:**  
Check that my full training loop works before i run a longer baseline experiment.

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