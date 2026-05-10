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

## Experiment 1 - Baseline CNN 10-epoch run

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

## Experiment 3 - Learning rate adjustment
**Results summary - Learning rate `1e-3` - `(lr=0.001, 10 epochs)`**

- Epoch 1: train loss 3.5477, train acc 4.99%, val loss 3.3589, val acc 7.74%.
- Epoch 2: train loss 3.3246, train acc 10.02%, val loss 3.2488, val acc 11.01%.
- Epoch 3: train loss 3.0844, train acc 15.01%, val loss 3.1576, val acc 11.82%.
- Epoch 4: train loss 2.8419, train acc 20.45%, val loss 3.0550, val acc 13.86%.
- Epoch 5: train loss 2.5662, train acc 27.41%, val loss 3.0793, val acc 13.59%.
- Epoch 6: train loss 2.2128, train acc 36.55%, val loss 3.1957, val acc 17.26%.
- Epoch 7: train loss 1.7917, train acc 48.44%, val loss 3.5054, val acc 16.44%.
- Epoch 8: train loss 1.3304, train acc 60.84%, val loss 3.8438, val acc 16.17%.
- Epoch 9: train loss 0.8656, train acc 74.76%, val loss 4.5143, val acc 17.93%.
- Epoch 10: train loss 0.4854, train acc 86.07%, val loss 5.3296, val acc 17.12%.

**Best validation accuracy:** 17.93% at epoch 9.

**Results summary - Learning rate `5e-4` - `(lr=0.0005, 10 epochs)`**

- Epoch 1: train loss 3.5683, train acc 3.74%, val loss 3.4198, val acc 8.56%.
- Epoch 2: train loss 3.3747, train acc 8.49%, val loss 3.3063, val acc 8.70%.
- Epoch 3: train loss 3.1974, train acc 13.01%, val loss 3.1920, val acc 9.10%.
- Epoch 4: train loss 2.9695, train acc 18.78%, val loss 3.0712, val acc 13.18%.
- Epoch 5: train loss 2.7123, train acc 24.18%, val loss 3.0678, val acc 15.08%.
- Epoch 6: train loss 2.4456, train acc 31.59%, val loss 3.1359, val acc 16.85%.
- Epoch 7: train loss 2.1538, train acc 38.49%, val loss 3.1886, val acc 18.48%.
- Epoch 8: train loss 1.8407, train acc 47.52%, val loss 3.4025, val acc 17.66%.
- Epoch 9: train loss 1.4910, train acc 56.25%, val loss 3.5971, val acc 16.98%.
- Epoch 10: train loss 1.1460, train acc 66.95%, val loss 4.0480, val acc 18.48%.

**Best validation accuracy:** 18.48%, reached at epochs 7 and 10.

**Results summary - Learning rate `1e-4` - `(lr=0.0001, 10 epochs)`**

- Epoch 1: train loss 3.6044, train acc 3.16%, val loss 3.5795, val acc 3.12%.
- Epoch 2: train loss 3.5092, train acc 6.05%, val loss 3.4318, val acc 6.79%.
- Epoch 3: train loss 3.3656, train acc 9.14%, val loss 3.3469, val acc 7.34%.
- Epoch 4: train loss 3.2600, train acc 11.72%, val loss 3.2652, val acc 8.97%.
- Epoch 5: train loss 3.1636, train acc 14.40%, val loss 3.2322, val acc 11.01%.
- Epoch 6: train loss 3.0840, train acc 15.42%, val loss 3.2067, val acc 10.73%.
- Epoch 7: train loss 2.9923, train acc 18.38%, val loss 3.2055, val acc 11.68%.
- Epoch 8: train loss 2.9051, train acc 19.77%, val loss 3.1710, val acc 12.23%.
- Epoch 9: train loss 2.8022, train acc 22.15%, val loss 3.1941, val acc 11.55%.
- Epoch 10: train loss 2.7181, train acc 24.66%, val loss 3.1846, val acc 13.18%.

**Best validation accuracy:** 13.18% at epoch 10.

**Learning-rate comparison**

- `0.001`: best validation accuracy 17.93%, final train accuracy 86.07%, final val loss 5.3296.
- `0.0005`: best validation accuracy 18.48%, final train accuracy 66.95%, final val loss 4.0480.
- `0.0001`: best validation accuracy 13.18%, final train accuracy 24.66%, final val loss 3.1846.

**Decision:**  
`0.0005` is currently the best learning rate for this baseline. It gives the highest validation accuracy, 
`18.48%`, while overfitting less aggressively than `0.001`. Learning rate `0.0001` is too slow within 10 epochs.
