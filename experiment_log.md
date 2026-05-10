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

## Experiment 4 - Batch size adjustment

**Results summary - Batch size 16, learning rate 0.0005, 10 epochs**

- Epoch 1: train loss 3.5504, train acc 4.76%, val loss 3.3858, val acc 7.74%.
- Epoch 2: train loss 3.3372, train acc 9.71%, val loss 3.2433, val acc 11.14%.
- Epoch 3: train loss 3.1149, train acc 14.30%, val loss 3.1267, val acc 11.82%.
- Epoch 4: train loss 2.8514, train acc 21.47%, val loss 3.0626, val acc 14.67%.
- Epoch 5: train loss 2.5699, train acc 28.43%, val loss 3.1042, val acc 13.18%.
- Epoch 6: train loss 2.2495, train acc 36.41%, val loss 3.1691, val acc 17.12%.
- Epoch 7: train loss 1.8883, train acc 45.01%, val loss 3.4483, val acc 14.67%.
- Epoch 8: train loss 1.4856, train acc 56.25%, val loss 3.6806, val acc 16.58%.
- Epoch 9: train loss 1.0845, train acc 68.68%, val loss 4.1872, val acc 17.12%.
- Epoch 10: train loss 0.7354, train acc 78.94%, val loss 4.6936, val acc 16.03%.

**Best validation accuracy:** 17.12%, reached at epochs 6 and 9.

**Results summary - Batch size 32, learning rate 0.0005, 10 epochs**

- Epoch 1: train loss 3.5683, train acc 3.84%, val loss 3.4107, val acc 7.34%.
- Epoch 2: train loss 3.3712, train acc 8.19%, val loss 3.2951, val acc 9.51%.
- Epoch 3: train loss 3.1925, train acc 12.94%, val loss 3.1717, val acc 12.23%.
- Epoch 4: train loss 2.9680, train acc 18.10%, val loss 3.0716, val acc 13.99%.
- Epoch 5: train loss 2.7312, train acc 23.30%, val loss 3.0487, val acc 15.90%.
- Epoch 6: train loss 2.4746, train acc 30.37%, val loss 3.0692, val acc 15.49%.
- Epoch 7: train loss 2.1830, train acc 38.04%, val loss 3.2522, val acc 15.49%.
- Epoch 8: train loss 1.8862, train acc 45.96%, val loss 3.3536, val acc 17.80%.
- Epoch 9: train loss 1.5569, train acc 55.57%, val loss 3.5849, val acc 17.80%.
- Epoch 10: train loss 1.2500, train acc 65.18%, val loss 3.9722, val acc 18.75%.

**Best validation accuracy:** 18.75% at epoch 10.

**Results summary - Batch size 64, learning rate 0.0005, 10 epochs**

- Epoch 1: train loss 3.5896, train acc 3.43%, val loss 3.5091, val acc 6.25%.
- Epoch 2: train loss 3.4221, train acc 7.74%, val loss 3.3225, val acc 7.34%.
- Epoch 3: train loss 3.2384, train acc 11.72%, val loss 3.2301, val acc 9.51%.
- Epoch 4: train loss 3.0597, train acc 17.02%, val loss 3.1299, val acc 11.28%.
- Epoch 5: train loss 2.8794, train acc 19.97%, val loss 3.1073, val acc 12.64%.
- Epoch 6: train loss 2.7179, train acc 24.86%, val loss 3.0338, val acc 15.22%.
- Epoch 7: train loss 2.5018, train acc 30.23%, val loss 3.0841, val acc 14.27%.
- Epoch 8: train loss 2.2981, train acc 35.53%, val loss 3.1428, val acc 13.72%.
- Epoch 9: train loss 2.0901, train acc 41.24%, val loss 3.2623, val acc 15.62%.
- Epoch 10: train loss 1.8898, train acc 46.47%, val loss 3.2987, val acc 15.22%.

**Best validation accuracy:** 15.62% at epoch 9.

**Decision:**  
Batch size `64` learned more slowly than batch size `32`, also achieved weaker validation performance.
It reached only `15.62%` best validation accuracy, compared with `18.75%` for batch size `32`. 
The final training accuracy was also lower, `46.47%`.
This suggests that the larger batch size gave fewer weight updates per epoch and did not learn enough within 10 epochs.

## Experiment 5, 6, 7, 8 - Adding BatchNorm, testing dropout, testing a winder CNN, testing a deeper CNN

**Experiment - CNN + BatchNorm**

Settings:
- Architecture: baseline CNN with BatchNorm2d after each Conv2d layer.
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 64x64
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5289, train acc 7.03%, val loss 3.3935, val acc 9.24%.
- Epoch 2: train loss 3.2601, train acc 12.40%, val loss 3.1888, val acc 11.96%.
- Epoch 3: train loss 3.0114, train acc 17.29%, val loss 3.0601, val acc 14.54%.
- Epoch 4: train loss 2.7697, train acc 23.98%, val loss 2.9512, val acc 17.39%.
- Epoch 5: train loss 2.5577, train acc 28.60%, val loss 2.9740, val acc 17.39%.
- Epoch 6: train loss 2.3602, train acc 34.31%, val loss 2.9684, val acc 17.66%.
- Epoch 7: train loss 2.1639, train acc 40.69%, val loss 2.9517, val acc 18.21%.
- Epoch 8: train loss 1.9370, train acc 45.96%, val loss 2.9488, val acc 20.92%.
- Epoch 9: train loss 1.7587, train acc 52.24%, val loss 3.0081, val acc 18.75%.
- Epoch 10: train loss 1.5509, train acc 58.56%, val loss 2.9871, val acc 21.47%.

Best validation accuracy:
- 21.47% at epoch 10.

**Decision:** 
Adding BatchNorm improved the baseline validation accuracy from 18.75% to 21.47%. 
The validation loss also stayed lower than in the original baseline run.
THis suggests that BatchNorm made training more stable and helped the model generalise better. 
However, the model still shows overfitting because training accuracy reached 58.56% while 
validation accuracy reached only 21.47%.

**Experiment - BatchNorm CNN + Dropout(0.3)**

Settings:
- Architecture: baseline CNN with BatchNorm2d after each Conv2d layer and Dropout(0.3) after the classifier ReLU.
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 64x64
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5655, train acc 5.16%, val loss 3.4105, val acc 8.70%.
- Epoch 2: train loss 3.4067, train acc 7.78%, val loss 3.3143, val acc 9.65%.
- Epoch 3: train loss 3.2964, train acc 9.44%, val loss 3.2280, val acc 10.73%.
- Epoch 4: train loss 3.1558, train acc 12.26%, val loss 3.1070, val acc 14.27%.
- Epoch 5: train loss 3.0841, train acc 13.35%, val loss 3.1023, val acc 15.22%.
- Epoch 6: train loss 2.9806, train acc 15.69%, val loss 3.0255, val acc 15.22%.
- Epoch 7: train loss 2.8838, train acc 18.61%, val loss 3.0190, val acc 15.22%.
- Epoch 8: train loss 2.7692, train acc 21.13%, val loss 2.9878, val acc 15.76%.
- Epoch 9: train loss 2.7045, train acc 22.28%, val loss 2.9831, val acc 18.21%.
- Epoch 10: train loss 2.6151, train acc 25.10%, val loss 2.9277, val acc 18.61%.

Best validation accuracy:
- 18.61% at epoch 10.

**Decision:**
Adding Dropout(0.3) reduced overfitting because training accuracy only reached 25.10%, 
compared with 58.56% for the BatchNorm-only model. 
However, it also reduced validation accuracy from 21.47% to 18.61%. 
This suggests that 0.3 value was too strong for the classifier at this stage with underfitting so i won't add it

**Experiment - Wider CNN + BatchNorm**

Settings:
- Architecture: wider CNN with BatchNorm2d after each Conv2d layer.
- Channels: 32 → 64 → 128
- Dropout: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 64x64
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5830, train acc 4.65%, val loss 3.4434, val acc 8.42%.
- Epoch 2: train loss 3.3041, train acc 10.46%, val loss 3.2683, val acc 10.46%.
- Epoch 3: train loss 3.0358, train acc 16.47%, val loss 3.1650, val acc 11.55%.
- Epoch 4: train loss 2.8206, train acc 22.45%, val loss 3.0482, val acc 16.17%.
- Epoch 5: train loss 2.6258, train acc 25.92%, val loss 2.9682, val acc 16.58%.
- Epoch 6: train loss 2.3892, train acc 31.79%, val loss 2.9653, val acc 19.29%.
- Epoch 7: train loss 2.1929, train acc 38.18%, val loss 3.0240, val acc 18.48%.
- Epoch 8: train loss 1.9619, train acc 44.19%, val loss 2.9074, val acc 21.06%.
- Epoch 9: train loss 1.7518, train acc 49.90%, val loss 2.9514, val acc 20.52%.
- Epoch 10: train loss 1.5470, train acc 55.91%, val loss 3.1189, val acc 20.79%.

Best validation accuracy:
- 21.06% at epoch 8.

Interpretation:
The wider CNN learned the training data well but did not improve validation accuracy compared with the smaller BatchNorm model.
The best validation accuracy was 21.06%, slightly below the BatchNorm only result of 21.47%. 
Since the wider model is larger and slower but did not generalise better,
I didn't keep this in the architecture.

**Experiment - Deeper CNN + BatchNorm**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer.
- Channels: 16 → 32 → 64 → 128
- Dropout: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 64x64
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5159, train acc 6.18%, val loss 3.3088, val acc 10.46%.
- Epoch 2: train loss 3.1180, train acc 15.39%, val loss 3.1593, val acc 13.32%.
- Epoch 3: train loss 2.8364, train acc 22.04%, val loss 3.0397, val acc 15.90%.
- Epoch 4: train loss 2.5771, train acc 28.67%, val loss 2.9317, val acc 17.66%.
- Epoch 5: train loss 2.3445, train acc 34.48%, val loss 2.8428, val acc 19.84%.
- Epoch 6: train loss 2.1084, train acc 41.64%, val loss 2.8258, val acc 22.83%.
- Epoch 7: train loss 1.8124, train acc 50.17%, val loss 2.9593, val acc 20.24%.
- Epoch 8: train loss 1.5544, train acc 60.26%, val loss 2.8722, val acc 21.88%.
- Epoch 9: train loss 1.3055, train acc 66.54%, val loss 3.1233, val acc 18.75%.
- Epoch 10: train loss 1.0350, train acc 75.65%, val loss 3.1820, val acc 19.57%.

Best validation accuracy:
- 22.83% at epoch 6.

Interpretation:
Adding a fourth convolution block improved validation accuracy compared with the smaller BatchNorm CNN. 
The best validation accuracy increased from 21.47% to 22.83%.
This suggests that teh extra depth helped the model learn more useful visual features.
However, after epoch 6 the validation accuracy dropped while training accuracy continued rising, showing overfitting.
I kept it as the architecture at this stage.