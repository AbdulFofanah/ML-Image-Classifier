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

## Experiment 5, 6, 7, 8, 9, 10, 11 - Adding BatchNorm, testing dropout, testing a winder CNN, testing a deeper CNN, tesing AdaptiveAvgPool, testing a bigger image size, testing image sixe 128x128

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

**Decision:**
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

**Experiment - Deeper CNN + BatchNorm + AdaptiveAvgPool**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)) before the classifier.
- Channels: 16 → 32 → 64 → 128
- Dropout: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 64x64
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5201, train acc 6.18%, val loss 3.3212, val acc 9.92%.
- Epoch 2: train loss 3.1281, train acc 14.91%, val loss 3.1313, val acc 12.77%.
- Epoch 3: train loss 2.8462, train acc 21.91%, val loss 3.0267, val acc 14.40%.
- Epoch 4: train loss 2.5953, train acc 27.96%, val loss 2.9473, val acc 15.35%.
- Epoch 5: train loss 2.3761, train acc 33.22%, val loss 2.8840, val acc 17.66%.
- Epoch 6: train loss 2.1489, train acc 39.88%, val loss 2.8470, val acc 21.20%.
- Epoch 7: train loss 1.8552, train acc 49.59%, val loss 2.9143, val acc 19.84%.
- Epoch 8: train loss 1.6168, train acc 58.22%, val loss 2.8953, val acc 23.64%.
- Epoch 9: train loss 1.3538, train acc 65.08%, val loss 2.9997, val acc 20.92%.
- Epoch 10: train loss 1.0997, train acc 73.37%, val loss 3.0865, val acc 20.92%.

Best validation accuracy:
- 23.64% at epoch 8.

**Decision:**
Adding AdaptiveAvgPool2d((4, 4)) kept the final feature map fixed at 128×4×4 before the classifier.
At 64x64 input size this should not substantially change the architecture - the feature map was already 4x4.
It makes the model compatible with larger image sizes later and this run achieved the best validation accuracy so far, 23.64%,
so i'm keeping the adaptive pooling as part of the current architecture.

**Experiment - Image size 96x96 with Deeper CNN + BatchNorm + AdaptiveAvgPool**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)) before the classifier.
- Channels: 16 → 32 → 64 → 128
- Dropout: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 96x96
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5412, train acc 6.45%, val loss 3.4248, val acc 8.15%.
- Epoch 2: train loss 3.2512, train acc 11.85%, val loss 3.2575, val acc 13.04%.
- Epoch 3: train loss 3.0398, train acc 17.12%, val loss 3.1305, val acc 13.86%.
- Epoch 4: train loss 2.8581, train acc 21.03%, val loss 3.0192, val acc 16.17%.
- Epoch 5: train loss 2.7115, train acc 24.32%, val loss 2.8990, val acc 20.79%.
- Epoch 6: train loss 2.5495, train acc 28.33%, val loss 2.8411, val acc 21.06%.
- Epoch 7: train loss 2.3975, train acc 32.40%, val loss 2.8636, val acc 21.47%.
- Epoch 8: train loss 2.2647, train acc 36.45%, val loss 2.7376, val acc 23.78%.
- Epoch 9: train loss 2.1207, train acc 39.74%, val loss 3.0302, val acc 19.29%.
- Epoch 10: train loss 2.0039, train acc 42.22%, val loss 2.9070, val acc 22.83%.

Best validation accuracy:
- 23.78% at epoch 8.

**Decision:**
Increasing the input image size from 64x64 to 96x96 slightly improved the best validation accuracy from 23.64% to 23.78%. 
Small improvement, but the larger image size didn't hurt performance and may have allowed the model to use slightly more 
visual detail. AdaptiveAvgPool2d((4, 4)) allowed this image-size change without changing the classifier input size.

**Experiment - Image size 128x128 with Deeper CNN + BatchNorm + AdaptiveAvgPool**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)) before the classifier.
- Channels: 16 → 32 → 64 → 128
- Dropout: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Image size: 128x128
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5138, train acc 7.27%, val loss 3.3391, val acc 11.68%.
- Epoch 2: train loss 3.1488, train acc 13.69%, val loss 3.1297, val acc 14.54%.
- Epoch 3: train loss 2.8927, train acc 19.36%, val loss 2.9958, val acc 16.71%.
- Epoch 4: train loss 2.6973, train acc 23.74%, val loss 2.9004, val acc 19.29%.
- Epoch 5: train loss 2.5431, train acc 27.51%, val loss 2.8302, val acc 19.29%.
- Epoch 6: train loss 2.3937, train acc 32.98%, val loss 2.7389, val acc 23.64%.
- Epoch 7: train loss 2.2187, train acc 35.84%, val loss 2.7364, val acc 21.33%.
- Epoch 8: train loss 2.0803, train acc 41.81%, val loss 2.7195, val acc 23.10%.
- Epoch 9: train loss 1.9293, train acc 44.80%, val loss 2.7989, val acc 23.10%.
- Epoch 10: train loss 1.8031, train acc 48.74%, val loss 2.8597, val acc 22.69%.

Best validation accuracy:
- 23.64% at epoch 6.

**Decision:**
Increasing the image size to 128x128 did not improve validation accuracy compared with 96x96. 
The best validation accuracy was 23.64%, slightly below the 96x96 result of 23.78%. 
The 128x128 run was also slower because the larger images increased computation. 
I settled with 96x96 as the current best image size.

## Experiment 12 Data augmentation

**Experiment - RandomHorizontalFlip only**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomHorizontalFlip(p=0.5)
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5515, train acc 6.32%, val loss 3.4283, val acc 9.65%.
- Epoch 2: train loss 3.2873, train acc 10.19%, val loss 3.1983, val acc 11.55%.
- Epoch 3: train loss 3.0834, train acc 15.08%, val loss 3.1303, val acc 13.72%.
- Epoch 4: train loss 2.9310, train acc 18.07%, val loss 2.9961, val acc 15.62%.
- Epoch 5: train loss 2.7991, train acc 20.86%, val loss 2.8719, val acc 20.65%.
- Epoch 6: train loss 2.6824, train acc 25.24%, val loss 2.8557, val acc 19.97%.
- Epoch 7: train loss 2.5684, train acc 27.48%, val loss 2.8195, val acc 22.42%.
- Epoch 8: train loss 2.4726, train acc 28.80%, val loss 2.7688, val acc 19.84%.
- Epoch 9: train loss 2.3763, train acc 30.64%, val loss 2.7564, val acc 23.64%.
- Epoch 10: train loss 2.3203, train acc 33.25%, val loss 2.8037, val acc 22.01%.

Best validation accuracy:
- 23.64% at epoch 9.

**Interpretation**
RandomHorizontalFlip(p=0.5) reduced training accuracy compared with the no-augmentation run, 
did not improve validation accuracy, achieving 23.64% compared with the previous best of 23.78%. 
The difference is small, so horizontal flip was not clearly harmful, but it was not clearly beneficial on its own.

**Experiment — Phase 7B: RandomHorizontalFlip + RandomRotation(10)**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomHorizontalFlip(p=0.5) + RandomRotation(10)
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5854, train acc 5.20%, val loss 3.4906, val acc 7.34%.
- Epoch 2: train loss 3.3266, train acc 10.70%, val loss 3.2621, val acc 11.28%.
- Epoch 3: train loss 3.1222, train acc 13.89%, val loss 3.1949, val acc 11.28%.
- Epoch 4: train loss 2.9853, train acc 17.53%, val loss 3.1616, val acc 12.23%.
- Epoch 5: train loss 2.8781, train acc 18.58%, val loss 2.9600, val acc 18.34%.
- Epoch 6: train loss 2.7647, train acc 22.32%, val loss 2.8793, val acc 19.16%.
- Epoch 7: train loss 2.6718, train acc 25.24%, val loss 2.8469, val acc 19.43%.
- Epoch 8: train loss 2.5970, train acc 26.49%, val loss 2.8078, val acc 20.38%.
- Epoch 9: train loss 2.5521, train acc 27.55%, val loss 2.9380, val acc 19.97%.
- Epoch 10: train loss 2.4644, train acc 29.79%, val loss 2.7881, val acc 21.74%.

Best validation accuracy:
- 21.74% at epoch 10.

**Interpretation:**
Adding RandomRotation(10) alongside horizontal flipping reduced validation accuracy from 23.64% with flip only to 21.74%.
This suggests that rotation made the augmented training images too different or harder for the current model to learn from.
i removed RandomRotation(10) for the next augmentation experiment.

**Experiment - RandomHorizontalFlip + ColorJitter**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomHorizontalFlip(p=0.5) + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5704, train acc 5.50%, val loss 3.4154, val acc 9.10%.
- Epoch 2: train loss 3.3470, train acc 9.54%, val loss 3.2904, val acc 8.29%.
- Epoch 3: train loss 3.1438, train acc 13.79%, val loss 3.1381, val acc 13.59%.
- Epoch 4: train loss 3.0108, train acc 16.20%, val loss 3.0439, val acc 14.81%.
- Epoch 5: train loss 2.8897, train acc 20.48%, val loss 2.9659, val acc 18.07%.
- Epoch 6: train loss 2.7743, train acc 22.49%, val loss 2.9012, val acc 18.34%.
- Epoch 7: train loss 2.6628, train acc 24.86%, val loss 2.8442, val acc 20.79%.
- Epoch 8: train loss 2.5801, train acc 27.07%, val loss 2.7217, val acc 22.15%.
- Epoch 9: train loss 2.4749, train acc 30.30%, val loss 2.9219, val acc 20.24%.
- Epoch 10: train loss 2.4010, train acc 31.83%, val loss 2.6584, val acc 21.88%.

Best validation accuracy:
- 22.15% at epoch 8.

**Interpretation:**
Adding ColorJitter with horizontal flipping reduced validation accuracy compared with both the no-augmentation result and the flip-only result.
The model trained more slowly and reached only 22.15% best validation accuracy, compared with 23.78% without augmentation.
This suggests that the colour changes may have made the training data harder without improving generalisation for this current setup.

**Experiment - Safe RandomResizedCrop**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0))
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5227, train acc 6.56%, val loss 3.4161, val acc 8.83%.
- Epoch 2: train loss 3.2401, train acc 11.92%, val loss 3.2979, val acc 10.60%.
- Epoch 3: train loss 3.0427, train acc 16.10%, val loss 3.1883, val acc 12.36%.
- Epoch 4: train loss 2.8920, train acc 19.46%, val loss 3.1076, val acc 13.99%.
- Epoch 5: train loss 2.7388, train acc 23.34%, val loss 2.9891, val acc 17.80%.
- Epoch 6: train loss 2.6167, train acc 26.02%, val loss 3.2484, val acc 13.18%.
- Epoch 7: train loss 2.4828, train acc 29.21%, val loss 3.0752, val acc 17.53%.
- Epoch 8: train loss 2.4053, train acc 31.28%, val loss 2.7698, val acc 22.15%.
- Epoch 9: train loss 2.2700, train acc 34.24%, val loss 2.8422, val acc 23.23%.
- Epoch 10: train loss 2.1575, train acc 38.96%, val loss 2.7055, val acc 23.91%.

Best validation accuracy:
- 23.91% at epoch 10.

**Interpretation:**
Safe RandomResizedCrop with scale=(0.8, 1.0) achieved the best augmentation result so far, reaching 23.91% validation accuracy.
This slightly improved on the previous best result of 23.78% without augmentation. 
The improvement was small, but it suggests that mild crop/zoom variation may help the model generalise slightly better than 
flip, rotation, or colour jitter for this setup.

**Experiment - Safe RandomResizedCrop + RandomHorizontalFlip**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0)) + RandomHorizontalFlip(p=0.5)
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5697, train acc 5.88%, val loss 3.4668, val acc 8.56%.
- Epoch 2: train loss 3.3376, train acc 10.90%, val loss 3.3303, val acc 8.56%.
- Epoch 3: train loss 3.1150, train acc 13.76%, val loss 3.1341, val acc 12.23%.
- Epoch 4: train loss 2.9584, train acc 17.66%, val loss 3.1020, val acc 13.45%.
- Epoch 5: train loss 2.8406, train acc 21.30%, val loss 3.1831, val acc 14.27%.
- Epoch 6: train loss 2.7168, train acc 23.88%, val loss 2.9797, val acc 16.30%.
- Epoch 7: train loss 2.6092, train acc 26.83%, val loss 2.8520, val acc 21.60%.
- Epoch 8: train loss 2.5133, train acc 27.75%, val loss 2.8844, val acc 20.92%.
- Epoch 9: train loss 2.4468, train acc 29.79%, val loss 2.7533, val acc 25.00%.
- Epoch 10: train loss 2.3687, train acc 31.69%, val loss 2.9057, val acc 22.01%.

Best validation accuracy:
- 25.00% at epoch 9.

**Interpretation:**
Combining safe RandomResizedCrop(scale=(0.8, 1.0)) with RandomHorizontalFlip(p=0.5) produced the best validation accuracy so far at 25.00%.
This improved over RandomResizedCrop alone at 23.91% and over the previous no-augmentation best of 23.78%.
The training accuracy stayed relatively low, suggesting that augmentation made memorisation harder while improving generalisation.

# Experiment 13 AdamW

**Experiment - AdamW with weight_decay=0.0001**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0)) + RandomHorizontalFlip(p=0.5)
- Validation/test augmentation: none
- Optimiser: AdamW
- Weight decay: 0.0001
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10

Results:
- Epoch 1: train loss 3.5698, train acc 6.39%, val loss 3.4692, val acc 6.66%.
- Epoch 2: train loss 3.3371, train acc 10.60%, val loss 3.3191, val acc 8.83%.
- Epoch 3: train loss 3.1167, train acc 13.82%, val loss 3.1637, val acc 12.50%.
- Epoch 4: train loss 2.9676, train acc 16.88%, val loss 3.0821, val acc 13.72%.
- Epoch 5: train loss 2.8533, train acc 20.21%, val loss 3.1529, val acc 13.86%.
- Epoch 6: train loss 2.7419, train acc 23.06%, val loss 2.9972, val acc 16.85%.
- Epoch 7: train loss 2.6440, train acc 26.05%, val loss 2.8625, val acc 20.38%.
- Epoch 8: train loss 2.5450, train acc 27.38%, val loss 2.8629, val acc 20.11%.
- Epoch 9: train loss 2.4872, train acc 28.23%, val loss 2.7828, val acc 22.96%.
- Epoch 10: train loss 2.4023, train acc 30.23%, val loss 2.9742, val acc 21.33%.

Best validation accuracy:
- 22.96% at epoch 9.

**Interpretation:**
AdamW with weight_decay=0.0001 reduced validation accuracy compared with standard Adam. 
The best validation accuracy dropped from 25.00% with Adam to 22.96% with AdamW. 
This suggests that this weight decay value added too much regularisation or slowed the useful learning for the current setup,
so i removed it

**Experiment - Adam + CosineAnnealingLR**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0)) + RandomHorizontalFlip(p=0.5)
- Validation/test augmentation: none
- Optimiser: Adam
- Initial learning rate: 0.0005
- Scheduler: CosineAnnealingLR with T_max=10
- Batch size: 32
- Epochs: 10

Results:
- Epoch 1: train loss 3.5685, train acc 5.98%, val loss 3.4858, val acc 7.88%, lr 0.00048776.
- Epoch 2: train loss 3.3342, train acc 10.60%, val loss 3.3285, val acc 8.29%, lr 0.00045225.
- Epoch 3: train loss 3.1028, train acc 14.06%, val loss 3.1483, val acc 12.50%, lr 0.00039695.
- Epoch 4: train loss 2.9476, train acc 17.43%, val loss 3.0699, val acc 13.59%, lr 0.00032725.
- Epoch 5: train loss 2.8254, train acc 20.75%, val loss 3.1055, val acc 15.22%, lr 0.00025000.
- Epoch 6: train loss 2.7040, train acc 25.10%, val loss 2.9618, val acc 16.44%, lr 0.00017275.
- Epoch 7: train loss 2.6111, train acc 27.24%, val loss 2.9219, val acc 18.89%, lr 0.00010305.
- Epoch 8: train loss 2.5338, train acc 28.46%, val loss 2.8467, val acc 20.65%, lr 0.00004775.
- Epoch 9: train loss 2.4850, train acc 30.84%, val loss 2.7967, val acc 21.74%, lr 0.00001224.
- Epoch 10: train loss 2.4582, train acc 31.05%, val loss 2.8030, val acc 21.47%, lr 0.00000000.

Best validation accuracy:
- 21.74% at epoch 9.

**Interpretation:**
CosineAnnealingLR reduced validation accuracy compared with the fixed learning-rate Adam setup.
The best validation accuracy dropped from 25.00% to 21.74%. The learning rate decayed very low by the final epochs,
reaching almost zero, which likely slowed the useful learning too much within the 10-epoch experiment - might perform
better when doing it on higher epochs. removed ti for the time being btu will test again at higher epochs

# Fixing mistake made in augmentation
**Experiment - RandomResizedCrop only**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0))
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5381, train acc 6.22%, val loss 3.4159, val acc 8.15%.
- Epoch 2: train loss 3.2367, train acc 13.52%, val loss 3.1963, val acc 12.23%.
- Epoch 3: train loss 3.0359, train acc 16.41%, val loss 3.0767, val acc 14.67%.
- Epoch 4: train loss 2.8510, train acc 20.75%, val loss 3.0318, val acc 16.03%.
- Epoch 5: train loss 2.7086, train acc 24.86%, val loss 2.9702, val acc 17.12%.
- Epoch 6: train loss 2.5667, train acc 28.26%, val loss 2.8238, val acc 19.70%.
- Epoch 7: train loss 2.4446, train acc 30.37%, val loss 2.8029, val acc 22.42%.
- Epoch 8: train loss 2.3191, train acc 34.44%, val loss 2.6790, val acc 23.64%.
- Epoch 9: train loss 2.1924, train acc 38.25%, val loss 2.7205, val acc 23.64%.
- Epoch 10: train loss 2.0705, train acc 41.51%, val loss 2.6674, val acc 26.63%.

Best validation accuracy:
- 26.63% at epoch 10.

**Interpretation**
After correcting RandomResizedCrop so that it operated directly on the original image rather than after an initial Resize
crop-only augmentation achieved the best validation accuracy so far at 26.63%. 
This improved over the previous no-augmentation result of 23.78% and suggested that a crop variation
helps the model generalise better.

**Experiment - Corrected RandomResizedCrop + RandomHorizontalFlip**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.8, 1.0)) + RandomHorizontalFlip(p=0.5)
- Validation/test augmentation: none
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 10
- Optimiser: Adam

Results:
- Epoch 1: train loss 3.5345, train acc 6.45%, val loss 3.4190, val acc 8.97%.
- Epoch 2: train loss 3.2386, train acc 11.68%, val loss 3.1993, val acc 12.09%.
- Epoch 3: train loss 3.0348, train acc 16.00%, val loss 3.2111, val acc 11.96%.
- Epoch 4: train loss 2.9254, train acc 18.68%, val loss 2.9852, val acc 16.71%.
- Epoch 5: train loss 2.7842, train acc 22.11%, val loss 2.9206, val acc 17.26%.
- Epoch 6: train loss 2.6843, train acc 24.18%, val loss 2.9810, val acc 16.03%.
- Epoch 7: train loss 2.5887, train acc 28.29%, val loss 2.8258, val acc 18.48%.
- Epoch 8: train loss 2.4980, train acc 28.80%, val loss 2.7521, val acc 21.74%.
- Epoch 9: train loss 2.4050, train acc 31.66%, val loss 2.6629, val acc 22.01%.
- Epoch 10: train loss 2.3228, train acc 33.53%, val loss 2.6430, val acc 23.23%.

Best validation accuracy:
- 23.23% at epoch 10.

**Interpretation:**
After correcting RandomResizedCrop so that it operates directly on the original image, 
I tested adding RandomHorizontalFlip(p=0.5). 
This reduced validation accuracy from 26.63% with crop-only to 23.23%. 
Therefore, horizontal flipping did not help this corrected augmentation setup
, I removed it

# Experiment 14 - 30 epoch experiemnt
** longer training of current setup**

Settings:
- Architecture: deeper CNN with BatchNorm2d after each Conv2d layer and AdaptiveAvgPool2d((4, 4)).
- Image size: 96x96
- Training augmentation: corrected RandomResizedCrop(size=96, scale=(0.8, 1.0)) only
- Validation/test augmentation: none; deterministic Resize((96, 96))
- Optimiser: Adam
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 30
- DataLoader num_workers: 0

Best result:
- Best validation accuracy: 33.83% at epoch 29.
- Epoch 29 training accuracy: 81.52%.
- Epoch 29 validation loss: 2.9066.

**Interpretation:**
Training for longer improved validation accuracy from 26.63% at 10 epochs to 33.83% at 30 epochs,
showing that the model benefited from additional training time. 
Later epochs the training accuracy rose above 80% while validation accuracy stayed around the low 30% range. 
This indicates overfitting sop io should do some measure to tackle that

** Visual Geometry Group, 30 epochs**

Settings:
- Architecture: VGG-style CNN with two Conv2d + BatchNorm2d + ReLU layers before each MaxPool2d.
- Blocks: 3→32→32, 32→64→64, 64→128→128.
- Image size: 96x96
- Training augmentation: corrected RandomResizedCrop(size=96, scale=(0.8, 1.0)) only
- Validation/test augmentation: deterministic Resize((96, 96))
- Optimiser: Adam
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 30
- DataLoader num_workers: 0

Best result:
- Best validation accuracy: 37.09% at epoch 29.
- Epoch 29 training accuracy: 88.76%.
- Epoch 29 validation loss: 2.7174.

Interpretation:
The VGG-style CNN improved validation accuracy from the previous best of 33.83% to 37.09%.
This suggests that using two convolutional layers before each pooling operation helped the model learn richer visual features
for pet breed classification. However, the training accuracy reached 88.76% while validation accuracy was 37.09%,
indicating overfitting. 

**VGG-style CNN with Dropout(0.2)**
Adding Dropout(0.2) reduced overfitting by lowering training accuracy, but it did not improve validation accuracy. 
The best validation accuracy fell from 37.09% without Dropout to 36.28% with Dropout. 
This suggests that Dropout(0.2) restricted learning too much for this model, i removed it

# Experiment 50
**Experiment - VGG style CNN with stronger ((0.7)) RandomResizedCrop, 30 epochs**

Settings:
- Architecture: VGG-style CNN without Dropout.
- Image size: 96x96
- Training augmentation: RandomResizedCrop(size=96, scale=(0.7, 1.0))
- Validation/test augmentation: deterministic Resize((96, 96))
- Loss function: CrossEntropyLoss()
- Optimiser: Adam
- Learning rate: 0.0005
- Batch size: 32
- Epochs: 30
- DataLoader num_workers: 0

Best result:
- Best validation accuracy: 38.32% at epoch 30.
- Epoch 30 training accuracy: 82.13%.
- Epoch 30 validation loss: 2.6163.

Interpretation:
Strengthening the RandomResizedCrop augmentation from scale=(0.8, 1.0) to scale=(0.7, 1.0) improved validation accuracy
from the previous best of 37.09% to 38.32%.
This suggests that stronger crop/zoom variation helped reduce overfitting and improved generalisation to unseen validation images.


# experient whatever - fine tuning
Experiment: 3-block VGG, lower learning rate
Architecture: 3-block VGG-style CNN with BatchNorm and AdaptiveAvgPool2d((4,4))
Training transform: RandomResizedCrop scale=(0.7,1.0), image size 96
Validation transform: Resize 96, ToTensor, Normalize
Loss: CrossEntropyLoss
Optimiser: Adam
Learning rate: 0.0003
Batch size: 32
Epochs: 30
Best validation accuracy: 39.95%
Best epoch: 29

experiment: transforms.RandomHorizontalFlip(p=0.25),
The best validation accuracy: 40.62%
Training loss (4 decimal places): 0.8974
Training accuracy: 73.71%
Validation loss (4 decimal places): 2.2806
Validation accuracy: 40.62%
Epoch time: 15.80 seconds

Experiment : Adam + ReduceLROnPlateau(mode="max", factor=0.5, patience=3)
The best validation accuracy: 44.16%
Training loss (4 decimal places): 0.9456
Training accuracy: 74.49%
Validation loss (4 decimal places): 2.1255
Validation accuracy: 42.66%
Epoch time: 17.81 seconds
Learning rate before scheduler update: 0.000075
Learning rate for next epoch: 0.000075

Experiment : Adam + ReduceLROnPlateau(mode="max", factor=0.5, patience=3) + Learning rate 0.0004:
Best validation accuracy = 45.24%
Best epoch = 29
Epoch 29
Training loss = 0.8624
Training accuracy = 75.65%
Validation loss = 1.9877
Validation accuracy = 45.24%
LR = 0.000200

Experiment : Adam + ReduceLROnPlateau(mode="max", factor=0.25, patience=3) + Learning rate 0.0004:
Best validation accuracy = 47.55%
Epoch 29
Training loss = 0.7704
Training accuracy = 78.80%
Validation loss = 1.8724
Validation accuracy = 47.55%
LR = 0.000100

Experiment : transforms.RandomHorizontalFlip(p=0.15),
Epoch 27
Validation accuracy = 48.10%
Training accuracy = 78.74%
Validation loss = 1.9196
Learning rate = 0.000100
The best validation accuracy: 48.10%

# New scheduler and higher image size
Experiment: 112x112 + 4-Block CNN + OneCycleLR
IMAGE_SIZE = 112
4-block VGG-style CNN
AdaptiveAvgPool2d((1,1))
Classifier: Linear(256 → 37)
Adam optimiser
Initial learning rate = 0.001
OneCycleLR scheduler
max_lr = 0.002
RandomResizedCrop scale=(0.7,1.0)
RandomHorizontalFlip p=0.15
CrossEntropyLoss
Batch size = 32
Seed = 67

Training loss (4 decimal places): 0.3963
Training accuracy: 91.30%
Validation loss (4 decimal places): 1.2747
Validation accuracy: 63.32%
Epoch time: 17.87 seconds
Learning rate before scheduler update: 0.000000
Saved the new best model to model.pth
Learning rate for next epoch: 0.000000

Training complete
The best validation accuracy: 63.32%
re-run - The best validation accuracy: 63.86%

I made a couple of major changes tha were necessary to compute the higher image size effectively:
1. Image size increased from 96x96 to 112x112.
2. Architecture changed from a 3-block CNN to a 4-block CNN to process those new details
3. Final feature depth increased from 128 channels to 256 channels as there was an additional block
4. Pooling changed from AdaptiveAvgPool2d((4,4)) to AdaptiveAvgPool2d((1,1)) - Taking 1x1 averages out the entire feature map
5. Classifier changed from a hidden-layer classifier to a direct Linear(256 → 37) classifier - 
rreduced the amount of parameters to learn from, from the hidden layer which in theory 
reduces over fit as it reduces the models ability to memorise training data (9,472 parameters)
6. Scheduler changed from ReduceLROnPlateau to OneCycleLR -
1-Cycle Policy rapidly increases the learning rate to a peak taking the model out of a bad minimas and then
reduces the learning rate to nearly zero , so it can refine the weights. This should allow us to train faster
as i am limited to 30 epochs
7. Learning rate strategy changed from plateau-based LR to batch-by-batch LR cycling as the adjustments go from
30 per epoch to 2,760 allow it to learn faster