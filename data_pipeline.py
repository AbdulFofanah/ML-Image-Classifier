# This python file serves the purpose to load the Oxford pet trimap dataset

import random  # crop/flip should be random during training

import numpy as np  # used to convert the trimap image into an array
import torch  # used to convert the mask into a tensor
import torchvision.transforms.functional as tf  # lets us manually apply the same transform to image and trimap
from torchvision.transforms import InterpolationMode # Specific resizing methods
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Dataset  # custom trimap wrapper
from sklearn.model_selection import train_test_split # reproducible split

# computed from the training set
TRAINING_MEAN = [0.478537, 0.445716, 0.39523]
TRAINING_STD = [0.260152, 0.254637, 0.262637]

IMAGE_SIZE = 128  # upgraded to 128 to give the model more visual details
BATCH_SIZE = 16
SPLIT_SEED = 67
# Training Dataset 3,496 images (95% of the split - roughly 94/95 per breed)  |
# Validation Dataset 184 images (5% of the split - roughly 5 per breed)       |
VALIDATION_RATIO = 0.05  # <---------------------------------------------------
NUMBER_OF_WORKERS = 0

# NEW: implemented trimap as coursework discussion said it was permitted
# This is used for training, validation, and test because the model expects a 4-channel input.
class OxfordPetTrimapDataset(Dataset):

    def __init__(
        self,
        root = "./data",
        split = "trainval",
        indices = None,  #splitting is declared when initialising the training and validation dataset
        image_size = IMAGE_SIZE,
        mean = TRAINING_MEAN,
        std = TRAINING_STD,
        training = False,
        download = True,

    ):
        # Loads the dataset
        self.dataset = OxfordIIITPet(
                                        root=root,
                                        split=split,
                                        target_types=["category", "segmentation"], # image, (label, trimap)
                                        transform=None,
                                        target_transform=None,
                                        download=download,
                                    )

        self.indices = indices
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.training = training

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)   # length of entire data set
        else:
            return len(self.indices)   # length of specific subset whether it be training or validation

    # This first loads the pet sample and returns a 4-channel tensor with
    # the rgb and the pet mask made from trinamp
    def __getitem__(self, index):
        # picks between the full dataset and trainval dataset
        if self.indices is None:
            custom_index = index
        else:
            custom_index = self.indices[index]

        # Loads the pet image and the target ( = label and trimap segmentation)
        pet_image, target = self.dataset[custom_index]
        label = target[0]
        trimap = target[1]

        pet_image = pet_image.convert("RGB") # Convert this image into RGB mode (3 channels)
        trimap = trimap.convert("L") # Convert this image into L mode (single channel)

        if self.training:    # check if im training and if not im in validation or test mode
            # Generates a random crop
            crop_top, crop_left, crop_height, crop_width = self._get_random_resized_crop_params(pet_image)

            pet_image = tf.resized_crop(
                                          pet_image,
                                          top=crop_top,
                                          left=crop_left,
                                          height=crop_height,
                                          width=crop_width,
                                          size=[self.image_size, self.image_size], # resize to image size
                                          interpolation=InterpolationMode.BILINEAR, # bilinear blends nearby pixels
                                       )

            trimap = tf.resized_crop(
                                        trimap,
                                        top=crop_top,
                                        left=crop_left,
                                        height=crop_height,
                                        width=crop_width,
                                        size=[self.image_size, self.image_size],
                                        interpolation=InterpolationMode.NEAREST, # picks the closest pixel value
                                    )

            # randomly flip the image and the trimap
            # the equivalent of transforms.RandomHorizontalFlip(p=0.5)
            if random.random() < 0.5:
                pet_image = tf.hflip(pet_image)
                trimap = tf.hflip(trimap)

        else:
            # testing and validation
            pet_image = tf.resize(
                                    pet_image,
                                    size=[self.image_size, self.image_size],
                                    interpolation=InterpolationMode.BILINEAR,
                                 )

            trimap = tf.resize(
                                trimap,
                                size=[self.image_size, self.image_size],
                                interpolation=InterpolationMode.NEAREST,
                              )

        rgb_tensor = tf.to_tensor(pet_image) # convert to tensor
        rgb_tensor = tf.normalize(rgb_tensor, mean=self.mean, std=self.std) # normalise

        # converting a PIL image to a NumPy array uses uint8 as default
        trimap_array = np.array(trimap, dtype=np.uint8)

        # original trimap values are 1 = pet, 2 = background, 3 = boundary
        # trimap_array != 2 returns an array of true and false
        # astype(np.float32) converts the boolean to 1 and 0
        pet_mask_array = (trimap_array != 2).astype(np.float32)

        # adds the 1 channel - [1, 128, 128] - this is so that it can be joined with the rgb image
        pet_mask_tensor = torch.from_numpy(pet_mask_array).unsqueeze(0)

        # combines both rgb tensor and pet mask tensor
        # [4, 128, 128] - 4 channels
        combined_tensor = torch.cat([rgb_tensor, pet_mask_tensor], dim=0)

        return combined_tensor, label

    '''
    This function attempts to emulate my previous 
    transforms.RandomResizedCrop(size=128, scale=(0.7, 1.0))
    '''
    def _get_random_resized_crop_params(self, image):
        # get the image size
        width, height = image.size
        area = width * height

        # 70% and 100% of the original image area - tuned value
        image_scale = (0.7, 1.0)

        # The crop aspect ratio  varies between 3:4 and 4:3.
        # This stops the crop becoming too thin or stretched.
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        # Converts the ratio range into log space
        # (torchvision samples aspect ratios logarithmically)
        log_ratio = torch.log(torch.tensor(ratio))

        for _ in range(10):
            # Randomly choose the target crop area.
            target_area = area * random.uniform(image_scale[0], image_scale[1])

            # Randomly choose the aspect ratio of the crop.
            aspect_ratio = torch.exp(
                                     torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
                                    ).item()

            # Calculate crop width and crop height
            crop_width = int(round((target_area * aspect_ratio) ** 0.5))
            crop_height = int(round((target_area / aspect_ratio) ** 0.5))

            # Check if the crop fits inside the original image.
            if 0 < crop_width <= width and 0 < crop_height <= height:
                # Randomly choose where the crop starts.
                crop_top = random.randint(0, height - crop_height)
                crop_left = random.randint(0, width - crop_width)

                # Return the crop position and crop size.
                return crop_top, crop_left, crop_height, crop_width

        # fall back to a centred crop
        # if no valid random crop was found after 10 attempts
        centered_ratio = width / height

        if centered_ratio < min(ratio):
            crop_width = width
            crop_height = int(round(crop_width / min(ratio)))

        elif centered_ratio > max(ratio):
            crop_height = height
            crop_width = int(round(crop_height * max(ratio)))

        else:
            crop_width = width
            crop_height = height

        crop_top = (height - crop_height) // 2
        crop_left = (width - crop_width) // 2

        return crop_top, crop_left, crop_height, crop_width

def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUMBER_OF_WORKERS):
    # loads the trainval labels so the official trainval split can be split into training and validation sets
    label_trainval_data = OxfordIIITPet(
                                            root="./data",
                                            split="trainval",
                                            target_types="category",
                                            transform=None,
                                            download=True,
                                        )

    # Stratified 95/05 split which keeps the class distribution similar in training and validation.
    # I did this as it helps make sure all 37 breeds are represented fairly in both the training and validation sets.
    all_image_position_indices = list(range(len(label_trainval_data)))  # list of all image positions in the trainval_data dataset.
    all_image_labels = label_trainval_data._labels  # the category label for every image, used for stratified splitting

    training_indices, validation_indices = train_test_split(  # split the position of the indices
        all_image_position_indices,
        test_size=VALIDATION_RATIO,  # 5% which is the validation set
        random_state=SPLIT_SEED,  # reproducible split with the seed
        stratify=all_image_labels  # preserves the class distribution
    )

    training_data = OxfordPetTrimapDataset(     # load training dataset
                                                root="./data",
                                                split="trainval",
                                                indices=training_indices,
                                                image_size=IMAGE_SIZE,
                                                mean=TRAINING_MEAN,
                                                std=TRAINING_STD,
                                                training=True,
                                                download=True,
                                         )

    validation_data = OxfordPetTrimapDataset(   # load validation dataset
                                                root="./data",
                                                split="trainval",
                                                indices=validation_indices,
                                                image_size=IMAGE_SIZE,
                                                mean=TRAINING_MEAN,
                                                std=TRAINING_STD,
                                                training=False,
                                                download=True,
                                            )

    test_data = OxfordPetTrimapDataset(         # load test dataset
                                                root="./data",
                                                split="test",
                                                indices=None,
                                                image_size=IMAGE_SIZE,
                                                mean=TRAINING_MEAN,
                                                std=TRAINING_STD,
                                                training=False,
                                                download=True,
                                      )

    # Data Loaders
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,  # allows us to change the value when calling the function
        shuffle=True,
        num_workers=num_workers,  # allows us to change the value when calling the function
        pin_memory=True  # speeds up the CPU to GPU transfer as i'm using CUDA
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,  # allows us to change the value when calling the function
        shuffle=False,
        num_workers=num_workers,  # allows us to change the value when calling the function
        pin_memory=True  # speeds up the CPU to GPU transfer as i'm using CUDA
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,  # allows us to change the value when calling the function
        shuffle=False,
        num_workers=num_workers,  # allows us to change the value when calling the function
        pin_memory=True  # speeds up the CPU to GPU transfer as i'm using CUDA
    )

    return training_loader, validation_loader, test_loader

def main():
    training_loader, validation_loader, test_loader = get_data_loaders()

    # python can iterate through batches of data
    batch_images, batch_labels = next(iter(training_loader))
    print(
        f"Training size: {len(training_loader.dataset)}\n"
        f"Validation size: {len(validation_loader.dataset)}\n"
        f"Test size: {len(test_loader.dataset)}\n"
        f"Training batches: {len(training_loader)}\n"
        f"Validation batches: {len(validation_loader)}\n"
        f"Test batches: {len(test_loader)}\n"
        f"Batch image tensor shape: {batch_images.shape}\n"
        f"Batch label tensor shape: {batch_labels.shape}\n"
        f"RGB min/max: {batch_images[:, :3].min().item():.4f} / {batch_images[:, :3].max().item():.4f}\n"
        f"Mask min/max: {batch_images[:, 3:].min().item():.4f} / {batch_images[:, 3:].max().item():.4f}"
    )

if __name__ == "__main__":
    main()