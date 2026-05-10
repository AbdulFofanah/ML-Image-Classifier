# This python file serves the purpose to:
"""
1) Load the dataset
2) use transforms on the data
3) split it into training data and validation data
4) Use the DataLoaders to load the images in batches
"""

import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset  # Subset selects dataset items by index without copying images
from sklearn.model_selection import train_test_split

TRAINING_MEAN = [0.479322, 0.446042, 0.395331]
TRAINING_STD = [0.255698, 0.250303, 0.258181]

IMAGE_SIZE = 96
BATCH_SIZE = 32
SPLIT_SEED = 67
VALIDATION_RATIO = 0.2
NUMBER_OF_WORKERS = 0


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUMBER_OF_WORKERS):
    # Allows the batch size and worker count to be changed during future experiments.
    image_transformation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # small and fast training
        transforms.ToTensor(),
        transforms.Normalize(
                              mean=TRAINING_MEAN,
                              std=TRAINING_STD
                            )
    ])

    trainval_data = OxfordIIITPet(  # Load official trainval dataset
        root="./data",
        split="trainval",
        target_types="category",
        transform=image_transformation,
        download=True,
    )

    test_data = OxfordIIITPet(  # Load official test dataset
        root="./data",
        split="test",
        target_types="category",
        transform=image_transformation,
        download=True,
    )

    # Stratified 80/20 split which keeps the class distribution similar in training and validation.
    # I did this as it helps make sure all 37 breeds are represented fairly in both the training and validation sets.
    all_image_position_indices = list(range(len(trainval_data)))  # list of all image positions in the trainval_data dataset.
    all_image_labels = trainval_data._labels  # the category label for every image, used for stratified splitting

    training_indices, validation_indices = train_test_split(  # split the position of the indices
        all_image_position_indices,
        test_size=VALIDATION_RATIO,  # 20% which is the validation set
        random_state=SPLIT_SEED,  # reproducible split with the seed
        stratify=all_image_labels  # preserves the class distribution
    )

    training_data = Subset(trainval_data, training_indices)
    validation_data = Subset(trainval_data, validation_indices)

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
        f"Batch label tensor shape: {batch_labels.shape}"
    )

if __name__ == "__main__":
    main()