import torch
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128
BATCH_SIZE = 16
SPLIT_SEED = 67
VALIDATION_RATIO = 0.1
NUMBER_OF_WORKERS = 0

def main():
    # Only resizes and convert to tensor as im computign the mean and std
    mean_std_transforms = transforms.Compose([
                                                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                                transforms.ToTensor()
                                            ])

    trainval_data = OxfordIIITPet( # Load official trainval dataset
                                    root="./data",
                                    split="trainval",
                                    target_types="category",
                                    transform=mean_std_transforms,
                                    download=True
                                )

    # Stratified 90/2\10 split which keeps the class distribution similar in training and validation.
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

    training_loader = DataLoader(
                                    training_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUMBER_OF_WORKERS
                                )

    rgb_channel_sum = torch.zeros(3)
    rgb_channel_sum_sqaured = torch.zeros(3)
    total_pixels = 0

    for training_images, training_labels in training_loader:
        # [batch_size, 3 (RGB), height, width]
        batch_size, channels, height, width = training_images.shape

        total_pixels += batch_size * height * width

        rgb_channel_sum += training_images.sum(dim=[0, 2, 3])
        rgb_channel_sum_sqaured += (training_images ** 2).sum(dim=[0, 2, 3])

    mean = rgb_channel_sum / total_pixels
    std = torch.sqrt((rgb_channel_sum_sqaured / total_pixels) - (mean ** 2))

    print("TRAINING_MEAN =", [round(value.item(), 6) for value in mean])
    print("TRAINING_STD =", [round(value.item(), 6) for value in std])


if __name__ == "__main__":
    main()