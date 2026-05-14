import torch
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_pipeline import (
                            OxfordPetTrimapDataset,
                            BATCH_SIZE,
                            SPLIT_SEED,
                            VALIDATION_RATIO,
                            NUMBER_OF_WORKERS
                          )

from model import OxfordPetCNN


MODEL_SAVE_PATH = "model.pth"
NUM_CLASSES = 37


def evaluate_training_set(model, training_loader, device):
    model.eval()  # evaluation mode

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # disable gradient tracking
        for batch_images, batch_labels in training_loader:  # loop through the training dataset
            batch_images = batch_images.to(device, non_blocking=True)  # moves images to my selected device (cuda) efficiently
            batch_labels = batch_labels.to(device, non_blocking=True)  # moves labels to my selected device (cuda) efficiently

            output_logits = model(batch_images)  # sends the images through the CNN

            predicted_classes = output_logits.argmax(dim=1)  # predicted class for each image
            correct_predictions += (predicted_classes == batch_labels).sum().item()  # count the correct predictions
            total_predictions += batch_labels.size(0)  # count the total images

    accuracy = correct_predictions / total_predictions

    return accuracy


def main():
    # Check if a GPU is available to use instead
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # loads the trainval labels so the official trainval split can be split into training and validation sets
    label_trainval_data = OxfordIIITPet(
                                            root="./data",
                                            split="trainval",
                                            target_types="category",
                                            transform=None,
                                            download=True,
                                        )

    # Stratified 95/5 split which keeps the class distribution similar in training and validation.
    # I did this as it helps make sure all 37 breeds are represented fairly in both the training and validation sets.
    all_image_position_indices = list(range(len(label_trainval_data)))  # list of all image positions in the trainval_data dataset.
    all_image_labels = label_trainval_data._labels  # the category label for every image, used for stratified splitting

    training_indices, validation_indices = train_test_split(  # split the position of the indices
        all_image_position_indices,
        test_size=VALIDATION_RATIO,  # 5% which is the validation set
        random_state=SPLIT_SEED,  # reproducible split with the seed
        stratify=all_image_labels  # preserves the class distribution
    )

    # Loads the training dataset again but with training=False.
    # This means there is no random crop and no random horizontal flip.
    # Lecturer (James Walker) said Q14 should use the training part without data augmentation, as augmentation could affect the score
    # in the course discussions
    training_data_no_augmentation = OxfordPetTrimapDataset(
                                                            root="./data",
                                                            split="trainval",
                                                            indices=training_indices,
                                                            training=False,
                                                            download=True,
                                                          )

    training_loader_no_augmentation = DataLoader(
                                                    training_data_no_augmentation,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=NUMBER_OF_WORKERS,
                                                    pin_memory=True
                                                )

    model = OxfordPetCNN(num_classes=NUM_CLASSES).to(device)  # create the CNN
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))  # Load learned weights

    training_accuracy = evaluate_training_set(
                                                    model,
                                                    training_loader_no_augmentation,
                                                    device
                                               )

    print(
        f"\nTraining subset evaluation complete\n"
        f"Training subset size: {len(training_data_no_augmentation)}\n"
        f"Training accuracy without augmentation: {training_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()