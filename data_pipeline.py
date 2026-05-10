# This python file serves the purpose to:
"""
1) Load the dataset
2) use transforms on the data
3) split it into training data and validation data
4) Use the DataLoaders to load the images in batches
"""
import torch
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset # subset is a pointer tool
from sklearn.model_selection import train_test_split


def main():
    training_mean = [0.479322, 0.446042, 0.395331]
    training_std = [0.255698, 0.250303, 0.258181]

    image_transformation = transforms.Compose([
                                                 transforms.Resize((64, 64)), # small and fast training
                                                 transforms.ToTensor(),
                                                 transforms.Normalize( mean=training_mean,
                                                                       std=training_std
                                                                     )
                                               ])

    trainval_data = OxfordIIITPet(     # Load official trainval dataset
        root="./data",
        split="trainval",
        target_types="category",
        transform=image_transformation,
        download=True,
    )

    test_data = OxfordIIITPet( # Load official test dataset
        root="./data",
        split="test",
        target_types="category",
        transform=image_transformation,
        download=True,
    )

    # Stratified 80/20 split which keeps the class distribution similar in training and validation.
    # I did this as it helps make sure all 37 breeds are represented fairly in both the training and validation sets.
    all_image_pos_indicies = list(range(len(trainval_data))) # list of all image positions in the trainval_data dataset.
    all_image_labels = trainval_data._labels # the class label for every image in trainval_data
    
    training_indices, validation_indices = train_test_split(  # split the position of the indices
        all_image_pos_indicies,
        test_size=0.2, #20% in validation set
        random_state=67, # reproducible split with the seed
        stratify=all_image_labels # preservers the class distribution
    )

    training_data = Subset(trainval_data, training_indices)
    validation_data = Subset(trainval_data, validation_indices)




    """ Random split - i replaced this with Stratifdied ssplit instead
        #fixed random generator for the train/validation split
        fixed_random_gen = torch.Generator().manual_seed(67) #random shuffle pattern - reproducible split
    
        # trainval split into 80% training and 20 % validation - common values used in Deep learning textbook
        # allows us to maintain the validation set as a point of comparison
        training_size = int(0.8 * len(trainval_data))
        validation_size = len(trainval_data) - training_size
    
        # perform the split on the trainval data into 80%/20% with the same random seed
        training_data, validation_data = random_split(
                                            trainval_data,
                                    [training_size, validation_size],
                                            generator=fixed_random_gen
                                            )
        
    """

    # Data Loaders
    training_loader = DataLoader(
        training_data,
        batch_size=32, #baseline value
        shuffle=True,
        num_workers=0 #baseline value
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=32, #baseline value
        shuffle=False,
        num_workers=0 #baseline value
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32, #baseline value
        shuffle=False,
        num_workers=0 #baseline value
    )

    # python can iterate through batches of data
    batch_images, batch_labels = next(iter(training_loader))
    print(
        f"Training size: {len(training_data)}\n"
        f"Validation size: {len(validation_data)}\n"
        f"Test size: {len(test_data)}\n"
        f"Training batches: {len(training_loader)}\n"
        f"Validation batches: {len(validation_loader)}\n"
        f"Test batches: {len(test_loader)}\n"
        f"Batch image tensor shape: {batch_images.shape}\n"
        f"Batch label tensor shape: {batch_labels.shape}"
    )

if __name__ == "__main__":
    main()