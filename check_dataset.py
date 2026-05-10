# Check that the Oxford-IIIT Pet dataset loads correctly.
import torchvision
import torchvision.transforms as transforms

def main():
    print("Oxford-IIIT Pet Dataset Check\n")

    image_preparation = transforms.ToTensor()

    trainval_data = torchvision.datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        transform=image_preparation,
        download=True,
    )

    test_data = torchvision.datasets.OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        transform=image_preparation,
        download=True,
    )

    image, label = trainval_data[0]

    print(
        f"Trainval images: {len(trainval_data)}\n"
        f"Test images: {len(test_data)}\n"
        f"Number of classes: {len(trainval_data.classes)}\n"
        f"Class count check: {'Correct' if len(trainval_data.classes) == 37 else 'Incorrect'}\n"
        f"First image tensor shape: {image.shape}\n"
        f"First label index: {label}\n"
        f"First class name: {trainval_data.classes[label]}"
    )

if __name__ == "__main__":
    main()