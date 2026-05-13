import torch
import torch.nn as nn # for the loss function
from tqdm import tqdm # for progress bars

from data_pipeline import get_data_loaders
from model import OxfordPetCNN


MODEL_SAVE_PATH = "model.pth"
NUM_CLASSES = 37
BATCH_SIZE = 16

def evaluate_test(model, test_loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_images, batch_labels in tqdm(test_loader, desc="Testing"): #loop with a progress bar for testing
            batch_images = batch_images.to(device, non_blocking=True)  #moves images to my selected device (cuda) efficiently
            batch_labels = batch_labels.to(device, non_blocking=True)  #moves images to my selected device (cuda) efficiently

            output_logits = model(batch_images) # sends the images through the CNN
            loss = loss_function(output_logits, batch_labels) # (test loss) compares the model’s predictions with the correct labels

            total_loss += loss.item() * batch_images.size(0)

            predicted_classes = output_logits.argmax(dim=1)  # predicted class for each image.
            correct_predictions += (predicted_classes == batch_labels).sum().item()  # count the correct predictions
            total_predictions += batch_labels.size(0)  # count the total images

    average_loss = total_loss / total_predictions
    accuracy = correct_predictions / total_predictions

    return average_loss, accuracy


def main():
    # Check if a GPU is available to use instead
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # get the data
    _, _, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    model = OxfordPetCNN(num_classes=NUM_CLASSES).to(device) # create the CNN
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device)) # Load learned weights

    loss_function = nn.CrossEntropyLoss() # suitable here because its multi-class classification

    # Evaluate the model on the test set
    test_loss, test_accuracy = evaluate_test(
                                                model,
                                                test_loader,
                                                loss_function,
                                                device
                                            )

    print(
        f"\nOfficial test complete\n"
        f"Test loss (4 decimal places): {test_loss:.4f}\n"
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()