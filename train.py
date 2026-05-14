import random
import time # for measuring epoch duration

import numpy as np
import torch
import torch.nn as nn # for the loss function
import torch.optim as optim # for the optimiser
from tqdm import tqdm # for progress bars

from data_pipeline import get_data_loaders
from model import OxfordPetCNN

MODEL_SAVE_PATH = "model.pth"
NUM_CLASSES = 37
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MAX_LEARNING_RATE = 0.002
BATCH_SIZE = 16
CHOSEN_SEED = 67

# Python, NumPy, and PyTorch should start the random generator from the same point for reproducibility
def set_seed(seed=CHOSEN_SEED):
    random.seed(seed)                  # controls Python’s built-in randomness.
    np.random.seed(seed)               # controls NumPy randomness.
    torch.manual_seed(seed)            # controls PyTorch randomness on CPU
    torch.cuda.manual_seed_all(seed)   # controls PyTorch randomness on GPU.
    # deterministic calculations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, training_loader, loss_function, optimiser, device, scheduler):
    model.train() # Training mode

    total_loss_across_epoch = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_images, batch_labels in tqdm(training_loader, desc="Training"): #loop with a progress bar
        batch_images = batch_images.to(device, non_blocking=True) # moves images to my selected device (cuda) efficiently
        batch_labels = batch_labels.to(device, non_blocking=True) # moves labels to my selected device (cuda) efficiently

        optimiser.zero_grad() # clear old gradients (PyTorch accumulates gradients by default)
        output_logits = model(batch_images) # sends the images through the CNN
        loss = loss_function(output_logits, batch_labels) # (training loss) compares the model’s predictions with the correct labels

        loss.backward() # backpropagation - calculates the gradients
        optimiser.step() # update the weights

        scheduler.step() # Adjusts the learning rate after every batch to follow the smooth 1-Cycle ramp-up and cool-down curve.

        # loss.item() = plain number value of the loss
        # batch_images.size(0) = number of images in the batch.
        total_loss_across_epoch += loss.item() * batch_images.size(0)

        predicted_classes = output_logits.argmax(dim=1) # predicted class for each image.
        correct_predictions += (predicted_classes == batch_labels).sum().item() # count the correct predictions
        total_predictions += batch_labels.size(0) # count the total images

    average_loss = total_loss_across_epoch / total_predictions
    accuracy = correct_predictions / total_predictions

    return average_loss, accuracy

def evaluate(model, validation_loader, loss_function, device):
    model.eval() # evaluation mode

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # disable gradient tracking
        for batch_images, batch_labels in tqdm(validation_loader, desc="Validation"): #loop with a progress bar
            batch_images = batch_images.to(device, non_blocking=True) # moves images to my selected device (cuda) efficiently
            batch_labels = batch_labels.to(device, non_blocking=True) # moves labels to my selected device (cuda) efficiently

            output_logits = model(batch_images) # sends the images through the CNN
            loss = loss_function(output_logits, batch_labels) # (validation loss) compares the model’s predictions with the correct labels
            total_loss += loss.item() * batch_images.size(0) # total loss

            predicted_classes = output_logits.argmax(dim=1) # predicted class for each image.
            correct_predictions += (predicted_classes == batch_labels).sum().item() # count the correct predictions
            total_predictions += batch_labels.size(0) # count the total images

    average_loss = total_loss / total_predictions
    accuracy = correct_predictions / total_predictions

    return average_loss, accuracy

def main():
    set_seed()

    # Check if a GPU is available to use instead
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # get the data
    training_loader, validation_loader, _ = get_data_loaders(
                                                              batch_size=BATCH_SIZE
                                                            )


    model = OxfordPetCNN(num_classes=NUM_CLASSES).to(device) # create the CNN
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1) # suitable here because its multi-class classification

    optimiser = optim.AdamW( # Adaptive moment estimation with weight decay
                                model.parameters(),
                                lr=LEARNING_RATE,
                                weight_decay=1e-3
                            )

    # OneCycleLR trains faster in fewer epochs (30 epoch limit)
    # 1-Cycle Policy rapidly increases the learning rate to a peak taking the model out of a bad minima
    # It then reduces the learning rate to nearly zero , so it can refine the weights
    scheduler = optim.lr_scheduler.OneCycleLR(
                                                optimiser,
                                                max_lr=MAX_LEARNING_RATE,  # ramps-up to a max to break local minima then back down to refine
                                                epochs=NUM_EPOCHS,
                                                steps_per_epoch=len(training_loader), # number of batches in one epoch
                                                # pct_start - the model spends 40% of the time (12 epochs) climbing up to maximum speed
                                                pct_start = 0.4  # Shifts the peak to 40% of the run
                                            )

    best_validation_accuracy = 0.0
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1} of {NUM_EPOCHS}")
        start_time = time.time()

        # Train the model for one epoch
        training_loss, training_accuracy = train_one_epoch(
                                                            model,
                                                            training_loader,
                                                            loss_function,
                                                            optimiser,
                                                            device,
                                                            scheduler
                                                          )

        # Evaluate the model on the validation set
        validation_loss, validation_accuracy = evaluate(
                                                            model,
                                                            validation_loader,
                                                            loss_function,
                                                            device
                                                       )

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"\nTraining loss (4 decimal places): {training_loss:.4f}\n"
            f"Training accuracy: {training_accuracy * 100:.2f}%"
        )

        print(
            f"Validation loss (4 decimal places): {validation_loss:.4f}\n"
            f"Validation accuracy: {validation_accuracy * 100:.2f}%"
        )

        print(f"Epoch time: {epoch_time:.2f} seconds")
        current_learning_rate = optimiser.param_groups[0]["lr"]     # reads the learning rate being used in the current epoch
        print(f"Learning rate at the end of the epoch: {current_learning_rate:.6e}")

        # Save the model if validation accuracy improves
        # If validation accuracy is tied, save the model with the lower validation loss
        if ((validation_accuracy > best_validation_accuracy) or
            (validation_accuracy == best_validation_accuracy and validation_loss < best_validation_loss)):
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved the new best model to {MODEL_SAVE_PATH}")


        updated_learning_rate = optimiser.param_groups[0]["lr"]  # reads the learning rate that will be used next epoch
        print(f"Learning rate going into next epoch: {updated_learning_rate:.6e}")


    print(
        f"\nTraining complete\n"
        f"Final training accuracy: {training_accuracy * 100:.2f}%\n"
        f"The best validation accuracy: {best_validation_accuracy * 100:.2f}%\n"
        f"The best validation loss: {best_validation_loss:.4f}\n"
        f"The saved best model epoch: {best_epoch} out of {NUM_EPOCHS}"
         )


if __name__ == "__main__":
    main()
