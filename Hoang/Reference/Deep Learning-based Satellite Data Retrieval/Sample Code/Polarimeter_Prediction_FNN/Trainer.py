import torch.nn as nn
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import math
import os
import torch.optim as optim

# Define to use CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def training_doublehead(train_dataloader, val_dataloader, model, save_path):

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Training Loop
    num_epochs = 200  # Number of training epochs

    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    best_loss = math.inf
    early_stop_count = 0
    early_stop = 20

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0  # Track the total loss for this epoch

        for inputs, targets1, targets2 in train_dataloader:
            # Move inputs and targets to the GPU (if available)
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            output1, output2 = model(inputs)

            # Compute the loss
            loss1 = criterion(output1, targets1)
            loss2 = criterion(output2, targets2)

            total_loss = loss1 + loss2

            # Backward pass: compute the gradients
            total_loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the training loss
            train_loss += total_loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)

        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0  # Track total test loss for this epoch

        with torch.inference_mode():  # No gradients needed
            for inputs, targets1, targets2 in val_dataloader:
                inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
                output1, output2 = model(inputs)
                loss1 = criterion(output1, targets1)
                loss2 = criterion(output2, targets2)

                total_loss = loss1 + loss2

                test_loss += total_loss.item()

        # Calculate average validation loss for this epoch
        avg_test_loss = test_loss / len(val_dataloader)
        test_loss_values.append(avg_test_loss)

        # Store the epoch number
        epoch_count.append(epoch + 1)

        # Print the average losses for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}')

        # If the test results on validation data is better than the history, save the model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saving Model with Loss {best_loss:.3f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        # If the model does not improve for a certain amount of times, break the training loop
        if early_stop_count >= early_stop:
            print('\nModel is not improving, stop the training')
            break

    # Complete the training and plot the training result
    # Plot train and test loss values
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Train and Test Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Set x-ticks to show integers only
    plt.xticks(ticks=[0, len(epoch_count)/2, len(epoch_count)])

    plt.show()


def training(train_dataloader, val_dataloader, model, save_path):

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Training Loop
    num_epochs = 200  # Number of training epochs

    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    best_loss = math.inf
    early_stop_count = 0
    early_stop = 20

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0  # Track the total loss for this epoch

        for inputs, targets in train_dataloader:
            # Move inputs and targets to the GPU (if available)
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            output = model(inputs)

            # Compute the loss
            loss = criterion(output, targets)

            # Backward pass: compute the gradients
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_dataloader)
        train_loss_values.append(avg_train_loss)

        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0  # Track total test loss for this epoch

        with torch.inference_mode():  # No gradients needed
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        # Calculate average validation loss for this epoch
        avg_test_loss = test_loss / len(val_dataloader)
        test_loss_values.append(avg_test_loss)

        # Store the epoch number
        epoch_count.append(epoch + 1)

        # Print the average losses for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}')

        # If the test results on validation data is better than the history, save the model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saving Model with Loss {best_loss:.3f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        # If the model does not improve for a certain amount of times, break the training loop
        if early_stop_count >= early_stop:
            print('\nModel is not improving, stop the training')
            break

    # Complete the training and plot the training result
    # Plot train and test loss values
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Train and Test Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Set x-ticks to show integers only
    plt.xticks(ticks=[0, len(epoch_count)/2, len(epoch_count)])

    plt.show()


