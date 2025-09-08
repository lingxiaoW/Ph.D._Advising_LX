import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training function
def train_model(model, train_loader, val_loader, save_path):
    epochs = 200

    # put the model to GPU
    model.to(device)

    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    best_loss = math.inf
    early_stop_count = 0
    early_stop = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, targets in train_loader:
            # move data and targets to the GPU
            data, targets = data.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(data)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass: compute the gradients
            loss.backward()

            # Update the weights
            optimizer.step()

            # Accumulate the training loss
            train_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(train_loader)
        train_loss_values.append(avg_train_loss)

        # Evaluate on the validation set
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Calculate average validation loss for this epoch
        avg_test_loss = val_loss / len(val_loader)
        test_loss_values.append(avg_test_loss)

        # Store the epoch number
        epoch_count.append(epoch + 1)

        # Print the average losses for this epoch
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}')

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