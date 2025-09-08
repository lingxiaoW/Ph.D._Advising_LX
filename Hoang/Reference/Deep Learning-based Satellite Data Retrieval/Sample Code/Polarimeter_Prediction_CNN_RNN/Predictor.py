import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from sklearn.metrics import r2_score, mean_squared_error


# Define to use CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prediction(test_dataloader, cnn_model, plot_row, plot=False):
    """
    Makes predictions using the DNN model, compares with ground truth labels,
    and generates plots for each output label if plot=True.

    Parameters:
    - test_dataloader (DataLoader): DataLoader for the test dataset.
    - dnn_model (torch.nn.Module): The trained DNN model.
    - plot (bool): Whether to generate plots for each output label.

    Returns:
    - predictions (list of tensors): Predicted values for the entire test set.
    - ground_truths (list of tensors): Ground truth labels for the entire test set.
    """
    cnn_model.eval()  # Set the model to evaluation mode
    cnn_model.to(device)    # move the model to GPU
    predictions = []
    ground_truths = []

    criterion = nn.MSELoss()
    loss = 0

    # No gradient calculation during testing
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = cnn_model(inputs)

            # Calculate the total loss on the testing data
            loss += criterion(outputs, targets)

            predictions.append(outputs)
            ground_truths.append(targets)

    # Combine all batches into single tensors
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

    r_square_list = []
    mse_list = []
    for i in range(22):
        r2 = r2_score(ground_truths[:, i], predictions[:, i])
        mse = mean_squared_error(ground_truths[:, i], predictions[:, i])
        r_square_list.append(r2)
        mse_list.append(mse)


    # Present the total loss in MSE:
    print(f"Total MSE Loss is {loss}")

    # Plotting if enabled
    if plot:
        num_outputs = 22  # Number of output labels

        # Calculate rows and columns for subplots
        n_rows = plot_row
        n_cols = (num_outputs + n_rows - 1) // n_rows  # Ceiling division to fit all outputs

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        axes = axes.flatten()  # Flatten axes array for easier iteration

        titles = ['l556', 'l385', 'l396', 'l413', 'l441', 'l470', 'l533', 'l549', 'l669', 'l759', 'l873',
                  'DoLP 556', 'DoLP 385', 'DoLP 396', 'DoLP 413', 'DoLP 441', 'DoLP 470', 'DoLP 533', 'DoLP 549',
                  'DoLP 669', 'DoLP 759', 'DoLP 873']
        for i in range(22):
            ax = axes[i]
            ax.scatter(ground_truths[:, i], predictions[:, i], alpha=0.6, edgecolors='b')
            ax.plot([ground_truths[:, i].min(), ground_truths[:, i].max()],
                    [ground_truths[:, i].min(), ground_truths[:, i].max()],
                    color='red', linestyle='--', linewidth=1)  # Line y = x
            ax.text(
                0.95, 0.05,  # x, y position (relative to axes)
                f"R: {r_square_list[i]:.5f}",  # Text for each subfigure
                ha='right',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                transform=ax.transAxes,  # Use axes-relative coordinates
                fontsize=12  # Adjust font size as needed
            )
            ax.set_title(titles[i])
            # ax.set_xlabel("Ground Truth")
            # ax.set_ylabel("Prediction")
            # ax.grid(True)

        # Turn off unused subplots
        for j in range(num_outputs, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def get_processing_time(test_dataloader, model):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # move the model to GPU
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            start_time = time.time()
            # Forward pass
            outputs = model(inputs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            break


    print(f"Elapsed Time is: {elapsed_time} seconds")