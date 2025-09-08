import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error

# Define to use CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prediction_doublehead(test_dataloader, dnn_model, plot=False):
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
    dnn_model.eval()  # Set the model to evaluation mode

    predictions_head1 = []
    predictions_head2 = []
    ground_truths_head1 = []
    ground_truths_head2 = []

    # No gradient calculation during testing
    with torch.no_grad():
        for inputs, targets1, targets2 in test_dataloader:
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)

            # Forward pass
            output1, output2 = dnn_model(inputs)

            predictions_head1.append(output1)
            predictions_head2.append(output2)
            ground_truths_head1.append(targets1)
            ground_truths_head2.append(targets2)

    # Combine all batches into single tensors
    predictions_head1 = torch.cat(predictions_head1, dim=0).cpu().numpy()
    ground_truths_head1 = torch.cat(ground_truths_head1, dim=0).cpu().numpy()

    predictions_head2 = torch.cat(predictions_head2, dim=0).cpu().numpy()
    ground_truths_head2 = torch.cat(ground_truths_head2, dim=0).cpu().numpy()

    # Plotting if enabled
    if plot:
        num_outputs = 22  # Number of output labels

        # Calculate rows and columns for subplots
        n_rows = 4
        n_cols = (num_outputs + n_rows - 1) // n_rows  # Ceiling division to fit all outputs

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        axes = axes.flatten()  # Flatten axes array for easier iteration

        titles = ['l556', 'l385', 'l396', 'l413', 'l441', 'l470', 'l533', 'l549', 'l669', 'l759', 'l873',
                  'DoLP 556', 'DoLP 385', 'DoLP 396', 'DoLP 413', 'DoLP 441', 'DoLP 470', 'DoLP 533', 'DoLP 549',
                  'DoLP 669', 'DoLP 759', 'DoLP 873']
        for i in range(11):
            ax = axes[i]
            ax.scatter(ground_truths_head1[:, i], predictions_head1[:, i], alpha=0.6, edgecolors='b')
            ax.plot([ground_truths_head1[:, i].min(), ground_truths_head1[:, i].max()],
                    [ground_truths_head1[:, i].min(), ground_truths_head1[:, i].max()],
                    color='red', linestyle='--', linewidth=1)  # Line y = x
            ax.set_title(titles[i])
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.grid(True)

        for i in range(11):
            ax = axes[11 + i]
            ax.scatter(ground_truths_head2[:, i], predictions_head2[:, i], alpha=0.6, edgecolors='b')
            ax.plot([ground_truths_head2[:, i].min(), ground_truths_head2[:, i].max()],
                    [ground_truths_head2[:, i].min(), ground_truths_head2[:, i].max()],
                    color='red', linestyle='--', linewidth=1)  # Line y = x
            ax.set_title(titles[11 + i])
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Prediction")
            ax.grid(True)

        # Turn off unused subplots
        for j in range(num_outputs, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



def prediction(test_dataloader, dnn_model, plot=False):
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
    dnn_model.eval()  # Set the model to evaluation mode

    predictions = []
    ground_truths = []
    loss = 0.0
    criterion = nn.MSELoss()
    # No gradient calculation during testing
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = dnn_model(inputs)
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

    print(f"MSE Loss for FNN is: {loss}")

    # Plotting if enabled
    if plot:
        num_outputs = 22  # Number of output labels

        # Calculate rows and columns for subplots
        n_rows = 2
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


def get_processing_time(test_dataloader, dnn_model):
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            start_time = time.time()
            # Forward pass
            outputs = dnn_model(inputs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            break


    print(f"Elapsed Time is: {elapsed_time} seconds")