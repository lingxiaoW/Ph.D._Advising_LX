import torch

import os
import glob
from LoadDataSet import load_data

from DNNs import RegressionModel, DNNWithTwoHeads
from Trainer import training, training_doublehead
from Predictor import prediction, prediction_doublehead, get_processing_time

# Define to use CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define whether train the model
training_flag = False
# Define using double head model ('DoubleHead') or not ('Regression_Model')
# model_selection = 'DoubleHead'
model_selection = 'Regression_Model'
# Define to plot the diagram or not
plot_flag = False

# Load DataSet
data_path = './../../Dataset/input-output-pairs'
pace318_train = glob.glob(os.path.join(data_path, 'pace318_train_set', 'H5', '*.h5'))
pace318_test = glob.glob(os.path.join(data_path, 'pace318_test_set', 'H5', '*.h5'))

pace325_train = glob.glob(os.path.join(data_path, 'pace325_train_set', 'H5', '*.h5'))
pace325_test = glob.glob(os.path.join(data_path, 'pace325_test_set', 'H5', '*.h5'))

# Create the dataset and dataloader
train_dataloader, val_dataloader, test_dataloader = load_data(pace318_train[0:10], pace318_train[11], model_selection)


# Define the model
input_size = 14  # Input shape (11 input features + 3 geo features)
if model_selection == 'DoubleHead':
    model = DNNWithTwoHeads(input_size).to(device)
else:
    output_size = 22  # Output shape (22 features)
    model = RegressionModel(input_size, output_size).to(device)

# Define a path to save the model
model_path = f'./Models/RegressionModels/{model_selection}.pth'


if training_flag == True:
    if model_selection == 'DoubleHead':
        training_doublehead(train_dataloader, val_dataloader, model, model_path)
    else:
        # train the model from scratch
        training(train_dataloader, val_dataloader, model, model_path)
else:
    # load previously trained model
    model.load_state_dict(torch.load(model_path, map_location=device))

# Show result on testing data
if model_selection == 'DoubleHead':
    prediction_doublehead(test_dataloader, model, plot=plot_flag)
else:
    prediction(test_dataloader, model, plot=plot_flag)
    get_processing_time(test_dataloader, model)

