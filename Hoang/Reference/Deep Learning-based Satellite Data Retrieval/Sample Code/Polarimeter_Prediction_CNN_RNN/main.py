import torch
import os
import glob

from LoadDataSet import load_data
from CNNs import TimeSeriesCNN
from RNNs import TimeSeriesLSTM
from Trainer import train_model
from Predictor import prediction, get_processing_time

# Define to use CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define whether train the model
training_flag = False
# Define the model_type: 'CNN' or 'RNN'
model_type = 'RNN'
# Define to plot the diagram or not
plot_flag = False

# Load DataSet
data_path = './../../Dataset/input-output-pairs'
pace318_train = glob.glob(os.path.join(data_path, 'pace318_train_set', 'H5', '*.h5'))
pace318_test = glob.glob(os.path.join(data_path, 'pace318_test_set', 'H5', '*.h5'))

pace325_train = glob.glob(os.path.join(data_path, 'pace325_train_set', 'H5', '*.h5'))
pace325_test = glob.glob(os.path.join(data_path, 'pace325_test_set', 'H5', '*.h5'))

# Window_size defines combine how many data instances into the input side
window_size = 3

# Create the dataset and dataloader
train_dataloader, val_dataloader, test_dataloader = load_data(pace318_train[0:10], pace318_train[11], window_size)

# Define the model
input_size = 14
output_size = 22

if model_type == 'CNN':
    model = TimeSeriesCNN(input_size=input_size, output_size=output_size, window_size=window_size)
else:
    hidden_size = 256
    model = TimeSeriesLSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size)

# Define a path to save the model
model_path = f'./Models/{model_type}_Models/{model_type}_{window_size}.pth'

if training_flag == True:
    train_model(model=model, train_loader=train_dataloader, val_loader=val_dataloader, save_path=model_path)
else:
    # load previously trained model
    saved_model_path = f'./Saved_Models/{model_type}_{window_size}.pth'
    model.load_state_dict(torch.load(saved_model_path, map_location=device))

get_processing_time(test_dataloader, model)

prediction(test_dataloader=test_dataloader, cnn_model=model, plot_row=2, plot=plot_flag)
