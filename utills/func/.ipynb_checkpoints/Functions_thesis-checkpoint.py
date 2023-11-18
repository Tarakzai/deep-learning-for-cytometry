import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import FlowCal
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fcsy import DataFrame
from PIL import Image
from tqdm import tqdm
import time
from tqdm.notebook import tqdm_notebook
import PIL
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from openTSNE import TSNE
import torchvision.models as models
import sklearn.metrics as metrics
from utills import tsne_functions
from utills import tsne_multiprocess.process_file_tsne
#from tsne_multiprocess import process_file_tsne
import multiprocessing
from functools import partial























#All function to be used

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 24 * 24, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(x)
        return x








class ModifiedCNN(nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=False)

        # Adjust input channels of the first convolutional layer
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the last fully connected layer of ResNet
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)

        # Add additional layers
        self.fc = nn.Linear(512, 2)  # Adjust the input size based on the ResNet output and your desired output size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


    
class ModifiedCNNT(nn.Module):
    def __init__(self):
        super(ModifiedCNNT, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)

        # Adjust input channels of the first convolutional layer
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the last fully connected layer of ResNet
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)

        # Add additional layers
        self.fc = nn.Linear(512, 2)  # Adjust the input size based on the ResNet output and your desired output size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x










def train_model(data_loader, learning_rate, num_epochs, model):
    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(torch.float)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in data_loader:
            inputs = inputs.to(torch.float)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    
    return model

def evaluate_model(test_loader, model):
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists 
    predictions = []
    labels = []
    probabilities = []

    # Loop over data
    for inputs, true_labels in test_loader:
        # Make predictions
        inputs = inputs.to(torch.float)
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # apply softmax function to get probas
            _, predicted = torch.max(probs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(true_labels.cpu().numpy())
            probabilities.extend(probs[:, 1].detach().cpu().numpy())  # detach the tensor and convert to numpy

    # Convert the lists of predictions, labels, and probabilities to NumPy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)

    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    # Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)

    # Calculate the area under the curve
    auc_score = auc(fpr, tpr)
    print('AUC: {:.2f}'.format(auc_score))

    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specificity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    
def min_max_tsne_multiprocess(df, path, column=None):
    Std_scl_test = StandardScaler().fit_transform(df)
    tsne = TSNE(perplexity=30, metric="euclidean", n_jobs=-1, n_components=2, random_state=42, verbose=False)
    transformation = tsne.fit(Std_scl_test)
    df_min_max = pd.DataFrame(columns=['X_min', 'X_max', 'Y_min', 'Y_max'])

    pool = multiprocessing.Pool()
    process_func = partial(process_file_tsne, column=column, transformation=transformation)
    results = pool.map(process_func, (os.path.join(path, filename) for filename in tqdm(os.listdir(path)) if 
                                      os.path.isfile(os.path.join(path, filename))))
    pool.close()
    pool.join()

    for result in results:
        if result is not None:
            df_min_max = df_min_max.append(result, ignore_index=True)

    return df_min_max['X_min'].min(), df_min_max['X_max'].max(), df_min_max['Y_min'].min(), df_min_max['Y_max'].max()




def sampled_cells_tsne(file_path,sample_size,column=None):
    '''
    file_path   : Path to the .fcs files
    sample_size : The number of cells to be sampled from each patient
    
    '''
    directory = file_path
    all_dfs = []
    
    for filename in tqdm(os.listdir(directory)):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if not f.endswith('.csv') and not f.endswith('.DS_Store'):
                df_dummy = DataFrame.from_fcs(f)
                new_cols=[]
                for columns in df_dummy.columns:
                    start_index=columns.find('(')
                    if start_index!=-1 :
                        new_name=columns[start_index:]
                    else:
                        new_name=columns
                    new_cols.append(new_name)
    
                df_dummy.columns=new_cols
                df_dummy
                if column is not None:
                    df_dummy.drop(column,axis=1,inplace=True)
                    
                if len(df_dummy) > sample_size:
                    sample_idx = np.random.choice(len(df_dummy), size=sample_size, replace=False)
                    df_sampled = df_dummy.iloc[sample_idx, :]
                else:
                    df_sampled = df_dummy
                all_dfs.append(df_sampled)
                
    df_selected_random = pd.concat(all_dfs, axis=0, ignore_index=True)
    return df_selected_random





def tsne_2darrays(path, df, x_min, x_max, y_min, y_max, bins, sample_ratio=None, column=None):
    scale = StandardScaler().fit_transform(df)
    tsne = TSNE(perplexity=30, metric="euclidean", n_jobs=-1, n_components=2, random_state=42, verbose=False)
    transformation = tsne.fit(scale)

    pool = multiprocessing.Pool()
    process_func = partial(tsne_functions.process_file_tsne_2darrays, transformation, x_min=x_min, x_max=x_max, y_min=y_min, 
                           y_max=y_max, bins=bins, sample_ratio=sample_ratio, column=column)
    results = pool.map(process_func, (os.path.join(path, filename) for filename in os.listdir(path) if 
                                      os.path.isfile(os.path.join(path, filename))))
    pool.close()
    pool.join()

    hist_arrays = []
    save_names = []

    for result in results:
        if result is not None:
            hist_arrays.append(result['hist_array'])
            save_names.append(result['fcs_file'])

    hist_df = pd.DataFrame({'hist_array': hist_arrays, 'fcs_file': save_names})

    return hist_df



class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.ToTensor() 
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        array = self.dataframe.iloc[index]['hist_array']
        label = self.dataframe.iloc[index]['label']  
        
        # Convert array to tensor
        tensor = self.transform(array)
        
        return tensor, label
    
    
    
def Confusion_matrix(data_loader,trained_model):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in tqdm_notebook(data_loader,desc="Confusion_matrix"):
        inputs = inputs.to(torch.float)
        
        
        output = trained_model(inputs) # Feed Network
            
    
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
            
    return confusion_matrix(y_true, y_pred)