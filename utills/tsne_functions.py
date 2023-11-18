import pandas as pd
import numpy as np 
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from fcsy import DataFrame
import os
import torch



def process_file_tsne_2darrays(tsne_transformation, path, x_min, x_max, y_min, y_max, bins, sample_ratio, column):
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_fcs = DataFrame.from_fcs(path)
        new_cols = []
        for columns in df_fcs.columns:
            start_index = columns.find('(')
            if start_index != -1:
                new_name = columns[start_index:]
            else:
                new_name = columns
            new_cols.append(new_name)

        df_fcs.columns = new_cols

        if column is not None:
            df_fcs.drop(column, axis=1, inplace=True)

        if sample_ratio is not None:
            df_fcs = df_fcs.sample(n=sample_ratio)

        Scaled = StandardScaler().fit_transform(df_fcs)
        new_transform = tsne_transformation.transform(Scaled)
        tsne_Df = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

        xbins = np.linspace(x_min, x_max, bins)
        ybins = np.linspace(y_min, y_max, bins)
        hist_array, _, _, _ = plt.hist2d(x=tsne_Df['t1'], y=tsne_Df['t2'], bins=[xbins, ybins])

        save_name = path.split('/')[-1]

        return {'hist_array': hist_array, 'fcs_file': save_name}

    return None



def process_file_sampled(tsne_transformation, path, x_min, x_max, y_min, y_max, bins, sample_ratio, column):
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_fcs = DataFrame.from_fcs(path)
        new_cols = []
        for columns in df_fcs.columns:
            start_index = columns.find('(')
            if start_index != -1:
                new_name = columns[start_index:]
            else:
                new_name = columns
            new_cols.append(new_name)

        df_fcs.columns = new_cols

        if column is not None:
            df_fcs.drop(column, axis=1, inplace=True)

        if sample_ratio is not None:
            sampled_dfs = []
            for _ in range(3):  # Perform sampling three times
                sampled_df = df_fcs.sample(n=sample_ratio, replace=False)
                sampled_dfs.append(sampled_df)

            hist_arrays = []
            save_names = []

            for i, sampled_df in enumerate(sampled_dfs):
                Scaled = StandardScaler().fit_transform(sampled_df)
                new_transform = tsne_transformation.transform(Scaled)
                tsne_Df = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

                xbins = np.linspace(x_min, x_max, bins)
                ybins = np.linspace(y_min, y_max, bins)
                hist_array, _, _, _ = plt.hist2d(x=tsne_Df['t1'], y=tsne_Df['t2'], bins=[xbins, ybins])

                save_name = f"{path.split('/')[-1]}_sample{i+1}"

                hist_arrays.append(hist_array)
                save_names.append(save_name)

            return {'hist_array': hist_arrays, 'fcs_file': save_names}

    return None


def process_file_tsne_2darrays_multi(tsne_transformation, path, bins, dimension,sample_ratio, column):
    
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_fcs = DataFrame.from_fcs(path)
        new_cols = []
        for columns in df_fcs.columns:
            start_index = columns.find('(')
            if start_index != -1:
                new_name = columns[start_index:]
            else:
                new_name = columns
            new_cols.append(new_name)

        df_fcs.columns = new_cols

        if column is not None:
            df_fcs.drop(column, axis=1, inplace=True)

        if sample_ratio is not None:
            df_fcs = df_fcs.sample(n=sample_ratio)

        Scaled = StandardScaler().fit_transform(df_fcs)
        new_transform = tsne_transformation.transform(Scaled)
        tsne_Df = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

        
        _, x_edges, y_edges, _ = plt.hist2d(x=tsne_Df['t1'], y=tsne_Df['t2'], bins=bins,cmap='Blues')
        
        x_indices = np.digitize(tsne_Df['t1'], x_edges)
        y_indices = np.digitize(tsne_Df['t2'], y_edges)
        df_pixels = df_fcs
        pixel_coordinates = np.column_stack((x_indices, y_indices))
        df_pixels[['pixel_coordinates_x', 'pixel_coordinates_y']] = pixel_coordinates
        df_mean = df_pixels.groupby(['pixel_coordinates_x', 'pixel_coordinates_y']).mean()
        df_mean.reset_index(inplace=True)
        
        images = []
        x = df_mean['pixel_coordinates_x']
        y = df_mean['pixel_coordinates_y']
        grid_size = max(max(x), max(y)) + 1
        image = np.zeros((grid_size, grid_size))
        for i in df_mean.drop(['pixel_coordinates_x', 'pixel_coordinates_y'], axis=1).columns:
            pixel_values = df_mean[i]
            for x_cord, y_cord, pixel_value in zip(x, y, pixel_values):
                image[y_cord, x_cord] = pixel_value
            images.append(image.copy())  # To avoid overwriting
        image_tensor = np.empty((dimension, 102, 102), dtype=np.float32)
        for i, image in enumerate(images):
            image_tensor[i] = image
        tensor_image = torch.from_numpy(image_tensor)
        

        save_name = path.split('/')[-1]

        return {'Tensor_image': tensor_image, 'fcs_file': save_name}

    return None




def process_file_tsne_2darrays_multi_cellcount(tsne_transformation, path, bins, dimension,sample_ratio, column):
    
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_fcs = DataFrame.from_fcs(path)
        new_cols = []
        for columns in df_fcs.columns:
            start_index = columns.find('(')
            if start_index != -1:
                new_name = columns[start_index:]
            else:
                new_name = columns
            new_cols.append(new_name)

        df_fcs.columns = new_cols

        if column is not None:
            df_fcs.drop(column, axis=1, inplace=True)

        if sample_ratio is not None:
            df_fcs = df_fcs.sample(n=sample_ratio)

        Scaled = StandardScaler().fit_transform(df_fcs)
        new_transform = tsne_transformation.transform(Scaled)
        tsne_Df = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

        
        hist, x_edges, y_edges, _ = plt.hist2d(x=tsne_Df['t1'], y=tsne_Df['t2'], bins=bins,cmap='Blues')
        
        x_indices = np.digitize(tsne_Df['t1'], x_edges)
        y_indices = np.digitize(tsne_Df['t2'], y_edges)
        df_pixels = df_fcs
        pixel_coordinates = np.column_stack((x_indices, y_indices))
        df_pixels[['pixel_coordinates_x', 'pixel_coordinates_y']] = pixel_coordinates
        df_mean = df_pixels.groupby(['pixel_coordinates_x', 'pixel_coordinates_y']).mean()
        df_mean.reset_index(inplace=True)
        
        #Resizing the hist to the multi channel and adding it to the images !!!
                
        # Resize the histogram to 102x102 using nearest-neighbor interpolation
        new_shape = (102, 102)
        resized_histogram = np.zeros(new_shape)
        
        # Calculate scaling factors for rows and columns
        row_scale = new_shape[0] / hist.shape[0]
        col_scale = new_shape[1] / hist.shape[1]
        
        # Fill in the new histogram using nearest-neighbor interpolation
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                orig_row = int(i / row_scale)
                orig_col = int(j / col_scale)
                resized_histogram[i, j] = hist[orig_row, orig_col]
        
        # Now, resized_histogram is a 102x102 histogram with data from the original histogram


        
        images = []
        x = df_mean['pixel_coordinates_x']
        y = df_mean['pixel_coordinates_y']
        grid_size = max(max(x), max(y)) + 1
        image = np.zeros((grid_size, grid_size))
        for i in df_mean.drop(['pixel_coordinates_x', 'pixel_coordinates_y'], axis=1).columns:
            pixel_values = df_mean[i]
            for x_cord, y_cord, pixel_value in zip(x, y, pixel_values):
                image[y_cord, x_cord] = pixel_value
            images.append(image.copy())  # To avoid overwriting
            
        images.append(resized_histogram)
        
        image_tensor = np.empty((dimension, 102, 102), dtype=np.float32)
        for i, image in enumerate(images):
            image_tensor[i] = image
        tensor_image = torch.from_numpy(image_tensor)
        

        save_name = path.split('/')[-1]

        return {'Tensor_image': tensor_image, 'fcs_file': save_name}

    return None




            
            
            
def process_file_tsne_2darrays_multi_cell_logcount(tsne_transformation, path, bins, dimension,sample_ratio, column):
    small_value = 1e-10
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_fcs = DataFrame.from_fcs(path)
        new_cols = []
        for columns in df_fcs.columns:
            start_index = columns.find('(')
            if start_index != -1:
                new_name = columns[start_index:]
            else:
                new_name = columns
            new_cols.append(new_name)

        df_fcs.columns = new_cols

        if column is not None:
            df_fcs.drop(column, axis=1, inplace=True)

        if sample_ratio is not None:
            df_fcs = df_fcs.sample(n=sample_ratio)

        Scaled = StandardScaler().fit_transform(df_fcs)
        new_transform = tsne_transformation.transform(Scaled)
        tsne_Df = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

        
        hist, x_edges, y_edges, _ = plt.hist2d(x=tsne_Df['t1'], y=tsne_Df['t2'], bins=bins,cmap='Blues')
        
        x_indices = np.digitize(tsne_Df['t1'], x_edges)
        y_indices = np.digitize(tsne_Df['t2'], y_edges)
        df_pixels = df_fcs
        pixel_coordinates = np.column_stack((x_indices, y_indices))
        df_pixels[['pixel_coordinates_x', 'pixel_coordinates_y']] = pixel_coordinates
        df_mean = df_pixels.groupby(['pixel_coordinates_x', 'pixel_coordinates_y']).mean()
        df_mean.reset_index(inplace=True)
        
        #Resizing the hist to the multi channel and adding it to the images !!!
                
        # Resize the histogram to 102x102 using nearest-neighbor interpolation
        new_shape = (102, 102)
        resized_histogram = np.zeros(new_shape)
        
        # Calculate scaling factors for rows and columns
        row_scale = new_shape[0] / hist.shape[0]
        col_scale = new_shape[1] / hist.shape[1]
        
        # Fill in the new histogram using nearest-neighbor interpolation
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                orig_row = int(i / row_scale)
                orig_col = int(j / col_scale)
                resized_histogram[i, j] = hist[orig_row, orig_col]
        
        # Now, resized_histogram is a 102x102 histogram with data from the original histogram

        hist_array_with_small_value = resized_histogram + small_value
            
        # Apply logarithm to the histogram arrays
        log_hist_array = np.log(hist_array_with_small_value)
        
        images = []
        x = df_mean['pixel_coordinates_x']
        y = df_mean['pixel_coordinates_y']
        grid_size = max(max(x), max(y)) + 1
        image = np.zeros((grid_size, grid_size))
        for i in df_mean.drop(['pixel_coordinates_x', 'pixel_coordinates_y'], axis=1).columns:
            pixel_values = df_mean[i]
            for x_cord, y_cord, pixel_value in zip(x, y, pixel_values):
                image[y_cord, x_cord] = pixel_value
            images.append(image.copy())  # To avoid overwriting
            
        images.append(log_hist_array)
        
        image_tensor = np.empty((dimension, 102, 102), dtype=np.float32)
        for i, image in enumerate(images):
            image_tensor[i] = image
        tensor_image = torch.from_numpy(image_tensor)
        

        save_name = path.split('/')[-1]

        return {'Tensor_image': tensor_image, 'fcs_file': save_name}

    return None

