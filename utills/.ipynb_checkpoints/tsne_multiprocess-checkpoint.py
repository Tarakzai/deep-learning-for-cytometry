import pandas as pd
import numpy as np 
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from fcsy import DataFrame
import os




def process_file_tsne(path, column, transformation):
    
    
    if not path.endswith('.csv') and not path.endswith('.DS_Store'):
        df_dummy = DataFrame.from_fcs(path)
        new_cols=[]
        for columns in df_dummy.columns:
            start_index=columns.find('(')
            if start_index!=-1 :
                new_name=columns[start_index:]
            else:
                new_name=columns
            new_cols.append(new_name)
    
        df_dummy.columns=new_cols
        if column is not None:
            df_dummy.drop(column,axis=1,inplace=True)

        Scaled = StandardScaler().fit_transform(df_dummy)
        new_transform = transformation.transform(Scaled)
        principalDf = pd.DataFrame(data=new_transform, columns=['t1', 't2'])

        x_min = principalDf['t1'].min()
        x_max = principalDf['t1'].max()
        y_min = principalDf['t2'].min()
        y_max = principalDf['t2'].max()

        return {'X_min': x_min, 'X_max': x_max, 'Y_min': y_min, 'Y_max': y_max}

    return None

