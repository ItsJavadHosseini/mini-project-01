import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path


#path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT/ 'DATA' / 'creditcard.csv'
MODELS_PATH = PROJECT_ROOT/'models'
SCALER_PATH = MODELS_PATH / 'scaler.pkl'

MODELS_PATH.mkdir(exist_ok=True)


def load_data(file_path=DATA_PATH):

    """
    load dataset
    out -> df
    """

    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f'dataset not found {file_path}')
    
    df = pd.read_csv(file_path)
    print(f'df loaded: {df.shape}')
    
    return df

#print(load_data(DATA_PATH))  
df = load_data(DATA_PATH)

    
#data preparation
def preparation_data(df):
    """
    input -> df
    out -> information of data
    """
    print(f'sampels: {len(df):,}')
    print(f'features: {len(df.columns)}')
    
    print('\nmissing values:')
    print(df.isnull().sum())    
    print(df.isna().sum())
    
    print(f'\nDuplicate: {df.duplicated().sum()}')
    
    print(f"Class dist: {df['Class'].value_counts()}")
    
    print(f"\nClass ratio: {df['Class'].value_counts(normalize=True)}")
    
    print(f'Discribe: {df.describe()}')
    
    return df 

#print(preparation_data(df))   
    

#manage duplicate
def manage_duplicate(df):
    """
    del duplicate
    """
    
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    removed = before - after
    balance = df['Class'].value_counts(normalize=True)
    
    print(f'Befor: {before}, After: {after}, Removed: {removed}, \nBalance: {balance}')
    return df

#print(manage_duplicate(df))

