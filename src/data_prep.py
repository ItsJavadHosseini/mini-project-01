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
#df = load_data(DATA_PATH)

    
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
    
#df= preparation_data(df)


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

#df = manage_duplicate(df)

def prepare_feature_target(df, target_columns):
    """
    input -> dataframs & target;
    out -> seprate target
    """
    X = df.drop(columns=[target_columns])
    y = df[target_columns]
    
    print(f'X.shape:{X.shape}')
    print(f'y.shape:{y.shape}')
    
    return X, y



#X, y = prepare_feature_target(df, 'Class')
    
def split_data(X, y, test_size=0.2, random_state=42):
    """
    split and stratified data;
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f'X-train: {X_train.shape}')
    print(f'X-train: {X_test.shape}')
    
    print(f'y train dist{y_train.value_counts(normalize=True)}')
    print(f'y test dist{y_test.value_counts(normalize=True)}')
    
    return X_train, X_test, y_train, y_test

#print(split_data(X, y))


#Scailing
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler
    
    
def save_scaler(scaler, path= SCALER_PATH):
    """
    save fitted scaller
    """
    joblib.dump(scaler, path)  
    print(f'scaler saved to: {path}')
    
    
    
if __name__ == "__main__":
    df = load_data()
    
    preparation_data(df)
    
    df = manage_duplicate(df)
    
    X, y = prepare_feature_target(df, 'Class')
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    save_scaler(scaler) 
    
    print(f'{100} * 100\nData prep complated.')   