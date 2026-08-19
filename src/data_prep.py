import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'DATA' / 'creditcard.csv'



def load_data(file_path=DATA_PATH):
    
    """
    load dataset
    out -> df
    """

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'dataset not found {file_path}')
    df = pd.read_csv(file_path)
    print(f'\n\ndf loaded...\n df shape: {df.shape}\n')
    return df


# data preparation
def explore_data(df):
    """
    input -> df
    out -> information of data
    
    analyze scr;
    descriptive gen;
    check mssing;
    class dist;
    check duplicate
    """
    print(f'=' * 25)
    print('\nInformation')
    print(f'sampels: {len(df):,}')
    print(f'features: {len(df.columns)-1}')
    print(f'\nmissing values:{df.isnull().sum().sum()}\n')
    print(df.isnull().sum())
    print(f'\nDuplicate: {df.duplicated().sum()}')
    print(f"\nClass dist count: {df['Class'].value_counts()}")
    print(f"\nClass ratio: {df['Class'].value_counts(normalize=True)}")
    print(f'\n\nDiscribe: {df.describe()}')
    print(f'\ninfo:')
    df.info()
    return df


# manage duplicate
def manage_duplicate(df):
    """
    del duplicate 
    
    """

    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    removed = before - after
    balance = df['Class'].value_counts(normalize=True)
    print(
        f'Befor: {before}, After: {after}, '
        f'Removed: {removed}, \nBalance: {balance}'
    )
    return df


def prepare_feature_target(df, target_columns):
    """
    input -> dataframs & target
    out -> separate target|
    gen X, y
    """
    X = df.drop(columns=[target_columns])
    y = df[target_columns]
    print(f'X.shape:{X.shape}')
    print(f'y.shape:{y.shape}')
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    split and stratified data
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print('----shape of X, y after split----')
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print('\n----Distribution--------')
    print('y train dist',y_train.value_counts(normalize=True))
    print('y test dist',y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test




if __name__ == "__main__":

    df = load_data()

    explore_data(df)

    df = manage_duplicate(df)

    X, y = prepare_feature_target(df, 'Class')

    X_train, X_test, y_train, y_test = split_data(X, y)
