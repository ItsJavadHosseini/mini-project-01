import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (cross_validate, StratifiedKFold)

from data_prep import (
    load_data, explore_data,
    manage_duplicate, prepare_feature_target,
    split_data
)

RANDOM_STATE = 42
N_SPLIT = 5


models = {
    'Logestic Regression' : Pipeline([
        ('scaler' , StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors= 5)) 
        
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('mmodel', DecisionTreeClassifier(random_state=RANDOM_STATE))
        
    ])
}



cv = StratifiedKFold(
    n_splits=N_SPLIT, 
    shuffle=True,
    random_state=RANDOM_STATE
)