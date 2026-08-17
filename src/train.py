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
        ('mmodel', DecisionTreeClassifier(random_state=RANDOM_STATE))
        
    ])
}



cv = StratifiedKFold(
    n_splits=N_SPLIT, 
    shuffle=True,
    random_state=RANDOM_STATE
)


def compare_models(X_train, y_train):
    """
    Compare all models using stratified cross-validation.
    """

    results = {}

    for name, pipeline in models.items():

        scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=[
                'accuracy',
                'precision',
                'recall',
                'f1'
            ],
            n_jobs=-1
        )

        results[name] = {

            'accuracy_mean':
                scores['test_accuracy'].mean(),

            'accuracy_std':
                scores['test_accuracy'].std(),

            'precision_mean':
                scores['test_precision'].mean(),

            'precision_std':
                scores['test_precision'].std(),

            'recall_mean':
                scores['test_recall'].mean(),

            'recall_std':
                scores['test_recall'].std(),

            'f1_mean':
                scores['test_f1'].mean(),

            'f1_std':
                scores['test_f1'].std()
        }

    return pd.DataFrame(results).T

if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Explore dataset
    explore_data(df)

    # Remove duplicate rows
    df = manage_duplicate(df)

    # Separate features and target
    X, y = prepare_feature_target(
        df,
        'Class'
    )

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Compare models using cross-validation
    results = compare_models(
        X_train,
        y_train
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results[
            [
                'accuracy_mean',
                'precision_mean',
                'recall_mean',
                'f1_mean'
            ]
        ].round(4).to_string()
    )

    print("\nCross-validation completed.")