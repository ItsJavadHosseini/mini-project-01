import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, f1_score,\
    recall_score, accuracy_score, precision_score

from sklearn.model_selection import (cross_validate, StratifiedKFold)

from data_prep import (
    load_data, explore_data,
    manage_duplicate, prepare_feature_target,
    split_data
)

RANDOM_STATE = 42
N_SPLIT = 5

models = {
    'Logistic Regression' : Pipeline([
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
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    for name, model in models.items():

        print(f"\nTraining: {name}")

        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
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
                scores['test_f1'].std(),
            # ROC-AUC
            "roc_auc_mean":
                scores["test_roc_auc"].mean(),

            "roc_auc_std":
                scores["test_roc_auc"].std(),

            # PR-AUC
            "pr_auc_mean":
                scores["test_pr_auc"].mean(),

            "pr_auc_std":
                scores["test_pr_auc"].std()
        }
        
    return pd.DataFrame(results).T


def select_best_model(result, metric='f1_mean'):
    """
    select the beast model 
    """
    beast_model_name = result[metric].idxmax()
    beast_model_score = result.loc[beast_model_name, metric]
    
    print(f'beast model= {beast_model_name}\nmeric= {metric}\nscore= {beast_model_score}')
    
    return beast_model_name

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    train one model on the entire training set
    and evalute it on the test set.
    """
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
   
    
    

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

    display_columns = [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
        "pr_auc_mean"
    ]

    print(
        results[display_columns]
        .round(4)
        .to_string()
    )
    
    
    
    # select beast model
    best_model_name = select_best_model(
        results,
        metric="f1_mean"
    )
    
    beast_model = models[best_model_name]
    test_results = evaluate_model(beast_model,X_train,y_train,X_test,y_test)
    
    
    print("\n" + "=" * 20)
    print('FAINAL TEST EVALUATION FOR BEAST MODEL')
    print("=" * 20)
    
    print(f'model: {best_model_name}')
    print(f'Accuracy: {test_results['accuracy']:.4f}')
    print(f'Precision: {test_results['precision']:.4f}')
    print(f'Recall: {test_results['recall']:.4f}')
    print(f'f1: {test_results['f1']:.4f}')
    print(f'cm: \n{test_results['confusion_matrix']}')
    

    print("\n" + "=" * 20)
    print('COMPARE EVALUATION')
    print("=" * 20)

    for name, model in models.items():
        print(f'\n\nEvaluating: {name}')
        result = evaluate_model(model, X_train, y_train,\
            X_test, y_test)
        
        
        print(f'model: {name}')
        print(f'Accuracy: {result['accuracy']:.4f}')
        print(f'Precision: {result['precision']:.4f}')
        print(f'Recall: {result['recall']:.4f}')
        print(f'f1: {result['f1']:.4f}')
        print(f'cm: \n{result['confusion_matrix']}')
        
    
    

    print("\n" + "=" * 20)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 20)

    print("\nCross-validation completed.")
    