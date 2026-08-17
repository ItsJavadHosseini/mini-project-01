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


# cross validation
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


def display_cv_results(results):
    """
    display the required cross validation metrics.
    """
    
    columns = [
        'precision_mean',
        'recall_mean',
        'f1_mean'
        ]
    
    print('\n5-Folld Stratified Cross Validation Results:')
    print(
        results[columns].round(4).to_string()
    )


def select_best_model(result, metric='f1_mean'):
    """
    select the beast model 
    """
    beast_model_name = result[metric].idxmax()
    beast_model_score = result.loc[beast_model_name, metric]
    print(f'beast model= {beast_model_name}\nmeric= {metric}\n\
        score= {beast_model_score}')

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

def scaling_experiment(model_without_scaling,\
    model_with_scaling,X_train,y_train,X_test,y_test):
    
    """
    
    compare model & experiment
    """
    results = {}
    models = {
        'Without Scaling' : model_without_scaling,
        'With Scaling' : model_with_scaling
    }

    for name, model in models.items():
        
        # train
        model.fit(
            X_train,
            y_train
        )
        
        #predict
        y_pred = model.predict(X_test)
        
        # Metrics
        results[name] = {
            'precision' : precision_score(y_test, y_pred),
            'recall' : recall_score(y_test, y_pred),
            'f1' : f1_score(y_test, y_pred),
        }
        
    return pd.DataFrame(results).T



# model experiment
knn_without_scaling = KNeighborsClassifier(n_neighbors=5)


knn_with_scaling = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(
        n_neighbors=5
    ))
])

def knn_hyperparameter_expriment(X_train, y_train,\
    X_test, y_test, k_values=(1, 5, 20)):
    """
    compare KNN performance for different values of K.   
    """
    results = {}
    for k in k_values:
        model = Pipeline([
            ('scaler',StandardScaler()),
            ('model',KNeighborsClassifier(n_neighbors=k))
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[f'k= {k}'] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }

    return pd.DataFrame(results).T


def desicion_tree_hyperparameter_experiment(X_train, y_train,\
    X_test, y_test, depth_values=(2, 5, 10, None)):
    """
    compare decision tree performance for
    different values of max_depth.
    """
    results = {
    
    }
    
    for depth in depth_values:
        model = DecisionTreeClassifier(max_depth=depth,random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[f'max depth = {depth}'] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
    return pd.DataFrame(results).T

def threshold_experiment(model, X_train, y_train,\
    X_test, y_test, thresholds=(0.3, 0.5, 0.7)):
    
    """
    return model performance using different classification threshold.
    """
    
    #train model
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    results = {}
    for threshold in thresholds:
        #convert probabilitis to class labels
        y_pred = (
            y_proba >= threshold
        ).astype(int)
        
        results[threshold] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
    return pd.DataFrame(results).T
    
    
    
    pass









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
        
    
    display_cv_results(results)
    
    
    print("\n" + "=" * 20)
    print("EXPERIMENT1 - KNN without/with scale")
    print("=" * 20)
    

    experiment_result = scaling_experiment(
        knn_without_scaling,
        knn_with_scaling,
        X_train,
        y_train,
        X_test,
        y_test
    )
    
    print(
        experiment_result.round(4).to_string()
    )
    
    

    print("\n" + "=" * 20)
    print("EXPERIMENT2 - KNN by hyperparammeters")
    print("=" * 20)

    knn_results = knn_hyperparameter_expriment(
        X_train, y_train, X_test, y_test
    )
    
    print(knn_results.round(4).to_string())


    print("\n" + "=" * 20)
    print("EXPERIMENT2 - Decisiontree by hyperparammeters")
    print("=" * 20)
    
    tree_results = desicion_tree_hyperparameter_experiment(
        X_train, y_train, X_test, y_test
    )

    print(tree_results.round(4).to_string())



    print("\n" + "=" * 20)
    print("EXPERIMENT 3 Classification Threshold")
    print("=" * 20)

    threshold_results = threshold_experiment(
        models['Logistic Regression'],
        X_train, 
        y_train,
        X_test,
        y_test
    )

    print('\nLogisticRegression:\n',threshold_results.round(4).to_string())

    threshold_results = threshold_experiment(
        models['Decision Tree'],
        X_train, 
        y_train,
        X_test,
        y_test
    )

    print('\nDecison Tree:\n',threshold_results.round(4).to_string())

    threshold_results = threshold_experiment(
        models['KNN'],
        X_train, 
        y_train,
        X_test,
        y_test
    )

    print('\nKNN:\n',threshold_results.round(4).to_string())




    print("\n" + "=" * 20)
    print("FINISH!")
    print("=" * 20)

    
    