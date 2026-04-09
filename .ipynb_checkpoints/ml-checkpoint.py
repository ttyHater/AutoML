"""
Production-Ready ML Pipeline Functions
======================================

Refactored regression and classification functions with:
- Consistent return types
- Input validation
- Comprehensive error handling
- Configurable hyperparameters
- Clear documentation
- Type hints
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List, Optional, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    cross_val_score,
    GridSearchCV,
    cross_validate
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================================
# VALIDATION & UTILITY FUNCTIONS
# ============================================================================

def _validate_inputs(X, y, numeric_features, categorical_features):
    """Validate input data and feature lists."""
    if X is None or len(X) == 0:
        raise ValueError("X cannot be empty")
    
    if y is None or len(y) == 0:
        raise ValueError("y cannot be empty")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
    
    # Check if features exist in X
    all_features = list(numeric_features) + (categorical_features or [])
    missing_features = set(all_features) - set(X.columns)
    
    if missing_features:
        raise ValueError(f"Features not found in X: {missing_features}")
    
    if len(numeric_features) == 0 and (not categorical_features or len(categorical_features) == 0):
        raise ValueError("Must specify at least one feature (numeric or categorical)")


def _validate_split_params(test_size, val_size, split_strategy):
    """Validate train/val/test split parameters."""
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if split_strategy == "train_val_test":
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size + val_size >= 1:
            raise ValueError(f"test_size + val_size must be < 1, got {test_size + val_size}")


def _build_preprocessor(numeric_features, categorical_features):
    """Build scikit-learn preprocessing pipeline."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if numeric_features and len(numeric_features) > 0:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features and len(categorical_features) > 0:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    return ColumnTransformer(transformers)


def _plot_cv_results(cv_results: Dict[str, np.ndarray], title: str = "Cross-Validation Results"):
    """Plot cross-validation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar plot
    axes[0].bar(range(len(cv_results['scores'])), cv_results['scores'], 
                alpha=0.7, color='steelblue')
    axes[0].axhline(y=cv_results['mean'], color='r', linestyle='--', 
                    label=f"Mean: {cv_results['mean']:.4f}")
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Score')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1].boxplot(cv_results['scores'], vert=True)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Score Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# REGRESSION FUNCTION
# ============================================================================

def regress(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    model_name: str = "linear",
    split_strategy: str = "train_test",
    test_size: float = 0.25,
    val_size: float = 0.25,
    n_splits: int = 5,
    n_repeats: int = 30,
    neighbor_range: range = range(1, 15),
    hyperparams: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    show_plot: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Regression pipeline with multiple model types and CV strategies.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable (continuous)
    numeric_features : List[str]
        Names of numeric feature columns
    categorical_features : List[str], optional
        Names of categorical feature columns. Default: None
    model_name : str, default='linear'
        Model type: 'linear', 'ridge', 'lasso', 'knn'
    split_strategy : str, default='train_test'
        Strategy for data splitting:
        - 'train_test': Simple 80/20 split
        - 'train_val_test': 60/20/20 split with validation set
        - 'k_fold': K-fold cross-validation
        - 'monte_carlo': Monte Carlo cross-validation (KNN only)
    test_size : float, default=0.25
        Proportion of data for test set
    val_size : float, default=0.25
        Proportion of data for validation set (train_val_test only)
    n_splits : int, default=5
        Number of folds for k_fold strategy
    n_repeats : int, default=30
        Number of iterations for monte_carlo strategy
    neighbor_range : range, default=range(1, 15)
        Range of k values to test for KNN
    hyperparams : Dict[str, Any], optional
        Model-specific hyperparameters. E.g., {'alpha': 0.5} for Ridge/Lasso
    random_state : int, default=42
        Random seed for reproducibility
    show_plot : bool, default=False
        Whether to display plots
    verbose : bool, default=True
        Whether to print results
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': fitted model (Pipeline)
        - 'metrics': dict of performance metrics
        - 'cv_results': dict of CV scores (if applicable)
        - 'training_scores': MC training scores (if monte_carlo)
        - 'test_scores': MC test scores (if monte_carlo)
        - 'X_train', 'X_test', 'y_train', 'y_test': split data (if train_test)
        - 'split_strategy': strategy used
    
    Examples
    --------
    >>> # Simple train/test split
    >>> results = regress(
    ...     X, y, 
    ...     numeric_features=['age', 'income'],
    ...     model_name='linear',
    ...     split_strategy='train_test'
    ... )
    >>> model = results['model']
    >>> print(results['metrics'])
    
    >>> # K-fold cross-validation
    >>> results = regress(
    ...     X, y,
    ...     numeric_features=['age', 'income'],
    ...     model_name='ridge',
    ...     split_strategy='k_fold',
    ...     n_splits=5,
    ...     hyperparams={'alpha': 1.0}
    ... )
    >>> print(f"CV R²: {results['cv_results']['mean']:.4f}")
    
    >>> # Monte Carlo for KNN
    >>> results = regress(
    ...     X, y,
    ...     numeric_features=['age', 'income'],
    ...     model_name='knn',
    ...     split_strategy='monte_carlo',
    ...     n_repeats=30,
    ...     show_plot=True
    ... )
    >>> training_scores, test_scores = results['training_scores'], results['test_scores']
    """
    
    # ---- Validation ----
    if categorical_features is None:
        categorical_features = []
    
    _validate_inputs(X, y, numeric_features, categorical_features)
    _validate_split_params(test_size, val_size, split_strategy)
    
    valid_models = ['linear', 'ridge', 'lasso', 'knn']
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}, got '{model_name}'")
    
    valid_strategies = ['train_test', 'train_val_test', 'k_fold', 'monte_carlo']
    if split_strategy not in valid_strategies:
        raise ValueError(f"split_strategy must be one of {valid_strategies}, got '{split_strategy}'")
    
    if split_strategy == 'monte_carlo' and model_name != 'knn':
        raise ValueError("monte_carlo strategy only supported for model_name='knn'")
    
    y = np.ravel(y)
    hyperparams = hyperparams or {}
    
    # ---- Build preprocessor ----
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    
    # ---- Initialize model ----
    if model_name == "linear":
        model = LinearRegression(**hyperparams)
    elif model_name == "ridge":
        model = Ridge(random_state=random_state, **hyperparams)
    elif model_name == "lasso":
        model = Lasso(random_state=random_state, **hyperparams)
    elif model_name == "knn":
        model = KNeighborsRegressor(**hyperparams)
    
    # ---- Monte Carlo CV (KNN only) ----
    if split_strategy == "monte_carlo":
        if verbose:
            print(f"\nMonte Carlo Regression (KNN) with {n_repeats} iterations")
            print(f"Testing k values: {list(neighbor_range)}\n")
        
        training_acc = pd.DataFrame()
        test_acc = pd.DataFrame()
        
        for seed in range(random_state, random_state + n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
            
            train_scores = []
            test_scores = []
            
            for k in neighbor_range:
                reg = KNeighborsRegressor(n_neighbors=k, **{k: v for k, v in hyperparams.items() if k != 'n_neighbors'})
                
                # Fit with preprocessing
                X_train_transformed = preprocessor.fit_transform(X_train)
                X_test_transformed = preprocessor.transform(X_test)
                
                reg.fit(X_train_transformed, y_train)
                train_scores.append(reg.score(X_train_transformed, y_train))
                test_scores.append(reg.score(X_test_transformed, y_test))
            
            training_acc[seed] = train_scores
            test_acc[seed] = test_scores
        
        # Print results
        if verbose:
            print("Monte Carlo Results (Mean ± Std):")
            for i, k in enumerate(neighbor_range):
                print(f"  k={k:2d} | Train R²: {training_acc.iloc[i].mean():.4f} ± {training_acc.iloc[i].std():.4f} | "
                      f"Test R²: {test_acc.iloc[i].mean():.4f} ± {test_acc.iloc[i].std():.4f}")
        
        # Plot
        if show_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            mean_train = training_acc.mean(axis=1)
            std_train = training_acc.std(axis=1)
            mean_test = test_acc.mean(axis=1)
            std_test = test_acc.std(axis=1)
            
            ax.plot(neighbor_range, mean_train, label="Training R²", color='blue', 
                   marker='o', linestyle='dashed', markersize=8)
            ax.fill_between(neighbor_range, mean_train - std_train/2, mean_train + std_train/2, 
                           alpha=0.2, color='blue')
            ax.plot(neighbor_range, mean_test, label="Test R²", color='red', 
                   marker='^', linestyle='-', markersize=8)
            ax.fill_between(neighbor_range, mean_test - std_test/2, mean_test + std_test/2, 
                           alpha=0.2, color='red')
            
            ax.set_xlabel("n_neighbors", fontsize=12)
            ax.set_ylabel("R²", fontsize=12)
            ax.set_title("Monte Carlo KNN Regressor: R² vs n_neighbors", fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return {
            'model': None,
            'metrics': None,
            'split_strategy': 'monte_carlo',
            'training_scores': training_acc,
            'test_scores': test_acc,
            'cv_results': None,
            'preprocessor': preprocessor
        }
    
    # ---- Build pipeline ----
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # ---- Train/Test Split ----
    if split_strategy == "train_test":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Regression - Train/Test Split")
            print(f"{'='*50}")
            for metric, value in metrics.items():
                print(f"Test {metric.upper()}: {value:.4f}")
        
        return {
            'model': pipeline,
            'metrics': metrics,
            'split_strategy': 'train_test',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'cv_results': None
        }
    
    # ---- Train/Val/Test Split ----
    elif split_strategy == "train_val_test":
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state
        )
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        metrics = {
            'val_r2': r2_score(y_val, y_val_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred)
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Regression - Train/Val/Test Split")
            print(f"{'='*50}")
            print(f"Validation R²: {metrics['val_r2']:.4f}")
            print(f"Test R²: {metrics['test_r2']:.4f}")
        
        return {
            'model': pipeline,
            'metrics': metrics,
            'split_strategy': 'train_val_test',
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred,
            'cv_results': None
        }
    
    # ---- K-Fold CV ----
    elif split_strategy == "k_fold":
        pipeline.fit(X, y)
        
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='r2')
        
        cv_results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'n_splits': n_splits
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Regression - {n_splits}-Fold Cross-Validation")
            print(f"{'='*50}")
            print(f"Fold R² Scores: {np.round(scores, 4)}")
            print(f"Mean R²: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        if show_plot:
            _plot_cv_results(cv_results, f"{model_name.upper()} {n_splits}-Fold CV Results")
        
        return {
            'model': pipeline,
            'metrics': None,
            'split_strategy': 'k_fold',
            'cv_results': cv_results
        }


# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def classify(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    model_name: str = "logistic",
    split_strategy: str = "train_test",
    test_size: float = 0.25,
    val_size: float = 0.25,
    n_splits: int = 5,
    n_repeats: int = 30,
    neighbor_range: range = range(1, 31),
    hyperparams: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    random_state: int = 42,
    show_plot: bool = False,
    verbose: bool = True,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Classification pipeline with multiple model types and CV strategies.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable (categorical)
    numeric_features : List[str]
        Names of numeric feature columns
    categorical_features : List[str], optional
        Names of categorical feature columns. Default: None
    model_name : str, default='logistic'
        Model type: 'logistic', 'svm', 'knn'
    split_strategy : str, default='train_test'
        Strategy for data splitting:
        - 'train_test': Simple 80/20 split with GridSearchCV
        - 'train_val_test': 60/20/20 split (no hyperparameter tuning)
        - 'k_fold': K-fold cross-validation with hyperparameter tuning
        - 'monte_carlo': Monte Carlo cross-validation (KNN only)
    test_size : float, default=0.25
        Proportion of data for test set
    val_size : float, default=0.25
        Proportion of data for validation set (train_val_test only)
    n_splits : int, default=5
        Number of folds for k_fold strategy
    n_repeats : int, default=30
        Number of iterations for monte_carlo strategy
    neighbor_range : range, default=range(1, 31)
        Range of k values to test for KNN monte_carlo
    hyperparams : Dict[str, Any], optional
        Model-specific hyperparameters (used if param_grid not provided)
    param_grid : Dict[str, List[Any]], optional
        Grid of hyperparameters for GridSearchCV. E.g., {'model__C': [0.1, 1, 10]}
    random_state : int, default=42
        Random seed for reproducibility
    show_plot : bool, default=False
        Whether to display plots
    verbose : bool, default=True
        Whether to print results
    n_jobs : int, default=-1
        Number of jobs for parallel processing (-1 = all cores)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': fitted model (Pipeline)
        - 'metrics': dict of performance metrics
        - 'cv_results': dict of CV scores (if applicable)
        - 'grid_search': GridSearchCV object (if tuning performed)
        - 'training_scores': MC training scores (if monte_carlo)
        - 'test_scores': MC test scores (if monte_carlo)
        - 'X_train', 'X_test', 'y_train', 'y_test': split data (if train_test)
        - 'split_strategy': strategy used
    
    Examples
    --------
    >>> # Simple train/test with hyperparameter tuning
    >>> results = classify(
    ...     X, y,
    ...     numeric_features=['age', 'income'],
    ...     model_name='logistic',
    ...     split_strategy='train_test',
    ...     param_grid={'model__C': [0.1, 1, 10]}
    ... )
    >>> print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    
    >>> # K-fold with custom hyperparameters
    >>> results = classify(
    ...     X, y,
    ...     numeric_features=['age', 'income'],
    ...     model_name='knn',
    ...     split_strategy='k_fold',
    ...     n_splits=5,
    ...     param_grid={
    ...         'model__n_neighbors': [3, 5, 7],
    ...         'model__weights': ['uniform', 'distance']
    ...     }
    ... )
    >>> print(f"CV Score: {results['cv_results']['mean']:.4f}")
    
    >>> # Monte Carlo for KNN
    >>> results = classify(
    ...     X, y,
    ...     numeric_features=['age', 'income'],
    ...     model_name='knn',
    ...     split_strategy='monte_carlo',
    ...     n_repeats=30,
    ...     show_plot=True
    ... )
    """
    
    # ---- Validation ----
    if categorical_features is None:
        categorical_features = []
    
    _validate_inputs(X, y, numeric_features, categorical_features)
    _validate_split_params(test_size, val_size, split_strategy)
    
    valid_models = ['logistic', 'svm', 'knn']
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}, got '{model_name}'")
    
    valid_strategies = ['train_test', 'train_val_test', 'k_fold', 'monte_carlo']
    if split_strategy not in valid_strategies:
        raise ValueError(f"split_strategy must be one of {valid_strategies}, got '{split_strategy}'")
    
    if split_strategy == 'monte_carlo' and model_name != 'knn':
        raise ValueError("monte_carlo strategy only supported for model_name='knn'")
    
    y = np.ravel(y)
    hyperparams = hyperparams or {}
    
    # ---- Build preprocessor ----
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    
    # ---- Initialize model ----
    if model_name == "logistic":
        model = LogisticRegression(max_iter=5000, random_state=random_state, **hyperparams)
        if param_grid is None:
            param_grid = {'model__C': np.logspace(-3, 3, 10)}
    
    elif model_name == "svm":
        model = LinearSVC(max_iter=5000, dual=False, random_state=random_state, **hyperparams)
        if param_grid is None:
            param_grid = {'model__C': np.logspace(-3, 3, 10)}
    
    elif model_name == "knn":
        model = KNeighborsClassifier(**hyperparams)
        if param_grid is None:
            param_grid = {
                'model__n_neighbors': range(1, 16),
                'model__weights': ['uniform', 'distance']
            }
    
    # ---- Monte Carlo CV (KNN only) ----
    if split_strategy == "monte_carlo":
        if verbose:
            print(f"\nMonte Carlo Classification (KNN) with {n_repeats} iterations")
            print(f"Testing k values: {list(neighbor_range)}\n")
        
        training_acc = pd.DataFrame()
        test_acc = pd.DataFrame()
        
        for seed in range(random_state, random_state + n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            
            train_scores = []
            test_scores = []
            
            for k in neighbor_range:
                knn_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('knn', KNeighborsClassifier(n_neighbors=k, **{kk: v for kk, v in hyperparams.items() if kk != 'n_neighbors'}))
                ])
                knn_pipeline.fit(X_train, y_train)
                train_scores.append(knn_pipeline.score(X_train, y_train))
                test_scores.append(knn_pipeline.score(X_test, y_test))
            
            training_acc[seed] = train_scores
            test_acc[seed] = test_scores
        
        # Print results
        if verbose:
            print("Monte Carlo Results (Mean ± Std):")
            for i, k in enumerate(neighbor_range):
                print(f"  k={k:2d} | Train Acc: {training_acc.iloc[i].mean():.4f} ± {training_acc.iloc[i].std():.4f} | "
                      f"Test Acc: {test_acc.iloc[i].mean():.4f} ± {test_acc.iloc[i].std():.4f}")
        
        # Plot
        if show_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            mean_train = training_acc.mean(axis=1)
            std_train = training_acc.std(axis=1)
            mean_test = test_acc.mean(axis=1)
            std_test = test_acc.std(axis=1)
            
            ax.errorbar(neighbor_range, mean_train, yerr=std_train/2, 
                       label="Training Accuracy", marker='o', capsize=5)
            ax.errorbar(neighbor_range, mean_test, yerr=std_test/2, 
                       label="Test Accuracy", marker='s', capsize=5)
            
            ax.set_xlabel("n_neighbors", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Monte Carlo KNN Classifier: Accuracy vs n_neighbors", fontsize=13)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        return {
            'model': None,
            'metrics': None,
            'split_strategy': 'monte_carlo',
            'training_scores': training_acc,
            'test_scores': test_acc,
            'cv_results': None,
            'grid_search': None,
            'preprocessor': preprocessor
        }
    
    # ---- Build pipeline ----
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # ---- Train/Test Split with GridSearchCV ----
    if split_strategy == "train_test":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_splitter, scoring='accuracy', 
            n_jobs=n_jobs, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cv_best_score': grid_search.best_score_
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Classification - Train/Test Split with GridSearchCV")
            print(f"{'='*60}")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"CV Best Score: {grid_search.best_score_:.4f}")
            print(f"\nTest Metrics:")
            for metric, value in metrics.items():
                if metric != 'cv_best_score':
                    print(f"  {metric.upper()}: {value:.4f}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        return {
            'model': best_model,
            'metrics': metrics,
            'split_strategy': 'train_test',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'cv_results': None,
            'grid_search': grid_search
        }
    
    # ---- Train/Val/Test Split (no hyperparameter tuning) ----
    elif split_strategy == "train_val_test":
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
        )
        
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        metrics = {
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred)
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Classification - Train/Val/Test Split")
            print(f"{'='*60}")
            print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return {
            'model': pipeline,
            'metrics': metrics,
            'split_strategy': 'train_val_test',
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'y_val_pred': y_val_pred,
            'y_test_pred': y_test_pred,
            'cv_results': None,
            'grid_search': None
        }
    
    # ---- K-Fold CV with GridSearchCV ----
    elif split_strategy == "k_fold":
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_splitter, scoring='accuracy', 
            n_jobs=n_jobs, verbose=0
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        
        cv_results = {
            'scores': grid_search.cv_results_['mean_test_score'],
            'mean': grid_search.best_score_,
            'std': grid_search.cv_results_['std_test_score'].mean(),
            'n_splits': n_splits,
            'best_params': grid_search.best_params_
        }
        
        if verbose:
            print(f"\n{model_name.upper()} Classification - {n_splits}-Fold Cross-Validation")
            print(f"{'='*60}")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"CV Best Score: {grid_search.best_score_:.4f}")
        
        if show_plot:
            _plot_cv_results(cv_results, f"{model_name.upper()} {n_splits}-Fold CV Results")
        
        return {
            'model': best_model,
            'metrics': None,
            'split_strategy': 'k_fold',
            'cv_results': cv_results,
            'grid_search': grid_search
        }


# ============================================================================
# UTILITY: Automatic Model Selection
# ============================================================================

def auto_regress(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    test_size: float = 0.25,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Automatically compare all regression models using k-fold CV.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    numeric_features : List[str]
        Names of numeric feature columns
    categorical_features : List[str], optional
        Names of categorical feature columns
    test_size : float, default=0.25
        For internal test splits
    n_splits : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=True
        Whether to print results
    
    Returns
    -------
    pd.DataFrame
        Comparison table of all models
    """
    models = ['linear', 'ridge', 'lasso', 'knn']
    results = []
    
    if verbose:
        print("\nAutomatically comparing regression models...")
        print("="*70)
    
    for model_name in models:
        result = regress(
            X, y,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_name=model_name,
            split_strategy='k_fold',
            n_splits=n_splits,
            random_state=random_state,
            verbose=False
        )
        
        results.append({
            'Model': model_name.upper(),
            'Mean R²': result['cv_results']['mean'],
            'Std R²': result['cv_results']['std']
        })
    
    comparison_df = pd.DataFrame(results).sort_values('Mean R²', ascending=False)
    
    if verbose:
        print(comparison_df.to_string(index=False))
        print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
        print("="*70)
    
    return comparison_df


def auto_classify(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    test_size: float = 0.25,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Automatically compare all classification models using k-fold CV.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    numeric_features : List[str]
        Names of numeric feature columns
    categorical_features : List[str], optional
        Names of categorical feature columns
    test_size : float, default=0.25
        For internal test splits
    n_splits : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=True
        Whether to print results
    
    Returns
    -------
    pd.DataFrame
        Comparison table of all models
    """
    models = ['logistic', 'svm', 'knn']
    results = []
    
    if verbose:
        print("\nAutomatically comparing classification models...")
        print("="*70)
    
    for model_name in models:
        result = classify(
            X, y,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_name=model_name,
            split_strategy='k_fold',
            n_splits=n_splits,
            random_state=random_state,
            verbose=False
        )
        
        results.append({
            'Model': model_name.upper(),
            'Mean Accuracy': result['cv_results']['mean'],
            'Std Accuracy': result['cv_results']['std']
        })
    
    comparison_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)
    
    if verbose:
        print(comparison_df.to_string(index=False))
        print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
        print("="*70)
    
    return comparison_df
