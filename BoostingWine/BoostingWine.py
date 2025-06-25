import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import preprocessing # For Min-Max Scaling
import os # For checking file existence and creating dummy data

# Define the file path for the dataset
CSV_FILE_PATH = 'wine.csv'

def load_wine_data(file_path):
    """
    Loads wine quality data from a CSV file.
    If the file is not found, a dummy CSV is created for demonstration.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'wine.csv' is in the same directory as the script.")
        print("Creating a dummy 'wine.csv' for demonstration purposes.")
        # Create a dummy CSV for demonstration
        dummy_data = {
            "fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, 7.5],
            "volatile acidity": [0.70, 0.88, 0.76, 0.28, 0.70, 0.66, 0.60, 0.65, 0.58, 0.50],
            "citric acid": [0.00, 0.00, 0.04, 0.56, 0.00, 0.00, 0.06, 0.00, 0.02, 0.12],
            "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 3.0],
            "chlorides": [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.065, 0.073, 0.080],
            "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 11.0, 13.0, 15.0, 15.0, 9.0, 15.0],
            "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 34.0, 40.0, 59.0, 21.0, 18.0, 30.0],
            "density": [0.9978, 0.9968, 0.9970, 0.9980, 0.9978, 0.9978, 0.9964, 0.9946, 0.9968, 0.9970],
            "pH": [3.51, 3.20, 3.26, 3.16, 3.51, 3.51, 3.30, 3.39, 3.26, 3.33],
            "sulphates": [0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.47, 0.65, 0.56],
            "alcohol": [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 10.8, 11.0, 10.4, 9.8],
            "quality": [5, 5, 5, 6, 5, 5, 5, 7, 7, 5]
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False, sep=';')
        print("Dummy 'wine.csv' created. Please replace it with your actual data.")
    try:
        # Use sep=';' as specified in the original snippet
        return pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

def transform_quality_to_binary_target(dataframe, quality_column='quality', new_target_column='tasty', threshold=7):
    """
    Transforms a continuous 'quality' regression problem into a binary classification problem.
    A wine is considered 'tasty' (1) if its quality is above the threshold, else (0).

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        quality_column (str): The name of the column containing quality scores.
        new_target_column (str): The name for the new binary target column.
        threshold (int): The quality score threshold to classify as 'tasty'.

    Returns:
        pandas.DataFrame: The DataFrame with the new binary target column.
    """
    print(f"\n--- Transforming 'quality' to binary '{new_target_column}' ---")
    print(f"  Threshold for 'tasty' (quality > {threshold}):")
    dataframe[new_target_column] = dataframe[quality_column].apply(lambda q: 1 if q > threshold else 0)
    print(f"Class distribution for '{new_target_column}':")
    print(dataframe[new_target_column].value_counts())
    return dataframe

def prepare_features_target_arrays(dataframe, feature_cols, target_col):
    """
    Extracts features and target into NumPy arrays.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column.

    Returns:
        tuple: (features (X) as numpy array, target (y) as numpy array).
    """
    features = dataframe[feature_cols].values
    target = dataframe[target_col].values
    print("\n--- Prepared Data Shapes ---")
    print(f"Features (X) shape: {features.shape}")
    print(f"Target (y) shape: {target.shape}")
    return features, target

def scale_features(X_data):
    """
    Scales the features using MinMaxScaler.
    This is often important for algorithms sensitive to feature scales,
    though AdaBoost's base estimator (often Decision Tree) is scale-invariant,
    scaling is a good general practice.

    Args:
        X_data (numpy.ndarray): The feature data to be scaled.

    Returns:
        numpy.ndarray: The scaled feature data.
    """
    print("\n--- Scaling Features with MinMaxScaler ---")
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    print(f"Scaled features (first 3 rows):\n{X_scaled[:3]}")
    return X_scaled

def split_data(X_data, y_data, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X_data (numpy.ndarray): The feature data.
        y_data (numpy.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.
                            Ensures reproducibility.

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    print(f"\nSplitting data into training ({(1-test_size)*100:.0f}%) and testing ({test_size*100:.0f}%) sets...")
    # `stratify=y_data` is crucial here because the 'tasty' classes might be imbalanced.
    # It ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=random_state, stratify=y_data
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def tune_adaboost_hyperparameters(X_train, y_train, random_state=42, cv_folds=5):
    """
    Performs GridSearchCV to find the optimal hyperparameters for an AdaBoost Classifier.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        random_state (int): Seed for reproducibility of the AdaBoostClassifier.
        cv_folds (int): Number of folds for cross-validation within GridSearchCV.

    Returns:
        sklearn.model_selection.GridSearchCV: The fitted GridSearchCV object,
                                              containing the best estimator.
    """
    print("\n--- Tuning AdaBoost Hyperparameters using GridSearchCV ---")
    # Initialize AdaBoostClassifier with a base estimator.
    # By default, it uses a DecisionTreeClassifier with max_depth=1 (a Decision Stump).
    # Setting random_state for reproducibility.
    ada_classifier = AdaBoostClassifier(random_state=random_state)

    # Define the parameter grid to search over
    param_dist = {
        # n_estimators: The maximum number of estimators at which boosting is terminated.
        #               More estimators can lead to better performance but also overfitting
        #               and longer training times.
        'n_estimators': [10, 50, 100, 200, 500],
        # learning_rate: Weights applied to each classifier at each boosting iteration.
        #                A lower learning rate requires more estimators.
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    }

    # Initialize GridSearchCV
    # estimator: The model to tune (AdaBoostClassifier)
    # param_grid: Dictionary of parameter names and values to test
    # cv: Number of folds for cross-validation for parameter tuning.
    # verbose: Controls the verbosity of the output.
    # scoring: The metric used to evaluate each model during cross-validation.
    grid_search = GridSearchCV(
        estimator=ada_classifier,
        param_grid=param_dist,
        cv=cv_folds,
        verbose=1,
        scoring='accuracy',
        refit=True # After search, refit the best model on the entire training data
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")
    print(f"Best estimator (trained model): {grid_search.best_estimator_}")

    return grid_search

def evaluate_model(fitted_model, X_test, y_test, target_names=['Not Tasty', 'Tasty']):
    """
    Evaluates the trained AdaBoost model (best estimator from GridSearchCV)
    on the test data and prints performance metrics.

    Args:
        fitted_model (sklearn.ensemble.AdaBoostClassifier): The trained AdaBoost model.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing target (true labels).
        target_names (list): List of class names for better confusion matrix readability.
    """
    print("\n--- Evaluating Model Performance on Test Set ---")
    prediction = fitted_model.predict(X_test)

    # Confusion Matrix: A table used to describe the performance of a classification model.
    # Rows represent true classes, columns represent predicted classes.
    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_test, prediction)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Accuracy Score: The proportion of correctly classified instances.
    accuracy = accuracy_score(y_test, prediction)
    print(f'\nAccuracy Score: {accuracy:.4f}')

    # Classification Report: Provides precision, recall, f1-score for each class.
    print("\nClassification Report:")
    print(classification_report(y_test, prediction, target_names=target_names))

if __name__ == "__main__":
    # Define parameters for data processing and model tuning
    FEATURE_COLUMNS = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    QUALITY_COLUMN = 'quality'
    TASTY_TARGET_COLUMN = 'tasty'
    QUALITY_THRESHOLD = 7 # Quality > 7 is 'tasty' (1), else 'not tasty' (0)
    TEST_DATA_SPLIT_RATIO = 0.3
    RANDOM_SEED = 42 # For reproducibility of splits and AdaBoost
    GRID_SEARCH_CV_FOLDS = 5

    # 1. Load Wine Data
    wine_df = load_wine_data(CSV_FILE_PATH)
    if wine_df is None:
        exit() # Exit if data loading failed

    # 2. Transform Regression Problem into Binary Classification
    wine_df = transform_quality_to_binary_target(
        wine_df,
        quality_column=QUALITY_COLUMN,
        new_target_column=TASTY_TARGET_COLUMN,
        threshold=QUALITY_THRESHOLD
    )

    # 3. Prepare Features and Target Arrays
    X, y = prepare_features_target_arrays(wine_df, FEATURE_COLUMNS, TASTY_TARGET_COLUMN)

    # 4. Preprocess Data: Scale Features
    X_scaled = scale_features(X)

    # 5. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y,
        test_size=TEST_DATA_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )

    # 6. Tune AdaBoost Hyperparameters using GridSearchCV
    grid_search_result = tune_adaboost_hyperparameters(
        X_train, y_train,
        random_state=RANDOM_SEED,
        cv_folds=GRID_SEARCH_CV_FOLDS
    )

    # Get the best estimator found by GridSearchCV
    best_adaboost_model = grid_search_result.best_estimator_

    # 7. Evaluate the Best Model on the Held-Out Test Set
    evaluate_model(best_adaboost_model, X_test, y_test)

    print("\nScript execution complete.")
