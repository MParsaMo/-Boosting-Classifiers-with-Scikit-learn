import pandas as pd # Included for general data science context, and for better confusion matrix display
import numpy as np # Used by scikit-learn internally
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import datasets # To load the Iris dataset

def load_iris_data():
    """
    Loads the Iris dataset from scikit-learn.

    The Iris dataset is a classic and very easy multi-class classification dataset.
    It consists of 150 samples of iris flowers, with 4 features each,
    and 3 possible target classes (species of iris).

    Returns:
        sklearn.utils.Bunch: A scikit-learn Bunch object containing data, target,
                             and target_names.
    """
    print("Loading Iris dataset...")
    iris_data = datasets.load_iris()

    # Print some information for better understanding
    print("\n--- Iris Data Overview ---")
    print(f"Features (data) shape: {iris_data.data.shape}") # (150 samples, 4 features)
    print(f"Target (labels) shape: {iris_data.target.shape}") # (150 samples,)
    print("Feature names:", iris_data.feature_names)
    print("Target names (species):", iris_data.target_names)
    print("\nFirst 5 rows of features:")
    print(iris_data.data[:5])
    print("\nFirst 5 target labels:")
    print(iris_data.target[:5])
    return iris_data

def prepare_features_target(iris_dataset):
    """
    Prepares the features (X) and target (y) arrays from the Iris dataset.

    Args:
        iris_dataset (sklearn.utils.Bunch): The loaded Iris dataset.

    Returns:
        tuple: A tuple containing (features (X), target (y)).
    """
    features = iris_dataset.data  # The features (e.g., sepal length, sepal width, etc.)
    target = iris_dataset.target # The target labels (species: 0, 1, 2)
    return features, target

def split_data(features, target, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        features (numpy.ndarray): The feature data.
        target (numpy.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                      Ensures reproducibility.

    Returns:
        tuple: A tuple containing (features_train, features_test, target_train, target_test).
    """
    print(f"\nSplitting data into training ({(1-test_size)*100:.0f}%) and testing ({test_size*100:.0f}%) sets...")
    # `stratify=target` ensures that the proportion of classes in the training and testing sets
    # is roughly the same as in the original dataset. This is crucial for classification tasks.
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    print(f"Training set size: {len(features_train)} samples")
    print(f"Testing set size: {len(features_test)} samples")
    return features_train, features_test, target_train, target_test

def train_gradient_boosting_model(features_train, target_train, n_estimators=100, random_state=123):
    """
    Trains a Gradient Boosting Classifier model.

    Args:
        features_train (numpy.ndarray): Training features.
        target_train (numpy.ndarray): Training target.
        n_estimators (int): The number of boosting stages to perform.
        random_state (int): Controls the random seed for reproducibility.

    Returns:
        sklearn.ensemble.GradientBoostingClassifier: The trained Gradient Boosting model.
    """
    print(f"\n--- Training Gradient Boosting Classifier ---")
    print(f"  n_estimators: {n_estimators}")
    print(f"  random_state: {random_state}")
    # Initialize GradientBoostingClassifier
    # n_estimators: The number of weak learners (decision trees) to build sequentially.
    # random_state: Ensures reproducibility of the model's stochastic elements.
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    model_fitted = model.fit(features_train, target_train)
    print("Gradient Boosting model training complete.")
    return model_fitted

def evaluate_model(model, features_test, target_test, target_names):
    """
    Evaluates the trained Gradient Boosting model on the test data and prints performance metrics.

    Args:
        model (sklearn.ensemble.GradientBoostingClassifier): The trained model.
        features_test (numpy.ndarray): Testing features.
        target_test (numpy.ndarray): Testing target (true labels).
        target_names (list): List of class names for better confusion matrix readability.
    """
    print("\n--- Evaluating Model Performance on Test Set ---")
    prediction = model.predict(features_test)

    # Confusion Matrix: A table used to describe the performance of a classification model.
    # Rows represent true classes, columns represent predicted classes.
    print('\nConfusion Matrix:')
    cm = confusion_matrix(target_test, prediction)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    # Accuracy Score: The proportion of correctly classified instances.
    accuracy = accuracy_score(target_test, prediction)
    print(f'\nAccuracy Score: {accuracy:.4f}')

    # Classification Report: Provides precision, recall, f1-score for each class.
    print("\nClassification Report:")
    print(classification_report(target_test, prediction, target_names=target_names))

if __name__ == "__main__":
    # Define parameters for model and data splitting
    TEST_DATA_SPLIT_RATIO = 0.3
    GB_N_ESTIMATORS = 100
    GB_RANDOM_STATE = 123 # Original random_state from snippet

    # 1. Load the Iris Dataset
    iris_dataset = load_iris_data()
    if iris_dataset is None:
        exit() # Should not happen for built-in datasets

    # 2. Prepare Features (X) and Target (y)
    X, y = prepare_features_target(iris_dataset)

    # 3. Split Data into Training and Testing Sets
    # Using a fixed random_state for train_test_split for reproducibility
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=TEST_DATA_SPLIT_RATIO,
        random_state=42 # Using 42 for split consistency across projects
    )

    # 4. Train the Gradient Boosting Model
    gradient_boosting_classifier = train_gradient_boosting_model(
        X_train, y_train,
        n_estimators=GB_N_ESTIMATORS,
        random_state=GB_RANDOM_STATE
    )

    # 5. Evaluate the Model on the Test Set
    evaluate_model(gradient_boosting_classifier, X_test, y_test, iris_dataset.target_names)

    print("\nScript execution complete.")
