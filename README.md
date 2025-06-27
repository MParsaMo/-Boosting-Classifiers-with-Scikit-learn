🌟 Boosting Classifiers with Scikit-learn
This repository contains two machine learning classification projects that demonstrate the power of Boosting Algorithms in Scikit-learn:

Gradient Boosting Classifier on the Iris Dataset

AdaBoost Classifier (with GridSearchCV) on the Wine Quality Dataset (converted to binary classification)

🧰 Dependencies
Make sure the following Python libraries are installed:

pip install numpy pandas scikit-learn

📁 Project Structure
1. 🌸 Gradient Boosting on Iris Dataset
File: iris_gradient_boosting.py
Goal: Classify iris flowers into three species using petal/sepal measurements.

Steps:
Loads the Iris dataset from Scikit-learn

Splits data into 70% training and 30% testing

Trains a GradientBoostingClassifier with 100 estimators

Evaluates the model using:

Confusion matrix

Accuracy score

Sample Output:
[[19  0  0]
 [ 0 13  2]
 [ 0  0 11]]
Accuracy: 0.9555

2. 🍷 AdaBoost on Wine Quality Dataset
File: wine_adaboost_gridsearch.py
Goal: Predict whether wine is "tasty" or not by converting the regression problem (quality score) into a binary classification task.

Steps:
Loads wine.csv dataset (semicolon-separated)

Defines features related to wine chemistry

Converts quality to binary target tasty (1 if quality > 7, else 0)

Scales feature values using MinMaxScaler

Splits into 70% training and 30% testing (with stratification)

Uses GridSearchCV to tune n_estimators and learning_rate for AdaBoostClassifier

Evaluates the model with:

Confusion matrix

Accuracy score

Sample Output:
[[140  10]
 [  9  41]]
Accuracy: 0.9022

🧠 Summary Table
| Project             | Dataset         | Classifier                 | Tuning Method | Evaluation                  |
| ------------------- | --------------- | -------------------------- | ------------- | --------------------------- |
| Iris Classification | Iris (built-in) | GradientBoostingClassifier | Default       | Accuracy + Confusion Matrix |
| Wine Quality        | wine.csv        | AdaBoostClassifier         | GridSearchCV  | Accuracy + Confusion Matrix |


📌 Notes
Both models use boosting (ensemble of weak learners).

GradientBoostingClassifier is slower but often more accurate.

AdaBoostClassifier can be faster and benefits significantly from hyperparameter tuning.

Stratification in the wine dataset ensures class balance.

👨‍💻 Author
Built for hands-on learning and experimentation with boosting algorithms using Scikit-learn.
