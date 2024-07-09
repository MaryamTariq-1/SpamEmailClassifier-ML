# SpamEmailClassifier-ML

## Overview

This repository contains the implementation of a Spam Email Classification project using various machine learning algorithms. The aim of this project is to categorize emails into predefined classes (spam or not spam) and to master classification techniques essential for predictive modeling.

## Project Goals

- Perform comprehensive data preparation, model selection, training, and evaluation for spam email classification.
- Implement and compare multiple classification algorithms such as Logistic Regression, Decision Trees, Support Vector Machines, Random Forest, and Gradient Boosting.
- Optimize model performance through hyperparameter tuning using Grid Search.
- Visualize evaluation metrics and interpret the results to choose the best-performing model.

## Tools and Technologies Used

- Python
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib

## Project Steps

### 1. Data Preparation

- Load the dataset and inspect its structure.
- Handle missing values by dropping or imputing them.
- Combine text columns into a single feature using TfidfVectorizer.
- Encode categorical target variables.

### 2. Model Selection

- Explore and implement various classification algorithms:
  - Logistic Regression
  - Decision Trees
  - Support Vector Machines
  - Random Forest
  - Gradient Boosting

### 3. Model Training

- Split the dataset into training and testing sets.
- Train each model on the training data.

### 4. Model Evaluation

- Evaluate each model's performance using metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Visualize results using a confusion matrix and ROC curve.

### 5. Hyperparameter Tuning

- Use Grid Search to optimize hyperparameters for the best-performing model.

### 6. Pipeline Creation

- Create a pipeline combining the vectorizer and the best model for seamless predictions.

## Results and Metrics

- Evaluate models based on their performance metrics such as accuracy, precision, recall, and F1 score.
- Select the best-performing model based on F1 score and cross-validation results.
- Visualize the ROC curve to assess model discrimination ability.

## Repository Structure

- `emails.csv`: Dataset containing email text and labels.
- `classification_spam_email_classification.ipynb`: Jupyter Notebook with the complete implementation of the project.
- `tfidf_vectorizer.pkl`: Pickled TfidfVectorizer for transforming text data.
- `spam_classifier_model.pkl`: Pickled best classification model after training and tuning.
- `spam_classifier_pipeline.pkl`: Pickled pipeline combining vectorizer and best model for direct use.


## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact:
- **Your Name:** Maryam Tariq
- **LinkedIn:** [Maryam Tariq](https://www.linkedin.com/in/maryamtariq1/)

