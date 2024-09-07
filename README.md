# Machine-Learning-Project
URL Classification Using Machine Learning

Overview
This project classifies URLs as either "Legitimate" or "Phishing" using machine learning techniques. The process involves inputting web URLs, extracting and selecting features, preprocessing data, scaling features, training models, and evaluating their performance. The final model classifies new URLs based on the best-performing machine learning model.

Project Structure
Web URL Input: Input and preprocess URLs for feature extraction.
Feature Selection: Extract and select relevant features from URLs.
Feature Vector Creation: Construct feature vectors from the selected features.
Historical Dataset: Load and preprocess historical datasets to train the model.
Data Preprocessing: Prepare the data for model training and evaluation.
Feature Scaling: Normalize the feature values for improved model performance.
Feature Selection: Identify and select the most important features for model training.
Machine Learning Models: Train and evaluate different machine learning models.
Hyperparameter Tuning: Optimize model performance through hyperparameter tuning.
Model Selection, Training, and Evaluation Metrics: Select the best model, train it, and evaluate its performance using various metrics.
Print Best Model: Output the details of the best-performing model.
Classification: Classify new URLs using the trained model.
Requirements
Python 3.x
Pandas
Scikit-learn
Requests
BeautifulSoup4
WhoIs
Numpy
Installation
To install the required libraries, use the following pip command:
pip install pandas scikit-learn requests beautifulsoup4 whois numpy

Steps
1. Web URL Input
Define methods for inputting and preprocessing web URLs. This step involves fetching and preparing URLs for feature extraction.

2. Feature Selection
Extract and analyze various types of features from URLs:

URL-Based Features: Derived directly from the URL, such as:

URL length
Presence of IP addresses
Use of URL shortening services
Number of special characters
Domain-Based Features: Related to the domain of the URL, including:

Domain age
SSL certificate status
Content-Based Features: Attributes obtained by analyzing the content of the URL’s web page, such as:

Length of HTML content
Length of JavaScript content
Number of hyperlinks
Number of forms on the page
3. Feature Vector Creation
Convert the selected features into a feature vector. This involves:

Combining all extracted features into a structured format.
Creating a feature matrix where each row represents a URL and each column represents a feature.
Ensuring that all feature vectors are consistent and have the same length, even if some features are absent.
4. Historical Dataset
Load and preprocess historical datasets to train the model:

Load Datasets: Import datasets containing labeled URLs (e.g., legitimate and phishing URLs).
Add Labels: Annotate the URLs with appropriate labels (0 for legitimate, 1 for phishing).
Combine and Clean Data: Merge datasets and remove duplicates to create a comprehensive dataset for training.
Preprocess Data: Sample URLs and handle missing values to prepare the dataset for feature extraction and model training.
5. Data Preprocessing
Prepare the data for model training and evaluation:

Handle Missing Values: Ensure that there are no NaN values in the dataset. Fill or impute missing values as necessary.
Split Data: Divide the dataset into training and test sets to evaluate model performance.
6. Feature Scaling
Normalize the feature values to improve model performance and convergence:

Standardization: Use StandardScaler to scale the features so that they have a mean of 0 and a standard deviation of 1.
StandardScaler is fitted to the training data and then used to transform both the training and test datasets to ensure consistent scaling.
This helps to balance the influence of different features on the model.
7. Feature Selection
Identify and select the most important features for model training:

Feature Importance: Analyze the importance of each feature in predicting the target variable.
Selection Methods: Use techniques like Recursive Feature Elimination (RFE) or feature importance from tree-based models to select the most relevant features.
8. Machine Learning Models
Train and evaluate various machine learning models to determine the best performer:

Models Used:

Random Forest: An ensemble method using multiple decision trees.
Support Vector Machine (SVM): A model that finds the optimal hyperplane for classification.
Gradient Boosting: An ensemble technique that builds models sequentially to correct errors made by previous models.
Evaluation Metrics: Assess models using metrics such as accuracy, precision, recall, and F1 score.

9. Hyperparameter Tuning
Optimize model performance through hyperparameter tuning:

Grid Search: Perform grid search to find the best parameters for each model.
Random Forest: Tune parameters like the number of estimators and maximum depth.
SVM: Tune parameters like the regularization parameter (C) and the kernel type.
Gradient Boosting: Tune parameters like the number of estimators and learning rate.
10. Model Selection, Training, and Evaluation Metrics
Select the best-performing model, train it, and evaluate its performance:

Model Selection: Choose the best model based on the evaluation metrics obtained from different models.
Training: Train the selected model using the training data.
Evaluation: Assess the trained model on the test data and calculate performance metrics such as accuracy, precision, recall, and F1 score to gauge its effectiveness.
11. Print Best Model
Output the details of the best-performing model:

Print information about the best model, including its type and performance metrics. This step helps to identify and confirm the most effective model for URL classification.
12. Classification
Classify new URLs using the trained model:

Feature Extraction: Extract features from the new URL using the same methods as for the training data.
Feature Vector Preparation: Convert the extracted features into a feature vector consistent with the training data format.
Scaling: Apply the same scaling transformations to the new URL features.
Prediction: Use the best model to predict whether the URL is "Legitimate" or "Phishing."



