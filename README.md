# Encryptix_ML
# Project 1 - Movie Genre Classification

This project aims to classify movie genres based on their plot summaries using machine learning techniques. We use text processing methods like TF-IDF vectorization and classifiers such as Support Vector Machines (SVM) to predict the genre of a movie.

## Dataset
The project uses three datasets:
1. **Training Data**: Contains movie titles, genres, and descriptions.
2. **Test Data**: Contains movie IDs, titles, and descriptions for which genres need to be predicted.
3. **Test Solutions**: Contains the actual genres for the test data for evaluation purposes.

Each dataset is provided in a text file with the following formats:
- Training Data: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- Test Data: `ID ::: TITLE ::: DESCRIPTION`
- Test Solutions: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`

## Exploratory Data Analysis (EDA)
Performed EDA to understand the distribution of movie genres in the training dataset and check for any missing values. This involves:
- Visualizing the distribution of genres using a count plot.
- Checking for missing values in the training and test datasets.

## Model Training and Evaluation

### Data Preprocessing
- Combine the title and description columns to create a unified text field for both training and test datasets.
- Convert the text data into numerical features using TF-IDF vectorization.

### Model
- Train a Support Vector Machine (SVM) classifier using the TF-IDF features extracted from the training data.
- Predict the genres for the test data using the trained SVM model.

### Evaluation
- Compare the predicted genres with the actual genres from the test solutions.
- Calculate the accuracy of the predictions.

## Results
The model's performance is evaluated using the accuracy metric.

# Project 2 - Credit Card Fraud Detection
This project aims to detect fraudulent credit card transactions using machine learning models. The dataset contains information about credit card transactions, including details such as transaction time, amount, merchant, and various demographic details about the cardholder.

## Project Overview
Credit card fraud detection is critical for financial institutions to prevent unauthorized transactions and protect customers. This project explores various machine learning algorithms to classify transactions as fraudulent or legitimate.

## Dataset
The dataset contains the following columns:

- trans_date_trans_time: Transaction date and time
- cc_num: Credit card number
- merchant: Merchant name
- category: Merchant category
- amt: Transaction amount
- first: First name of the cardholder
- last: Last name of the cardholder
- gender: Gender of the cardholder
- street: Street address of the cardholder
- city: City of the cardholder
- state: State of the cardholder
- zip: Zip code of the cardholder
- lat: Latitude of the cardholder's address
- long: Longitude of the cardholder's address
- city_pop: Population of the city
- job: Job of the cardholder
- dob: Date of birth of the cardholder
- trans_num: Transaction number
- unix_time: Transaction time in Unix format
- merch_lat: Latitude of the merchant's location
merch_long: Longitude of the merchant's location
- is_fraud: Target variable indicating if the transaction is fraudulent (1) or legitimate (0)

## Exploratory Data Analysis (EDA)

- Load and inspect data
- Basic statistics
- Check for missing values
- Distribution of target variable
- Correlation matrix
- Feature distributions
- Data Preprocessing

## Handle missing values
- Encode categorical variables using LabelEncoder
- Scale numerical features using StandardScaler
- Split features and labels for training and testing
  
## Model Selection and Training
- Logistic Regression
- Decision Tree
- Random Forest
  
## Model Evaluation
Evaluate models using classification report and ROC-AUC score

# Project 3 - SMS Spam Detection
The SMS Spam Detection project aims to classify SMS messages as either spam or legitimate (ham) using various machine learning models. This project utilizes a dataset of 5,574 SMS messages, labeled as 'ham' (legitimate) or 'spam'.

## Dataset
The dataset is sourced from the UCI Machine Learning Repository. It contains 5,574 SMS messages in English, with each message labeled as 'ham' or 'spam'.

### File Format
Each line in the file consists of two columns:
v1: The label (ham or spam)
v2: The raw text of the SMS message

## Models Used
The project utilizes three different machine learning models to classify the SMS messages:
- Naive Bayes: A probabilistic classifier based on Bayes' theorem.
- Logistic Regression: A linear model for binary classification.
- Support Vector Machine (SVM): A linear model for classification.
  
## Evaluation
The models are evaluated using the following metrics:
- Accuracy: The proportion of correctly classified messages.
- Precision: The proportion of true positives out of all predicted positives.
- Recall: The proportion of true positives out of all actual positives.
- F1-Score: The harmonic mean of precision and recall.
