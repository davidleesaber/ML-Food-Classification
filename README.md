# CSC311 Machine Learning Challenge: Predicting Food Preferences

## Project Overview
This repository contains the code and results for the **CSC311 Machine Learning Challenge**, where the goal was to develop a classifier to predict food preferences based on survey responses. The dataset provided included responses to various questions about food items, such as complexity, ingredients, settings, and associations. Our team applied machine learning techniques to preprocess the data, train models, and make predictions on unseen test data.

## Survey Questions and Expected Answers
The dataset included the following survey questions:
1. **Complexity (Scale 1-5)**: Participants rated how complex it is to make the food item (e.g., 1 = simple, 5 = complex).
2. **Number of Ingredients**: Participants estimated the number of ingredients required (e.g., "5: cheese, sauce, dough, meat, vegetables").
3. **Serving Setting**: Participants selected settings where they would expect the food to be served (e.g., "Weekday lunch," "At a party").
4. **Price**: Participants estimated the price for one serving (e.g., "$10 for a slice").
5. **Associated Movie**: Participants associated the food item with a movie (e.g., "Cloudy with a Chance of Meatballs").
6. **Drink Pairing**: Participants suggested drinks that pair well with the food item (e.g., "Coke").
7. **Person Association**: Participants described who the food reminds them of (e.g., "Friends").
8. **Hot Sauce Preference**: Participants indicated how much hot sauce they would add (e.g., "A little (mild)").

## Code Execution Order
To run the code and reproduce results, follow these steps:
1. **`cleaneddata.py`**: Preprocesses the raw survey data by cleaning and normalizing features.
2. **`npz_generator.py`**: Converts cleaned data into `.npz` format for efficient model training.
3. **`test.py`**: Trains and evaluates the machine learning models on processed data.

## Libraries Used
The following Python libraries were utilized in this project:
- **NumPy**: For numerical computations and efficient data handling.
- **pandas**: For data preprocessing and manipulation.
- **scikit-learn**: For machine learning models, including logistic regression and random forests.
- **Matplotlib & Seaborn**: For visualizing feature importance and model performance.

## Results
Our final model achieved an accuracy of **91.88%** on unseen test data. The report includes detailed insights into feature engineering, hyperparameter tuning, and model selection strategies.

## Repository Structure
- `cleaneddata.py`: Data preprocessing script.
- `npz_generator.py`: Converts cleaned data into `.npz` format.
- `test.py`: Model training and evaluation script.
- `report.pdf`: Comprehensive documentation of methodology, results, and conclusions.
