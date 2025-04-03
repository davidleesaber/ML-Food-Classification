# CSC311 Machine Learning Challenge: Predicting Food Preferences

## Project Overview
This repository contains the code and results for the **CSC311 Machine Learning Challenge**, where the goal was to develop a classifier to predict food preferences based on survey responses. The dataset provided included responses to various questions about food items, such as complexity, ingredients, settings, and associations. Our team applied machine learning techniques to preprocess the data, train models, and make predictions on unseen test data.

## Survey Questions and Expected Answers
The dataset included the following survey questions:
1. **Complexity (Scale 1-5)**: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most
complex)
2. **Number of Ingredients**: How many ingredients would you expect this food item to contain? (A number).
3. **Serving Setting**: In what setting would you expect this food to be served? Please check all that apply ("Weekday lunch", "Weekday dinner", "Weekend lunch", "Weekend dinner", "At a party").
5. **Price**: How much would you expect to pay for one serving of this food item? (A number).
6. **Associated Movie**: What movie do you think of when thinking of this food item? (e.g., "Cloudy with a Chance of Meatballs").
7. **Drink Pairing**: What drink would you pair with this food item? (e.g., "Coke").
8. **Person Association**: When you think about this food item, who does it remind you of? Please check all that apply ("Parents", "Siblings", "Friends", "Teachers", "Strangers").
9. **Hot Sauce Preference**: How much hot sauce would you add to this food item? Select one ("None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)", "I will have some of this food item with my hot sauce")

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
