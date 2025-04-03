import pandas as pd
import numpy as np
import json
import re


# Load saved model artifacts
artifacts = np.load('rf_model_artifacts_with_code.npz')
scaler_params = json.loads(artifacts['scaler_params'].item())
vocab_movie = json.loads(artifacts['vocab_movie'].item())
vocab_drink = json.loads(artifacts['vocab_drink'].item())
trees = json.loads(artifacts['trees'].item())

# Label index mapping (order matters!)
LABELS = ['pizza', 'shawarma', 'sushi']

def standardize_numeric(row, feature_name, mean, std):
    val = row[feature_name]
    return (val - mean) / std if std != 0 else 0.0

def tfidf_vectorize(text, vocab):
    vec = np.zeros(len(vocab))
    if pd.isna(text):
        return vec
    for word in text.lower().split():
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

def predict_tree(tree, x):
    node = 0
    while tree['children_left'][node] != -1:
        feature_index = tree['feature'][node]
        threshold = tree['threshold'][node]
        if x[feature_index] <= threshold:
            node = tree['children_left'][node]
        else:
            node = tree['children_right'][node]
    return np.argmax(tree['value'][node])

def predict_all(csv_filename):
    # df = pd.read_csv(csv_filename)
    df = clean_data(csv_filename)
    predictions = []

    for _, row in df.iterrows():
        # Apply domain-specific price rules
        if row['Q4_price'] == -1:
            predictions.append('pizza')
            continue
        elif row['Q4_price'] == -2:
            predictions.append('sushi')
            continue
        # elif row['Q4_price'] == -3:
        #     predictions.append('pizza')  # Or use more logic
        #     continue

        # Standardize numeric features
        x_numeric = []
        for i, feature in enumerate(scaler_params['features']):
            x_numeric.append(standardize_numeric(row, feature, scaler_params['mean'][i], scaler_params['scale'][i]))

        # Vectorize text features
        vec_movie = tfidf_vectorize(str(row.get('Q5_movie', '')), vocab_movie)
        vec_drink = tfidf_vectorize(str(row.get('Q6_drink', '')), vocab_drink)

        # Combine all features
        x = np.concatenate([x_numeric, vec_movie, vec_drink])

        # Run each tree and vote
        votes = [predict_tree(tree, x) for tree in trees]
        majority = max(set(votes), key=votes.count)
        predictions.append(LABELS[majority])

    return [prediction.title() for prediction in predictions]

def clean_data(input_csv_filename):
    # Load the CSV file
    df = pd.read_csv(input_csv_filename)

    # Process Q1 - leave as is (complexity rating)
    df['Q1_complexity'] = pd.to_numeric(df[
                                            'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'
                                        ],
                                        errors='coerce')


    df['Q2_ingredients'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(extract_ingredient_count)

    # Process Q3 - Break into binary columns
    settings = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
    for setting in settings:
        df[f'Q3_setting_{setting.replace(" ", "_").lower()}'] = df['Q3: In what setting would you expect this food to be served? Please check all that apply'].str.contains(setting, na=False).astype(int)

    df['Q4_price'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(extract_price)


    # Process Q5 - Break up OR lines into multiple rows and convert to lowercase
    df = expand_on_or(df, 'Q5: What movie do you think of when thinking of this food item?', 'Q5_movie')
    df['Q5_movie'] = df['Q5_movie'].fillna('none').str.lower()

    df['Q5_movie'] = df['Q5_movie'].apply(standardize_movie)

    # Process Q6 - Break up OR lines into multiple rows and convert to lowercase
    df = expand_on_or(df, 'Q6: What drink would you pair with this food item?', 'Q6_drink')

    # Step 1: Lowercase all and lock in 'pop'
    df['Q6_drink'] = df['Q6_drink'].str.lower()

    df['Q6_drink'] = df['Q6_drink'].apply(standardize_drink)


    # Process Q7 - Create binary columns for relationships
    relationships = ['Parents', 'Siblings', 'Friends', 'Teachers', 'Strangers']
    for rel in relationships:
        df[f'Q7_has_{rel.lower()}'] = df['Q7: When you think about this food item, who does it remind you of?'].str.contains(rel, na=False).astype(int)

    df['Q8_hot_sauce_level'] = df['Q8: How much hot sauce would you add to this food item?'].apply(convert_hot_sauce_to_numeric)

    # ✅ Clean the label column
    df['Label'] = df['Label'].str.strip().str.lower()

    # ✅ Fill missing numeric fields with -99
    df['Q1_complexity'] = df['Q1_complexity'].fillna(-99)
    df['Q2_ingredients'] = df['Q2_ingredients'].fillna(-99)
    df['Q4_price'] = df['Q4_price'].fillna(-99)
    df['Q8_hot_sauce_level'] = df['Q8_hot_sauce_level'].fillna(-99)


    # Select final columns
    final_columns = ['id', 'Label',
                    'Q1_complexity', 'Q2_ingredients',
                    'Q3_setting_week_day_lunch', 'Q3_setting_week_day_dinner',
                    'Q3_setting_weekend_lunch', 'Q3_setting_weekend_dinner',
                    'Q3_setting_at_a_party', 'Q3_setting_late_night_snack',
                    'Q4_price', 'Q5_movie', 'Q6_drink',
                    'Q7_has_parents', 'Q7_has_siblings',
                    'Q7_has_friends', 'Q7_has_teachers',
                    'Q7_has_strangers', 'Q8_hot_sauce_level']

    transformed_df = df[final_columns]

    return transformed_df


# Process Q2 - convert to a single number
def extract_ingredient_count(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return value

    value_str = str(value).lower()

    # Try direct conversion first
    try:
        return float(value_str)
    except ValueError:
        pass

    # Handle ranges like "6-7" or "3 to 5"
    range_match = re.search(r'(\d+)[-\s]+(to[\s]+)?(\d+)', value_str)
    if range_match:
        start = float(range_match.group(1))
        end = float(range_match.group(3))
        return (start + end) / 2

    # Handle phrases like "at least 3" or "3 or more"
    at_least_match = re.search(r'at least (\d+)|(\d+) or more', value_str)
    if at_least_match:
        num_group = at_least_match.group(1) or at_least_match.group(2)
        if num_group:
            return float(num_group)

    # Extract the first number found
    numbers = re.findall(r'\d+\.?\d*', value_str)
    if numbers:
        return float(numbers[0])

    try:
        return word_to_num(value_str.strip('.'))
    except ValueError:
        ...

    # A list of items in a comma separated list
    words = [word.strip() for word in value_str.split(",")]
    if len(words) > 1 and "i" not in value_str.split(" "):
        return len(words)

    # A list of items in bullet form
    if "*" in value_str:
        return value_str.count("*")

    if "and" in value_str or "&" in value_str:
        return 2

    if "i" in value_str.split() or "?" in value_str:
        return np.nan

    return 1

# Process Q4 - Convert to a number, using mean for ranges
def extract_price(value):
    if pd.isna(value):
        return np.nan

    value_str = str(value).lower()

    # Remove currency symbols and words
    value_str = re.sub(r'[$£€¥]|\bdollars?\b|\bcad\b', '', value_str)

    # Handle ranges
    range_match = re.search(r'(\d+\.?\d*)[-\s]+(to[\s]+)?(\d+\.?\d*)', value_str)
    if range_match:
        start = float(range_match.group(1))
        end = float(range_match.group(3))
        return (start + end) / 2

    # Extract the first number
    numbers = re.findall(r'\d+\.?\d*', value_str)
    if numbers:
        return float(numbers[0])

     # Detect which category it belongs to according to some special words
    if 'slice' in value_str:
        return -1  # -1 implies pizza
    elif 'sushi' in value_str:
        return -2 # -2 implies sushi

    try:
        return word_to_num(value_str.strip('.'))
    except ValueError:
        ...

    return np.nan

# Function to expand rows based on OR values and convert to lowercase
def expand_on_or(df, column_name, new_column_name):
    result_rows = []

    for _, row in df.iterrows():
        value = row[column_name]
        if pd.isna(value):
            row_copy = row.copy()
            row_copy[new_column_name] = np.nan
            result_rows.append(row_copy)
        else:
            value_str = str(value).lower()  # Convert to lowercase here

            # Clean punctuation - SUGGESTION
            value_str = re.sub(r'[^\w\s]', '', value_str)

            if ' or ' in value_str:
                options = re.split(r' [Oo][Rr] ', value_str)
                for option in options:
                    row_copy = row.copy()
                    row_copy[new_column_name] = option.strip()
                    result_rows.append(row_copy)
            else:
                row_copy = row.copy()
                row_copy[new_column_name] = value_str
                result_rows.append(row_copy)

    return pd.DataFrame(result_rows)


# Standardize movie names
def standardize_movie(movie):
    if not movie or movie.strip() == '':
        return 'none'
    if 'ninja' in movie:
        return 'ninja turtles'
    if 'home alone' in movie:
        return 'home alone'
    if 'spider' in movie:
        return 'spider man'
    if 'no' in movie:
        return 'none'
    if 'idk' in movie:
        return 'none'
    if 'none' in movie:
        return 'none'
    if 'no movie' in movie:
        return 'none'
    if 'don\'t' in movie:
        return 'none'
    if 'avengers' in movie:
        return 'avengers'
    if 'sushi' in movie:
        return 'jiro dreams of sushi'
    if "cloudy" in movie and "meatball" in movie:
        return "cloudy with a chance of meatballs"
    if "your name" in movie:
        return "your name"
    if "shrek 2" in movie:
        return "shrek 2"

    if "i dont" in movie:
        return "none"

    words = movie.split()
    if "any" in words and "cant" in words:
        return "none"

    if "think" in words:
        movie_rep = movie
        for strip in ["i think", "think", " of", " about", " the", " movies", " movie", " like", " named"]:
            movie_rep = movie_rep.removeprefix(strip)
        for delim in "which", "when", "as":
            movie_rep = movie_rep.replace(delim, ";")
        movie_rep = movie_rep.split(";")[0].strip()

        movie_words = movie_rep.split()
        if len(movie_words) < 4:
            return movie_rep

        if "like" in movie_rep:
            return movie_rep.split("like")[1].strip()

    return movie.strip()


# Step 2: Apply 'pop' rule first and store in a new column
def standardize_drink(drink):
    if pd.isna(drink):
        return 'none'
    if 'pop' in drink:
        return 'pop'
    if re.search(r'coca[-\s]?cola|pepsi|\bcola\b', drink):
        return 'coke'
    if re.search(r'\b\w+\s+(coke|cola)\b', drink):
        return 'coke'
    if 'coke' in drink:
        return 'coke'
    if 'water' in drink:
        return 'water'
    if 'soda' in drink:
        return 'soda'
    if 'juice' in drink:
        return 'juice'
    if 'milk' in drink:
        return 'milk'
    if 'tea' in drink:
        return 'tea'
    if 'soup' in drink:
        return 'soup'
    if 'wine' in drink:
        return 'wine'
    if 'beer' in drink:
        return 'beer'
    if 'lemon' in drink:
        return 'lemonade'
    if 'soft' in drink:
        return 'pop'
    if 'cococola' in drink:
        return 'coke'
    if 'canada dry' in drink:
        return 'canada dry'
    if 'ginger ale' in drink:
        return 'ginger ale'
    if 'ayran' in drink:
        return 'ayran'
    if 'carbonated' in drink:
        return 'pop'
    if 'fanta' in drink:
        return 'fanta'
    if 'orange' in drink:
        return 'juice'
    if 'alcohol' in drink:
        return 'alchohol'
    if 'beer' in drink:
        return 'beer'
    if 'sake' in drink:
        return 'sake'
    if 'any' in drink:
        return 'any'
    if 'cocktail' in drink:
        return 'cocktail'
    if 'mango lassi' in drink:
        return 'mango lassi'
    if 'champagne' in drink:
        return 'champagne'
    if 'matcha' in drink:
        return 'matcha'
    if 'no drink' in drink:
        return 'none'

    return drink.strip()


# Process Q8 - Convert to numerical scale and fix missing "none" values
def convert_hot_sauce_to_numeric(value):
    if pd.isna(value):
        return 0  # Explicitly set missing "none" values to 0

    hot_sauce_map = {
        'none': 0,
        'a little (mild)': 1,
        'a moderate amount (medium)': 2,
        'a lot (hot)': 3,
        'i will have some of this food item with my hot sauce': 2
    }

    value_str = str(value).lower()
    for key, numeric_value in hot_sauce_map.items():
        if key in value_str:
            return numeric_value

    return np.nan


######################################
# From word2number library
# https://github.com/akshaynagpal/w2n
######################################


american_number_system = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,
    'billion': 1000000000,
    'point': '.'
}

decimal_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

"""
#TODO
indian_number_system = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100,
    'thousand': 1000,
    'lac': 100000,
    'lakh': 100000,
    'crore': 10000000
}
"""


"""
function to form numeric multipliers for million, billion, thousand etc.

input: list of strings
return value: integer
"""


def number_formation(number_words):
    numbers = []
    for number_word in number_words:
        numbers.append(american_number_system[number_word])
    if len(numbers) == 4:
        return (numbers[0] * numbers[1]) + numbers[2] + numbers[3]
    elif len(numbers) == 3:
        return numbers[0] * numbers[1] + numbers[2]
    elif len(numbers) == 2:
        if 100 in numbers:
            return numbers[0] * numbers[1]
        else:
            return numbers[0] + numbers[1]
    else:
        return numbers[0]


"""
function to convert post decimal digit words to numerial digits
input: list of strings
output: double
"""


def get_decimal_sum(decimal_digit_words):
    decimal_number_str = []
    for dec_word in decimal_digit_words:
        if(dec_word not in decimal_words):
            return 0
        else:
            decimal_number_str.append(american_number_system[dec_word])
    final_decimal_string = '0.' + ''.join(map(str,decimal_number_str))
    return float(final_decimal_string)


"""
function to return integer for an input `number_sentence` string
input: string
output: int or double or None
"""


def word_to_num(number_sentence):
    if type(number_sentence) is not str:
        raise ValueError("Type of input is not string! Please enter a valid number word (eg. \'two million twenty three thousand and forty nine\')")

    number_sentence = number_sentence.replace('-', ' ')
    number_sentence = number_sentence.lower()  # converting input to lowercase

    if(number_sentence.isdigit()):  # return the number if user enters a number string
        return int(number_sentence)

    split_words = number_sentence.strip().split()  # strip extra spaces and split sentence into words

    clean_numbers = []
    clean_decimal_numbers = []

    # removing and, & etc.
    for word in split_words:
        if word in american_number_system:
            clean_numbers.append(word)

    # Error message if the user enters invalid input!
    if len(clean_numbers) == 0:
        raise ValueError("No valid number words found! Please enter a valid number word (eg. two million twenty three thousand and forty nine)")

    # Error if user enters million,billion, thousand or decimal point twice
    if clean_numbers.count('thousand') > 1 or clean_numbers.count('million') > 1 or clean_numbers.count('billion') > 1 or clean_numbers.count('point')> 1:
        raise ValueError("Redundant number word! Please enter a valid number word (eg. two million twenty three thousand and forty nine)")

    # separate decimal part of number (if exists)
    if clean_numbers.count('point') == 1:
        clean_decimal_numbers = clean_numbers[clean_numbers.index('point')+1:]
        clean_numbers = clean_numbers[:clean_numbers.index('point')]

    billion_index = clean_numbers.index('billion') if 'billion' in clean_numbers else -1
    million_index = clean_numbers.index('million') if 'million' in clean_numbers else -1
    thousand_index = clean_numbers.index('thousand') if 'thousand' in clean_numbers else -1

    if (thousand_index > -1 and (thousand_index < million_index or thousand_index < billion_index)) or (million_index>-1 and million_index < billion_index):
        raise ValueError("Malformed number! Please enter a valid number word (eg. two million twenty three thousand and forty nine)")

    total_sum = 0  # storing the number to be returned

    if len(clean_numbers) > 0:
        # hack for now, better way
        if len(clean_numbers) == 1:
                total_sum += american_number_system[clean_numbers[0]]

        else:
            if billion_index > -1:
                billion_multiplier = number_formation(clean_numbers[0:billion_index])
                total_sum += billion_multiplier * 1000000000

            if million_index > -1:
                if billion_index > -1:
                    million_multiplier = number_formation(clean_numbers[billion_index+1:million_index])
                else:
                    million_multiplier = number_formation(clean_numbers[0:million_index])
                total_sum += million_multiplier * 1000000

            if thousand_index > -1:
                if million_index > -1:
                    thousand_multiplier = number_formation(clean_numbers[million_index+1:thousand_index])
                elif billion_index > -1 and million_index == -1:
                    thousand_multiplier = number_formation(clean_numbers[billion_index+1:thousand_index])
                else:
                    thousand_multiplier = number_formation(clean_numbers[0:thousand_index])
                total_sum += thousand_multiplier * 1000

            if thousand_index > -1 and thousand_index != len(clean_numbers)-1:
                hundreds = number_formation(clean_numbers[thousand_index+1:])
            elif million_index > -1 and million_index != len(clean_numbers)-1:
                hundreds = number_formation(clean_numbers[million_index+1:])
            elif billion_index > -1 and billion_index != len(clean_numbers)-1:
                hundreds = number_formation(clean_numbers[billion_index+1:])
            elif thousand_index == -1 and million_index == -1 and billion_index == -1:
                hundreds = number_formation(clean_numbers)
            else:
                hundreds = 0
            total_sum += hundreds

    # adding decimal part to total_sum (if exists)
    if len(clean_decimal_numbers) > 0:
        decimal_sum = get_decimal_sum(clean_decimal_numbers)
        total_sum += decimal_sum

    return total_sum


if __name__ == "__main__":
    input_file = "cleaned_data_combined_modified.csv"
    preds = predict_all(input_file)
    df = clean_data(input_file)

    total_right = 0
    for i, p in enumerate(preds):
        label = df.iloc[i][df.columns[1]].title()
        correct_pred = label == p
        print(f"Row {i}: {p} ({label}; {correct_pred})")
        total_right += correct_pred
    print(f"Final accuracy: ({total_right}/{len(preds)})", total_right / len(preds))
