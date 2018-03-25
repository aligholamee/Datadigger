import numpy as np
import pprint 
import pandas as pd
from collections import deque

# Define root data path
DATA_ROOT = './data/'

# A function to separate the different outputs of each section.
# This will be used to display the data in the terminal in a readable way.
def separate_output(str):
    '''
        Displays an string as an argument in a clear form
    '''
    SHARP_COUNT = 100
    print('\n')
    for i in range(SHARP_COUNT):
        if(i == SHARP_COUNT-1):
            print("#")
        else:
            print("#", end="")

    # Display info at the center
    for i in range(int((SHARP_COUNT/2-len(str)/2))):
        print("",end=" ")

    print(str)

    for i in range(SHARP_COUNT):
        print("#", end="")
    print('\n')

# This function computes the number of missing values of each column in descending order 
def compute_nans(df):
    '''
        returns a dictionary consisting of column names and the number of NaNs in each column
    '''

    nans_dict = {}

    for col in df:
        nan_col_counter = 0
        for row in df[col]:
            if(row == '?'):
                nan_col_counter += 1
        nans_dict[str(col)] = nan_col_counter

    return nans_dict

# Fills in the missing values using the average of the given column
def fill_col_with_mean(df, col_index, specifier):
    '''
        returns a new dataframe in which its missing values in a given column with
        a given specifier are filled with the average of that column
    '''
    sum = 0
    count = 0
    col_mean = 0

    # Find the mean
    for row_index, row_value in df.iterrows():
        if(row_value[col_index] != '?'):
            sum = sum + row_value
            count += 1
    col_mean = sum / count

    for row_index, row_value in df.iterrows():
        if(row_value[col_index] == '?'):
            df[row_index][col_index] = col_mean

    return df[col_index]

# Load data
train_data = pd.read_csv(DATA_ROOT + 'train.csv')
test_data = pd.read_csv(DATA_ROOT + 'test.csv')

# Generate column names
COLUMN_NAMES = ['col_' + str(i+1) for i in range(train_data.shape[1])]

# Rename dataframe columns
train_data.columns = COLUMN_NAMES

# Get datatypes
separate_output('Training Data Types')
print(train_data.dtypes)

# Describe the data
separate_output('Statistical Information')
print(train_data.describe())

separate_output('Counts Values on a Column')
print(train_data['col_1'].value_counts())

# Get the number of missing values in a descending order
nan_cols = compute_nans(train_data)
separate_output('NaNs in Columns')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(nan_cols)

# Drop the columns with more than 500 missing values
separate_output('Columns After Drop')
cols_to_drop = [key for key, value in nan_cols.items() if value > 500]
train_data = train_data.drop(cols_to_drop, axis=1)
print(train_data.dtypes)
print(train_data.shape)

# Fill in the missing values of column 8 using its average
separate_output('Filled Missing in Column 8')
new_col = fill_col_with_mean(train_data, col_index='col_8', specifier='?')
print(new_col)