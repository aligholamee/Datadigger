import numpy as np
import pandas as pd

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
def compute_nans_desc(df):
    '''
        returns a sorted dictionary consisting of column names and the number of NaNs in each column
    '''

    nans_dict = {}

    for col in df:
        nan_col_counter = 0
        for row in df[col]:
            if(row == '?'):
                nan_col_counter += 1
        nans_dict[str(col)] = [nan_col_counter]

    return nans_dict

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
nan_sorted_cols = compute_nans_desc(train_data)
separate_output('NaNs in Columns')
print(nan_sorted_cols)
