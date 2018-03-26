import numpy as np
import pprint 
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

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

# This function converts the object types to numeric types
def obj_to_num(df):
    '''
        returns a new dataframe with all object types converted to numeric type
    '''
    
    for column in df.columns:
        if(df[column].dtype == 'object'):
            df[column] = df[column].astype('category')
            df[column] = df[column].cat.codes
    
    return df

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

# Fill in the missing values of column 8 using its average
separate_output('Data with Filled Missing Values in Float Dtypes')
train_data = train_data.replace('?', np.NaN)            # Fix non standard missing values
train_data.col_8 = train_data.col_8.astype(float)       # Mean does not work for int
train_data['col_8'].fillna(train_data['col_8'].mean(), inplace=True)
print(train_data.dtypes)

# Bring out the training labels before anything stupid happens
separate_output('Separated Training Labels')
train_labels = train_data['col_39']
train_data = train_data.drop(['col_39'], axis=1)
print(train_labels)

# Convert the categorical data to the numeric form
# Select the columns with object type
separate_output('Converted Data to Numeric Format')
train_data = obj_to_num(train_data)
print(train_data)

# Standardize the data using Standard Scaler
# Standardize the data along each column
separate_output('Standardized Train Data')
train_data = StandardScaler().fit_transform(train_data)
print(train_data.shape)

# Dimensionality Reduction with PCA
pca = PCA(n_components=5)   # Since we have 5 classes
pca.fit(train_data)         # fit pca on train data

# Apply transformation on both test and train data
train_data = pca.transform(train_data)
# test_data = pca.transform(test_data)

# Training a decision tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, train_labels)