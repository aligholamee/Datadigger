import numpy as np
import pandas as pd

# Define root data path
DATA_ROOT = './data/'

# Load data
train_data = pd.read_csv(DATA_ROOT + 'train.csv')
test_data = pd.read_csv(DATA_ROOT + 'test.csv')

# Generate column names
COLUMN_NAMES = ['col_' + str(i+1) for i in range(train_data.shape[1])]

# Rename dataframe columns
train_data.columns = COLUMN_NAMES

# Get datatypes
print(train_data.dtypes)

