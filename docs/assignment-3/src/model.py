import pandas as pd


DATA_ROOT = './data/'


# Extract the data and separate the labels
dataset = pd.read_csv(DATA_ROOT + 'spam.csv')
train_labels = dataset['v1']
train_data = dataset['v2']

