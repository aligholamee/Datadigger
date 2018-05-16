import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer

DATA_ROOT = './data/'


def extract_labels(data):
    """
        Extracts raw data and Its labels
    """
    
    return (dataset['v2'], data['v1'])
    
# Extract the data and separate the labels
dataset = pd.read_csv(DATA_ROOT + 'spam.csv')
train_data, train_labels = extract_labels(dataset)

print(train_data)



