import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

DATA_ROOT = './data/'


def extract_labels(data):
    """
        Extracts raw data and Its labels
    """
    
    return (dataset['v2'], data['v1'])
    
# Extract the data and separate the labels
dataset = pd.read_csv(DATA_ROOT + 'spam.csv')
train_data, train_labels = extract_labels(dataset)

# Extract features using Bag of Words model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(train_data)

# Enable TF-IDF
tf_idf = TfidfTransformer()
X_train = tf_idf.fit_transform(X_train)
print(X_train.shape)






