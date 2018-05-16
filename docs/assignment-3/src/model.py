import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

DATA_ROOT = './data/'


def extract_labels(data):
    """
        Extracts raw data and Its labels
    """
    
    return (dataset['v2'], data['v1'])
    
# Extract the data and separate the labels
dataset = pd.read_csv(DATA_ROOT + 'spam.csv')
train_data, train_labels = extract_labels(dataset)


# Split train and test data
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2)

# Extract features using Bag of Words model
count_vect = CountVectorizer()
train_data = count_vect.fit_transform(train_data)

# Enable TF-IDF
tf_idf = TfidfTransformer()
train_data = tf_idf.fit_transform(train_data)

# Train the Naive Bayes classifier
clf = MultinomialNB().fit(train_data, train_labels)









