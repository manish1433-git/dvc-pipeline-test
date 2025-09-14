import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

# setting project dir
project_dir=Path(__file__).resolve().parents[2]

parameters=yaml.safe_load(open(f'{project_dir}/params.yaml','r'))['build_features']['max_features']
# fetch the data from data/processed
train_data = pd.read_csv(f'{project_dir}/data/preprocessed/train_processed.csv')
test_data = pd.read_csv(f'{project_dir}/data/preprocessed/test_processed.csv')

train_data.fillna('',inplace=True)
test_data.fillna('',inplace=True)

# apply BoW
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=parameters)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df['label'] = y_test

# store the data inside data/features
#data_path = os.path.join("data","features")

os.makedirs(f'{project_dir}/data/features/')

train_df.to_csv(f'{project_dir}/data/features/train_bow.csv')
test_df.to_csv(f'{project_dir}/data/features/test_bow.csv')
