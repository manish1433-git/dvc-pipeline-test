import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df = df.drop('tweet_id', axis=1)
final_df = df[df['sentiment'].isin(['neutral','sadness'])].copy()
final_df['sentiment'] = final_df['sentiment'].replace({'happiness':0,'sadness':1})

train_data,test_data=train_test_split(final_df,test_size=0.2,random_state=42)

project_dir = Path(__file__).resolve().parents[2]
# Create the processed directory if it doesn't exist
os.makedirs(f'{project_dir}/data/processed', exist_ok=True)

# Save the train and test data
train_data.to_csv(f'{project_dir}/data/processed/train_data.csv', index=False)
test_data.to_csv(f'{project_dir}/data/processed/test_data.csv', index=False)