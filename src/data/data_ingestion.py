import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

project_dir = Path(__file__).resolve().parents[2]
logger=logging.getLogger('ingestion logging')
logger.setLevel('DEBUG')

# adding a stread handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# adding a file handler
file_handler=logging.FileHandler(f'{project_dir}/logs/error.log')
file_handler.setLevel('ERROR')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



def load_params(file_name: str) -> float:
    project_dir = Path(__file__).resolve().parents[2]
    try:
        parameters=yaml.safe_load(open(f'{project_dir}/{file_name}','r'))['data_ingestion']['test_size']
        return parameters
    except Exception as e:
        logger.error(f'some error has occured while loading parameters from file {file_name} {e}')
        raise

def read_data(url: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f'some error has occured while fetching the dataset, {e}')
        raise
#df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')


def process(df: pd.DataFrame) -> pd.DataFrame:

    try:
        logger.info('preprocessing starting')
        df = df.drop('tweet_id', axis=1)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness':0,'sadness':1})
        logger.info('preprocessing completed')
        return final_df
    except Exception as e:
        logger.error(f'Error while preprocessing {e}')
        raise


def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,path: str) -> None: 
    project_dir = Path(__file__).resolve().parents[2]

    try:
        os.makedirs(f'{project_dir}/{path}', exist_ok=True)    
        train_data.to_csv(f'{project_dir}/{path}/train_data.csv', index=False)
        test_data.to_csv(f'{project_dir}/{path}/test_data.csv', index=False)
    except Exception as e:
        print(e)
        raise


def main():
    test_size=load_params('params.yaml')
    df= read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df=process(df)
    train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
    save_data(train_data,test_data,'data/processed')


if __name__ == '__main__':
    main()