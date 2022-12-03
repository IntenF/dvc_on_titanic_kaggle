import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT/'data'/'raw'
OUT_DIR = ROOT/'data'/'out'

if __name__ == '__main__':
    train_csv = Path(RAW_DIR/'train.csv')
    test_csv = Path(RAW_DIR/'test.csv')
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    test_df['Survived'] = -1
    train_df['is_Test'] = 0
    test_df['is_Test'] = 1

    concat_df = pd.concat([train_df, test_df])
    concat_df = pd.get_dummies(concat_df , dummy_na  = False, drop_first = False,columns=['Sex', 'Ticket', 'Cabin', 'Embarked'])
    concat_df['Age'] = concat_df['Age'].fillna(train_df['Age'].mean())
    concat_df['Fare'] = concat_df['Fare'].fillna(train_df['Fare'].mean())
    del concat_df['Name']

    concat_df.to_pickle(OUT_DIR/'preprocess.pkl')
