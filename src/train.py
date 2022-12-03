import sys
from pathlib import Path
import yaml

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT/'data'/'out'

if __name__ == '__main__':
    df = pd.read_pickle(OUT_DIR/'preprocess.pkl')

    train_valid = df.query('is_Test == 0').copy()
    test = df.query('is_Test == 1').copy()

    train, valid = train_test_split(train_valid, test_size=0.2, shuffle=True, random_state=0)

    train_X = train.drop(columns=['Survived', 'is_Test'])
    train_y = train['Survived']
    valid_X = valid.drop(columns=['Survived', 'is_Test'])
    valid_y = valid['Survived']
    test_X = test.drop(columns=['Survived', 'is_Test'])

    clf = tree.DecisionTreeClassifier().fit(train_X, train_y)
    accuracy = (clf.predict(valid_X) == valid_y).mean().tolist()
    obj = {'accuracy': accuracy}
    with open('data/out/summary.yaml', 'w') as f:
        yaml.dump(obj, f, encoding='utf-8')

    test['Survived'] = clf.predict(test_X)
    test[['PassengerId', 'Survived']].to_csv('data/out/gender_submission.csv', index=None)