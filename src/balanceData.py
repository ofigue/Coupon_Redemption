# python balanceData.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import sklearn.neighbors._base

if __name__ == "__main__":
    train_val = pd.read_csv('../input/trainset.csv', index_col=False)
    
    X = train_val.loc[:, train_val.columns != 'target']
    y = train_val.loc[:, train_val.columns == 'target']
    
    Xcols = X.columns
    ycols = y.columns
    
    smt = SMOTE(n_jobs = -1) #sampling_strategy='auto', n_jobs = -1)
    X_sampled, y_sampled = smt.fit_resample(X, y)

    X = pd.DataFrame(X_sampled, columns=Xcols)
    y = pd.DataFrame(y_sampled, columns=ycols)
    
    trainSet = pd.concat([X, y], axis=1)
    trainSet.to_csv('../input/trainsetBalanced.csv', index=False)

