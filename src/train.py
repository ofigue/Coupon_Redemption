# Ref. https://github.com/abhishekkrthakur/mlframework
# to run it: % sh train.sh <technique>

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
#TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

#FOLD_MAPPPING = {
#    0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
#    1: [0, 2, 3, 4, 5, 6, 7, 8, 9],
#    2: [0, 1, 3, 4, 5, 6, 7, 8, 9],
#    3: [0, 1, 2, 4, 5, 6, 7, 8, 9],
#    4: [0, 1, 2, 3, 5, 6, 7, 8, 9],
#    5: [0, 1, 2, 3, 4, 6, 7, 8, 9],
#    6: [0, 1, 2, 3, 4, 5, 7, 8, 9],
#    7: [0, 1, 2, 3, 4, 5, 6, 8, 9],
#    8: [0, 1, 2, 3, 4, 5, 6, 7, 9],
#    9: [0, 1, 2, 3, 4, 5, 6, 7, 8]
#}

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    #df_test = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)     # train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)     # valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    #joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")