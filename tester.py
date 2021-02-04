# test the performance of a prediction
import numpy as np
import argparse
import json
import jsonlines
from sklearn.metrics import f1_score
import pandas as pd
import re
import os

def f1_above_theta(df, theta):
    # compute the f1 score of pairs with confidence >= theta
    df_slice = df.loc[df["match_confidence"]>=theta,["match","label"]]
    if df_slice["label"].count() == 0: # no tuples to predict
        return 1.0
    f1 = f1_score(df_slice["label"],df_slice["match"])
    return f1


def f1_all_with_manual(df,theta):
    # manually label all pairs with confidence < theta, and compute f1 score
    df_cp = df.loc[:,["match","label","match_confidence"]].copy()
    df_cp.loc[df_cp["match_confidence"]< theta,"match"] = df_cp.loc[df_cp["match_confidence"]<theta,"label"]
    f1 = f1_score(df_cp["label"],df_cp["match"])
    return f1


def ratio_above_theta(df,theta):
    # get the ratio of pairs with confidence below theta
    df_slice = df.loc[df["match_confidence"]>=theta,["match","label"]]
    label_length = df_slice["match"].count()
    total_length = df["match"].count()
    return label_length/total_length


def f1_with_manual_ratio(df, manual_ratio):
    # manually label a fixed ratio of pairs with lowest confidence
    df_cp = df.loc[:,["match","label"]].copy()
    total_length = df_cp["match"].count()
    label_length = int(total_length * manual_ratio)
    df_cp.iloc[0:label_length,0] = df_cp.iloc[0:label_length,1]
    # print(df_cp.head(100))
    f1 = f1_score(df_cp["label"],df_cp["match"])
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",type=str,default=None)
    hp = parser.parse_args()
    result = {"left": [],
              "right": [],
              "match": [],
              "match_confidence": [],
              "label": []}
    if not os.path.exists(hp.input_path):
        print("File %s not exist. Abort." % hp.input_path)
        exit()

    with jsonlines.open(hp.input_path) as reader:
        for obj in reader:
            result["left"].append(obj["left"])
            result["right"].append(obj["right"])
            result["match"].append(int(obj["match"]))
            result["match_confidence"].append(float(obj["match_confidence"]))
            result["label"].append(int(obj["label"]))

    df = pd.DataFrame(result).sort_values(by="match_confidence",ascending=True)
    # print(df.loc[:,["match","label","match_confidence"]])

    f1 = f1_score(df["label"],df["match"])
    print("%s, %f" % (hp.input_path,f1))

    theta_list = np.linspace(0.5,1,20)
    f1_above_list = []
    f1_all_list = []
    ratio_list = []
    for theta in theta_list:
        f1_above = f1_above_theta(df,theta)
        f1_all = f1_all_with_manual(df,theta)
        ratio = ratio_above_theta(df,theta)
        f1_above_list.append(f1_above)
        f1_all_list.append(f1_all)
        ratio_list.append(ratio)

    df_theta = pd.DataFrame({"theta":theta_list,
                             "f1_above":f1_above_list,
                             "f1_all":f1_all_list,
                             "ratio_above":ratio_list})


    ratio_list = np.linspace(0,0.3,20)
    f1_all_list = []
    for ratio in ratio_list:
        f1_all = f1_with_manual_ratio(df,ratio)
        f1_all_list.append(f1_all)

    df_ratio = pd.DataFrame({"ratio":ratio_list,
                             "f1_all":f1_all_list})

    output_path_theta = re.sub('\.jsonl$','_theta.csv',hp.input_path)
    df_theta.to_csv(output_path_theta)
    output_path_ratio = re.sub('\.jsonl$','_ratio.csv',hp.input_path)
    df_ratio.to_csv(output_path_ratio)


