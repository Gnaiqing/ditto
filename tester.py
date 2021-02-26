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

# def f1_all_with_manual(df,theta):
#     # manually label all pairs with confidence < theta, and compute f1 score
#     df_cp = df.loc[:,["match","label","match_confidence"]].copy()
#     df_cp.loc[df_cp["match_confidence"]< theta,"match"] = df_cp.loc[df_cp["match_confidence"]<theta,"label"]
#     f1 = f1_score(df_cp["label"],df_cp["match"])
#     return f1

def ratio_above_theta(df,theta):
    # get the ratio of pairs with confidence >= theta
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
    f1 = f1_score(df_cp["label"],df_cp["match"])
    return f1


def read_prediction_result(input_path):
    # read prediction result from input
    result = {"left": [],
              "right": [],
              "match": [],
              "match_confidence": [],
              "label": []}
    if not os.path.exists(input_path):
        print("File %s not exist. Abort." % input_path)
        return None

    with jsonlines.open(input_path) as reader:
        for obj in reader:
            result["left"].append(obj["left"])
            result["right"].append(obj["right"])
            result["match"].append(int(obj["match"]))
            result["match_confidence"].append(float(obj["match_confidence"]))
            result["label"].append(int(obj["label"]))

    df = pd.DataFrame(result)
    return df


def get_f1_above(df,theta_list,path_to_store):
    f1_above_list = []
    for theta in theta_list:
        f1_above = f1_above_theta(df,theta)
        f1_above_list.append(f1_above)

    if path_to_store is not None:
        df_f1_above = pd.DataFrame({"theta":theta_list,"F1 above":f1_above_list})
        df_f1_above.to_csv(path_to_store,index=False)
    return f1_above_list


def get_ratio_above(df,theta_list,path_to_store):
    ratio_above_list = []
    for theta in theta_list:
        ratio_above = ratio_above_theta(df,theta)
        ratio_above_list.append(ratio_above)

    if path_to_store is not None:
        df_ratio_above = pd.DataFrame({"theta":theta_list,"ratio above":ratio_above_list})
        df_ratio_above.to_csv(path_to_store,index=False)

    return ratio_above_list


def get_f1_with_oracle(df,ratio_list,path_to_store):
    f1_list = []
    for ratio in ratio_list:
        f1 = f1_with_manual_ratio(df,ratio)
        f1_list.append(f1)

    if path_to_store is not None:
        df_f1_with_oracle = pd.DataFrame({"ratio":ratio_list,"F1":f1_list})
        df_f1_with_oracle.to_csv(path_to_store,index=False)

    return f1_list


def get_brier_score(df):
    df_cp = df.copy()
    df_cp["brier"] = np.where(df["match"]==df["label"],
                              (1-df["match_confidence"])**2,
                              df["match_confidence"]**2)
    brier_score = df_cp["brier"].mean()
    return brier_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",type=str,default=None)
    hp = parser.parse_args()
    df = read_prediction_result(hp.input_path).sort_values(by="match_confidence",ascending=True)

    print("Evaluating ",hp.input_path)
    f1 = f1_score(df["label"],df["match"])
    print("F1 score: ",f1)
    brier = get_brier_score(df)
    print("Brier score: ",brier)
    os.system("echo %s, %f, %f >> tester_result.csv" % (hp.input_path,f1,brier))

    theta_list = np.linspace(0.5,1,21)
    f1_above_path = re.sub("\.jsonl$","_f1_above.csv",hp.input_path)
    get_f1_above(df,theta_list,f1_above_path)
    ratio_above_path = re.sub("\.jsonl$","_ratio_above.csv",hp.input_path)
    get_ratio_above(df,theta_list,ratio_above_path)

    ratio_list = np.linspace(0,0.3,21)
    f1_oracle_path = re.sub("\.jsonl$","_f1_oracle.csv",hp.input_path)
    get_f1_with_oracle(df,ratio_list,f1_oracle_path)