import argparse
import sys
import random
from ditto.augment import Augmenter
import json
import numpy as np
import os


def read_classification_file(path):
    """Read a train/eval classification dataset from file

    Args:
        path (str): the path to the dataset file

    Returns:
        list of str: the input sequences
        list of str: the labels
    """
    sents, labels = [], []
    for line in open(path,encoding='UTF-8'):
        items = line.strip().split('\t')

        # assert length
        assert len(items) <= 3, "Found examples with >3 tab-separated items"

        # only consider sentence and sentence pairs
        if len(items) < 2 or len(items) > 3:
            continue
        try:
            if len(items) == 2:
                sents.append(items[0])
                labels.append(items[1])
            else:
                sents.append(items[0] + ' [SEP] ' + items[1])
                labels.append(items[2])
        except:
            print('error @', line.strip())
    return sents, labels


def store_to_classification_file(path, sents, labels):
    fo = open(path, "w",encoding="UTF-8")
    for sent,label in zip(sents,labels):
        left,right = sent.split(" [SEP] ")
        s = left + "\t" + right + "\t" + label + "\n"
        fo.write(s)

    fo.close()


def build_ood_dataset(input_path, output_path,
                      ood_path = None,
                      ood_ratio = 0.0,
                      da = None,
                      da_ratio = 0.0):
    """
    Read a dataset file from input_path, mix it with a ood-dataset file
    # from ood_path with ood_ratio, augment the data, and output the ood-dataset to
    # output_path
    :param input_path: path to read input
    :param output_path: path to store output
    :param ood_path: out-of-distribution data
    :param ood_ratio: rate to sample out-of-distribution data
    :param da: data augmentation method used
    :param da_ratio: rate to add data augmentation
    :return: None
    """
    sents,labels = read_classification_file(input_path)
    if ood_path is not None:
        ood_sents, ood_labels = read_classification_file(ood_path)
        ood_length = len(ood_sents)

    ag = Augmenter()
    output_sents = []
    output_labels = []
    for sent, label in zip(sents,labels):
        # print(sent,label)
        if ood_path is not None and random.random() < ood_ratio:
            # sample a line from ood_path instead
            ood_id = random.randint(0,ood_length-1)
            sent = ood_sents[ood_id]
            label = ood_labels[ood_id]

        if da is not None and random.random() < da_ratio:
            sent = ag.augment_sent(sent, op=da)

        # print("after processing:")
        # print(sent,label)
        output_sents.append(sent)
        output_labels.append(label)

    store_to_classification_file(output_path,output_sents,output_labels)
    return


def get_config(task):
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    return config


def get_ood_input_path(task,ood_task=None,ood_ratio=.0,da=None,da_ratio=.0):
    config = get_config(task)
    input_path = config["testset"]
    ood_input_path = "%s_ood=%s_%.1f_da=%s_%.1f.txt" % (input_path[:-4],ood_task.replace("/","_"),ood_ratio,da,da_ratio)
    # if ood_task is not None:
    #     if da is not None:
    #         ood_input_path = "%s_ood=%s_%.2f_da=%s_%.2f.txt" % (input_path[:-4],ood_task,ood_ratio,da,da_ratio)
    #     else:
    #         ood_input_path = "%s_ood=%s_%.2f.txt" % (input_path[:-4],ood_task,ood_ratio)
    # else:
    #     if da is not None:
    #         ood_input_path = "%s_da=%s_%.2f.txt" % (input_path[:-4],da,da_ratio)
    #     else:
    #         ood_input_path = input_path
    return ood_input_path





def build_all_ood_dataset(task, ood_task, ood_ratios, da, da_ratios):
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    input_path = config["testset"]
    if ood_task is not None:
        ood_config = configs[ood_task]
        ood_path = ood_config["testset"]
    else:
        ood_path = None
        ood_ratios = [.0]

    if da is None:
        da_ratios = [.0]

    for ood_ratio in ood_ratios:
        for da_ratio in da_ratios:
            ood_input_path = get_ood_input_path(task,ood_task,ood_ratio,da,da_ratio)
            build_ood_dataset(input_path,ood_input_path,ood_path,ood_ratio,da,da_ratio)
    # if ood_task is not None and da is not None:
    #     ood_task = ood_task.replace("/","_")
    #     for ood_ratio in ood_ratios:
    #         for da_ratio in da_ratios:
    #             output_path = "%s_ood=%s_%.2f_da=%s_%.2f.txt" % (input_path[:-4],ood_task,ood_ratio,da,da_ratio)
    #             build_ood_dataset(input_path,output_path,ood_path,ood_ratio,da,da_ratio)
    #
    # elif ood_task is not None:
    #     ood_task = ood_task.replace("/","_")
    #     for ood_ratio in ood_ratios:
    #         output_path = "%s_ood=%s_%.2f.txt" % (input_path[:-4],ood_task,ood_ratio)
    #         build_ood_dataset(input_path,output_path,ood_path,ood_ratio,da,0.0)
    #
    # elif da is not None:
    #     for da_ratio in da_ratios:
    #         output_path = "%s_da=%s_%.2f.txt" % (input_path[:-4],da,da_ratio)
    #         build_ood_dataset(input_path,output_path,ood_path,0.0,da,da_ratio)


if __name__ == "__main__":
    ratio_list = np.linspace(0,1,11)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",type=str,default=None)
    parser.add_argument("--ood_task",type=str,default=None)
    parser.add_argument("--da",type=str,default=None)
    hp = parser.parse_args()
    build_all_ood_dataset(hp.task,hp.ood_task,ratio_list,hp.da,ratio_list)
