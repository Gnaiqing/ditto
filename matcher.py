import torch
import torch.nn as nn
import os
import numpy as np
import random
import json
import jsonlines
import csv
import re
import time
import argparse
import sys
import traceback

from torch.utils import data
from tqdm import tqdm
from apex import amp
from scipy.special import softmax

sys.path.insert(0, "Snippext_public")
from snippext.model import MultiTaskNet
from ditto.exceptions import ModelNotFoundError
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *

from torch.nn import CrossEntropyLoss


def to_str(row, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a data entry

    Args:
        row (Dictionary): the data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """

    # if the entry is already serialized
    if isinstance(row, str):
        return row

    content = ''
    for attr in row.keys():
        content += 'COL %s VAL %s ' % (attr, row[attr])

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    if dk_injector is not None:
        content = dk_injector.transform(content)

    return content


def classify(sentence_pairs, config, model, lm='distilbert', max_len=256):
    """Apply the MRPC model.

    Args:
        sentence_pairs (list of tuples of str): the sentence pairs
        config (dict): the model configuration
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length

    Returns:
        list of float: the scores of the pairs
    """
    inputs = []
    for (sentA, sentB) in sentence_pairs:
        inputs.append(sentA + '\t' + sentB)

    dataset = DittoDataset(inputs, config['vocab'], config['name'], lm=lm, max_len=max_len)
    iterator = data.DataLoader(dataset=dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=DittoDataset.pad)

    # prediction
    Y_logits = []
    Y_hat = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            logits, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_logits += logits.cpu().numpy().tolist()
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    results = []
    for i in range(len(inputs)):
        pred = dataset.idx2tag[Y_hat[i]]
        results.append(pred)

    return results, Y_logits


def predict(input_path, output_path, config, model,
            batch_size=1024,
            summarizer=None,
            lm='distilbert',
            max_len=256,
            dk_injector=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        output_path (str): the output file path
        config (Dictionary): the task configuration
        model (SnippextModel): the model for prediction
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        None
    """
    pairs = []

    def process_batch(rows, pairs, writer):
        try:
            predictions, logits = classify(pairs, config, model, lm=lm, max_len=max_len)
        except:
            # ignore the whole batch
            return
        scores = softmax(logits, axis=1)
        for row, pred, score in zip(rows, predictions, scores):
            output = {'left': row[0], 'right': row[1],
                'match': pred,
                'match_confidence': score[int(pred)]}
            writer.write(output)

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    start_time = time.time()
    with jsonlines.open(input_path) as reader,\
         jsonlines.open(output_path, mode='w') as writer:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append((to_str(row[0], summarizer, max_len, dk_injector),
                          to_str(row[1], summarizer, max_len, dk_injector)))
            rows.append(row)
            if len(pairs) == batch_size:
                process_batch(rows, pairs, writer)
                pairs.clear()
                rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, writer)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))


def classify_from_iterator(model, iterator, use_gpu):
    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    Y_logits = []
    Y_hat = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            x = x.to(device)
            y = y.to(device)
            logits, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_logits += logits.cpu().numpy().tolist()
            Y_hat += y_hat.cpu().numpy().tolist()

    return Y_hat, Y_logits


def get_label(iterator):
    # get the list of label from iterator
    Y = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            Y += y.numpy().tolist()

    return Y


def load_model(task, checkpoint, lm, use_gpu, fp16=True):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models

    # checkpoint = os.path.join(path, '%s.pt' % task)
    if not os.path.exists(checkpoint):
        raise ModelNotFoundError(checkpoint)

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    config = configs[task]
    config_list = [config]
    model = MultiTaskNet([config], device, True, lm=lm)

    saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)


    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model


def compute_e_value(score, u, y):
    # compute E score for conformal calibration
    p = score[y]    # probability of the true label
    q = 1-p  # probability of the false label
    if p >= q:
        return p*u
    else:
        return q + p*u


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Beer')
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='model/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--run_id", type=int, default=0)
    # calibration: select which predictive uncertainty estimation method is used
    parser.add_argument("--calibration", type=str,default=None)
    # ensemble: select how many models are used in ensemble. Valid only if calibration=="ensemble"
    parser.add_argument("--ensemble", type=int,default=5)
    # validset: path to validation set. When set to None, default valid set is used
    parser.add_argument("--validset", type=str,default=None)
    hp = parser.parse_args()

    start_time = time.time()
    # load the models
    models = []
    config = None
    if hp.calibration != "ensemble":
        hp.ensemble = 1

    for i in range(hp.ensemble):
        # the id of models used are run_id, run_id+1, ..., run_id+ensemble-1
        run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (hp.task, hp.lm, hp.da,
                                                                hp.dk, hp.summarize, str(hp.size), hp.run_id+i)
        run_tag = run_tag.replace('/', '_')
        checkpoint = os.path.join(hp.checkpoint_path, '%s_test.pt' % run_tag)
        if not os.path.exists(checkpoint):
            print("File %s not found. Abort." % checkpoint)
            exit()

        config, model = load_model(hp.task, checkpoint,
                                   hp.lm, hp.use_gpu, hp.fp16)
        models.append(model)

    summarizer = dk_injector = None
    if hp.summarize:
        summarizer = Summarizer(config, hp.lm)

    if hp.dk is not None:
        if 'product' in hp.dk:
            dk_injector = ProductDKInjector(config, hp.dk)
        else:
            dk_injector = GeneralDKInjector(config, hp.dk)

    # load validation set for calibration
    if hp.calibration == "conformal" or hp.calibration == "temperature":
        vocab = config['vocab']
        if hp.validset is None:
            validset = config['validset']
        else:
            validset = hp.validset

        if hp.summarize:
            validset = summarizer.transform_file(validset, max_len=hp.max_len)
        if hp.dk is not None:
            validset = dk_injector.transform_file(validset)
        valid_dataset = DittoDataset(validset,vocab,hp.task,lm=hp.lm)
        valid_iterator = data.DataLoader(dataset=valid_dataset,
                                         batch_size=64,
                                         shuffle=False,
                                         num_workers=0,
                                         collate_fn=DittoDataset.pad)
        # get prediction result for validation set
        Y_hat,Y_logits = classify_from_iterator(models[0],valid_iterator,hp.use_gpu)
        Y = get_label(valid_iterator)

        if hp.calibration == "temperature":
            temp_list = np.logspace(-1,1,100)
            min_nll = 1e3
            best_temp = 1
            loss = CrossEntropyLoss()
            Y = torch.Tensor(Y).type(torch.long)
            Y_logits = torch.Tensor(Y_logits)

            for temp in temp_list:
                Y_logits_temp = Y_logits * temp
                nll = float(loss(Y_logits_temp, Y))
                if nll < min_nll:
                    min_nll = nll
                    best_temp = temp

        elif hp.calibration == "conformal":
            scores = softmax(Y_logits, axis=1)
            # compute E values for conformal calibration
            E = []
            for i in range(len(Y)):
                u = random.uniform(0,1)
                e = compute_e_value(scores[i],u,Y[i])
                E.append(e)
            E = np.array(E)

    # update input & output paths
    if hp.input_path is None:
        hp.input_path = config["testset"]

    if hp.output_path is None:
        l = ["output"] + config["testset"].split("/")[1:-1]
        hp.output_path = "/".join(l)
        if not os.path.exists(hp.output_path):
            os.makedirs(hp.output_path)

        output_tag = "output_cal=%s_en=%d_id=%d.jsonl" % (hp.calibration, hp.ensemble,hp.run_id)
        hp.output_path = hp.output_path+"/" + output_tag

    print("Input path: ",hp.input_path)
    print("Output path: ", hp.output_path)

    # load data from input path
    testset = hp.input_path
    original_test_dataset = DittoDataset(testset,config["vocab"],hp.task,lm=hp.lm)
    output_rows = []
    output_labels = []

    test_size = len(original_test_dataset)
    for i in range(test_size):
        words, x, is_heads, tags, mask, y, seqlens, taskname = original_test_dataset[i]
        row = words.split(' [SEP] ')
        output_rows.append(row)
        output_labels.append(y)


    if hp.summarize:
        testset = summarizer.transform_file(testset, max_len=hp.max_len)
    if hp.dk is not None:
        testset = dk_injector.transform_file(testset)
    test_dataset = DittoDataset(testset,config["vocab"],hp.task,lm=hp.lm)
    test_iterator = data.DataLoader(dataset=test_dataset,
                                     batch_size=64,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=DittoDataset.pad)


    # run prediction
    if hp.calibration == "ensemble":
        for i in range(hp.ensemble):
            output_pred, output_logits = classify_from_iterator(models[i],test_iterator,hp.use_gpu)
            if i == 0:
                output_scores = np.array(softmax(output_logits, axis=1))
            else:
                output_scores += np.array(softmax(output_logits, axis=1))

        # average the prediction of ensemble models to get final result
        output_scores = output_scores / hp.ensemble
        idx2tag = {idx: tag for idx, tag in enumerate(config['vocab'])}
        output_pred = []
        for score in output_scores:
            pred = score.argmax()
            output_pred.append(idx2tag[pred])

        output_conf = output_scores.max(axis=1).tolist()

    elif hp.calibration == "conformal":
        output_pred, output_logits = classify_from_iterator(models[0],test_iterator,hp.use_gpu)
        output_scores = np.array(softmax(output_logits, axis=1))
        output_conf = output_scores.max(axis=1).tolist()
        # calibration on confidence
        for i in range(len(output_conf)):
            # calculate the quantile of output_conf[i]
            conf = output_conf[i]
            calibrated_conf = sum(E < conf) / len(E)
            # use the quantile to replace the original confidence
            if calibrated_conf < 0.5:
                output_conf[i] = 0.5
            else:
                output_conf[i] = calibrated_conf

    elif hp.calibration == "temperature":
        output_pred, output_logits = classify_from_iterator(models[0],test_iterator,hp.use_gpu)
        output_logits = (np.array(output_logits)*best_temp).tolist()
        output_scores = softmax(output_logits,axis=1)
        output_conf = output_scores.max(axis=1).tolist()

    else: # no calibration
        output_pred, output_logits = classify_from_iterator(models[0],test_iterator,hp.use_gpu)
        output_scores = np.array(softmax(output_logits, axis=1))
        output_conf = output_scores.max(axis=1).tolist()

    # print output
    with jsonlines.open(hp.output_path, mode='w') as writer:
        for row, pred, conf, label in zip(output_rows, output_pred, output_conf, output_labels):
            output = {'left': row[0],
                      'right': row[1],
                      'match': pred,
                      'match_confidence': conf,
                      'label': label}
            writer.write(output)
        writer.close()

    run_time = time.time() - start_time
    os.system("echo %s %f >> log_match.txt" % (hp.output_path, run_time))
