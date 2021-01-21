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
    content = ''
    # if the entry is already serialized
    if isinstance(row, str):
        content = row
    else:
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


def predict_noprint(input_path, config, model,
            batch_size=1024,
            summarizer=None,
            lm='distilbert',
            max_len=256,
            dk_injector=None):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        config (Dictionary): the task configuration
        model (SnippextModel): the model for prediction
        batch_size (int): the batch size
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        output_rows: list containing pair description
        output_predictions: list containing prediction results
        output_logits: list containing logits
    """
    pairs = []

    def process_batch(rows, pairs, writer):
        try:
            predictions, logits = classify(pairs, config, model, lm=lm, max_len=max_len)
        except:
            # ignore the whole batch
            return [],[],[]

        return rows, predictions, logits

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    with jsonlines.open(input_path) as reader:
        pairs = []
        rows = []
        output_rows = []
        output_predictions = []
        output_logits = []

        for idx, row in tqdm(enumerate(reader)):
            pairs.append((to_str(row[0], summarizer, max_len, dk_injector),
                          to_str(row[1], summarizer, max_len, dk_injector)))
            rows.append(row)
            if len(pairs) == batch_size:
                t_rows, t_predictions, t_logits = process_batch(rows, pairs, writer)
                output_rows += t_rows
                output_predictions += t_predictions
                output_logits += t_logits
                pairs.clear()
                rows.clear()

        if len(pairs) > 0:
            t_rows, t_predictions, t_logits = process_batch(rows, pairs, writer)
            output_rows += t_rows
            output_predictions += t_predictions
            output_logits += t_logits

        return output_rows, output_predictions, output_logits


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
    parser.add_argument("--calibration", type=str,default=None)
    parser.add_argument("--ensemble", type=int,default=5)
    hp = parser.parse_args()


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
        checkpoint = os.path.join(hp.checkpoint_path, '%s_dev.pt' % run_tag)

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
    if hp.calibration == "conformal":
        vocab = config['vocab']
        validset = config['validset']
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


    # update input & output paths
    if hp.input_path is None:
        hp.input_path = config["testset"]

    if hp.output_path is None:
        l = ["output"] + config["testset"].split("/")[1:-1]
        hp.output_path = "/".join(l)
        if not os.path.exists(hp.output_path):
            os.makedirs(hp.output_path)

        output_tag = "output_cal=%s_en=%d.jsonl" % (hp.calibration, hp.ensemble)
        hp.output_path = hp.output_path+"/" + output_tag

    print("Input path: ",hp.input_path)
    print("Output path: ", hp.output_path)

    # run prediction
    if hp.calibration == "ensemble":
        for i in range(hp.ensemble):
            output_rows, output_predictions, output_logits = predict_noprint(
                                        hp.input_path,config, models[i],
                                        summarizer=summarizer, lm=hp.lm,
                                        max_len=hp.max_len, dk_injector=dk_injector)
            if i == 0:
                output_scores = np.array(softmax(output_logits, axis=1))
            else:
                output_scores += np.array(softmax(output_logits, axis=1))
        output_scores = (output_scores / hp.ensemble).astype(str).tolist()
        output_pred = output_scores.argmax(axis=1).tolist()

    # print output
    with jsonlines.open(hp.output_path, mode='w') as writer:
        for row, pred, score in zip(output_rows, output_pred, output_scores):
            output = {'left': row[0],
                      'right': row[1],
                      'match': pred,
                      'match_confidence': score[int(pred)]}
            writer.write(output)












