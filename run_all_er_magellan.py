import os
import argparse

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Beer
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy
Textual/Company""".split('\n')

special_datasets = {
    'Structured/Beer': (32, 40),
    'Structured/iTunes-Amazon': (32, 40),
    'Structured/Fodors-Zagats': (32, 40),
    'Dirty/iTunes-Amazon': (32, 40),
    'Textual/Company': (32, 15)
}

ops = """swap
swap
append_col
del
swap
drop_col
swap
swap
append_col
drop_col
drop_col
swap
del""".split('\n')


lms = ['roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta',
       'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'bert']


# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')

def train_all_er_magellan(hp):
    for dataset, op, lm in zip(datasets, ops, lms):
        if hp.task != "All" and dataset not in hp.task:
            continue

        if dataset in special_datasets:
                batch_size, epochs = special_datasets[dataset]
        else:
            batch_size, epochs = 32, 15

        if not hp.batch_size is None:
            batch_size = hp.batch_size

        if not hp.epochs is None:
            epochs = hp.epochs

        for da in [True]:
            for dk in [True]:
                for run_id in range(hp.repeat):
                    cmd = """CUDA_VISIBLE_DEVICES=0 python train_ditto.py \
                  --task %s \
                  --logdir results_ditto/ \
                  --finetuning \
                  --batch_size %d \
                  --lr 3e-5 \
                  --fp16 \
                  --save_model \
                  --lm %s \
                  --n_epochs %d \
                  --run_id %d""" % (dataset, batch_size, lm, epochs, run_id+hp.start_id)
                    # if 'Company' in dataset:
                    #     cmd += ' --summarize'
                    if hp.su:
                        cmd += " --summarize"

                    if da:
                        cmd += ' --da %s' % op
                    if dk:
                        cmd += ' --dk general'
                    print(cmd)
                    os.system(cmd)

def match_all_er_magellan(hp):
    for dataset, op, lm in zip(datasets, ops, lms):
        if hp.task != "All" and dataset not in hp.task:
            continue

        for da in [True]:
            for dk in [True]:
                for run_id in range(hp.repeat):
                    cmd = """CUDA_VISIBLE_DEVICES=0 python matcher.py \
                          --task %s \
                          --lm %s \
                          --use_gpu \
                          --fp16 \
                          --run_id %d""" % (dataset, lm, run_id+hp.start_id)

                    # if 'Company' in dataset:
                    #     cmd += ' --summarize'
                    if hp.su:
                        cmd += ' --summarize'
                    if da:
                        cmd += ' --da %s' % op
                    if dk:
                        cmd += ' --dk general'

                    if hp.calibration is not None:
                        cmd += ' --calibration %s' % hp.calibration
                        if hp.calibration == "ensemble":
                            cmd += ' --ensemble 5'

                    print(cmd)
                    os.system(cmd)


def test_all_er_magellan(hp):
    for dataset in datasets:
        if hp.task != "All" and dataset not in hp.task:
            continue
        for run_id in range(hp.repeat):
            output_prefix = "output/er_magellan/" + dataset + "/"
            if hp.calibration == "ensemble":
                ensemble = 5
            else:
                ensemble = 1
            output_file_name = "output_cal=%s_en=%d_id=%d.jsonl" % (hp.calibration,ensemble,run_id + hp.start_id)
            output_file_name = output_prefix + output_file_name
            cmd = "python tester.py --input_path %s" % output_file_name
            # print(cmd)
            os.system(cmd)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--task", type=str, default="All")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs",type=int, default=None)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--su",dest="su", action="store_true")
    hp = parser.parse_args()
    if hp.type == "train":
        train_all_er_magellan(hp)
    elif hp.type == "match":
        match_all_er_magellan(hp)
    elif hp.type == "test":
        test_all_er_magellan(hp)
