import os
import time

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
    'Textual/Company': (32, 3)
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

to_trains = [False, False, False, False, False, False,
            False, False, False, False, False, False, False]

to_matches = [True, True, True, True, True, False,
              True, False, False, False, False, False, False]
# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')
def train_all_er_magellan():
    for dataset, op, lm, to_train in zip(datasets, ops, lms, to_trains):
        if not to_train:
            continue
        # if dataset in special_datasets:
        #         batch_size, epochs = special_datasets[dataset]
        #     else:
        #         batch_size, epochs = 32, 15
        batch_size, epochs = 32, 1

        cmd = "set CUDA_VISIBLE_DEVICES=0"
        print(cmd)
        os.system(cmd)

        for da in [True]:
            for dk in [True]:
                for run_id in range(1):
                    cmd = """python train_ditto.py \
                  --task %s \
                  --logdir results_ditto/ \
                  --finetuning \
                  --batch_size %d \
                  --lr 3e-5 \
                  --max_len 64 \
                  --fp16 \
                  --save_model \
                  --lm %s \
                  --n_epochs %d \
                  --run_id %d""" % (dataset, batch_size, lm, epochs, run_id)
                    if 'Company' in dataset:
                        cmd += ' --summarize'
                    if da:
                        cmd += ' --da %s' % op
                    if dk:
                        cmd += ' --dk general'
                    print(cmd)
                    os.system(cmd)


def match_all_er_magellan():
    for dataset, op, lm, to_match in zip(datasets, ops, lms, to_matches):
        if not to_match:
            continue

        batch_size = 32

        cmd = "set CUDA_VISIBLE_DEVICES=0"
        print(cmd)
        os.system(cmd)

        for da in [True]:
            for dk in [True]:
                for run_id in range(1):
                    cmd = """python matcher.py \
                          --task %s \
                          --lm %s \
                          --use_gpu \
                          --fp16 \
                          --max_len 64 \
                          --run_id %d""" % (dataset, lm, run_id)
                    if 'Company' in dataset:
                        cmd += ' --summarize'
                    if da:
                        cmd += ' --da %s' % op
                    if dk:
                        cmd += ' --dk general'
                    print(cmd)
                    os.system(cmd)


match_all_er_magellan()