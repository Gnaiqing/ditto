import os
import platform
from build_ood_dataset import get_ood_input_path,build_ood_dataset, get_config
import json
import numpy as np
import argparse

def get_matcher_command(task,run_id=0,calibration=None,da=True,dk=True,su=True,input=None,output=None,gpu_id=0):

    params = {
        "Dirty/DBLP-ACM": ("swap","roberta"),
        "Dirty/DBLP-GoogleScholar": ("swap","roberta"),
        "Dirty/iTunes-Amazon": ("append_col","roberta"),
        "Dirty/Walmart-Amazon": ("del","roberta"),
        "Structured/Amazon-Google":("swap","roberta"),
        "Structured/Beer":("drop_col","roberta"),
        "Structured/DBLP-ACM":("swap","roberta"),
        "Structured/DBLP-GoogleScholar":("swap","roberta"),
        "Structured/Fodors-Zagats":("append_col","roberta"),
        "Structured/iTunes-Amazon":("drop_col","roberta"),
        "Structured/Walmart-Amazon":("drop_col","roberta"),
        "Textual/Abt-Buy":("swap","roberta"),
        "Textual/Company":("del","bert")
    }

    try:
        op,lm = params[task]
    except ValueError:
        print("task %s not exist!" % task)
        return ""

    command = ""
    if platform.system().lower() == "linux":
        command += "CUDA_VISIBLE_DEVICES=%d " % gpu_id
    command += """python matcher.py \
    --task %s --lm %s --use_gpu --fp16 --run_id %d """ % (task,lm,run_id)

    if su:
        command += " --summarize"
    if da:
        command += " --da %s" % op
    if dk:
        command += " --dk general"

    if calibration is not None:
        command += " --calibration %s" % calibration

    if input is not None:
        command += " --input %s" % input

    if output is not None:
        command += " --output %s" % output

    return command


def get_ood_output_path(task,ood_task,ood_ratio,da,da_ratio,calibration,id):
    config = get_config(task)
    l = ["output"] + config["testset"].split("/")[1:-1]
    ood_output_path = "/".join(l)
    if not os.path.exists(ood_output_path):
        os.makedirs(ood_output_path)

    output_tag = "%s_ood=%s_%.1f_da=%s_%.1f_cal=%s_id=%d.jsonl" % \
                 (task.replace("/","_"),ood_task.replace("/","_"),ood_ratio,da,da_ratio,calibration,id)
    ood_output_path = ood_output_path + "/" + output_tag
    return ood_output_path


def run_matcher_on_ood_dataset(task,ood_task,da,calibration=None,run_id=0):
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    input_path = config["testset"]
    if ood_task is not None:
        ood_ratio_list = np.linspace(0,1,11)
    else:
        ood_ratio_list = [.0]

    if da is not None:
        da_ratio_list = np.linspace(0,1,11)
    else:
        da_ratio_list = [.0]

    for ood_ratio in ood_ratio_list:
        for da_ratio in da_ratio_list:
            ood_input_path = get_ood_input_path(task,ood_task,ood_ratio,da,da_ratio)
            if not os.path.exists(ood_input_path):
                # build ood input first
                ood_config = get_config(ood_task)
                ood_path = ood_config["testset"]
                build_ood_dataset(input_path,ood_input_path,ood_path,ood_ratio,da,da_ratio)
            ood_output_path = get_ood_output_path(task,ood_task,ood_ratio,da,da_ratio,calibration,run_id)
            print("OOD input path:",ood_input_path)
            print("OOD output path:",ood_output_path)
            cmd = get_matcher_command(task,run_id,calibration=calibration,
                                      input=ood_input_path,output=ood_output_path)
            print(cmd)
            os.system(cmd)
            test_cmd = "python tester.py --input_path "+ood_output_path
            print(test_cmd)
            os.system(test_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",type=str,default=None)
    parser.add_argument("--ood_task",type=str,default=None)
    parser.add_argument("--da",type=str,default=None)
    parser.add_argument("--calibration",type=str,default=None)
    parser.add_argument("--run_id",type=int,default=0)
    hp = parser.parse_args()
    run_matcher_on_ood_dataset(hp.task,hp.ood_task,hp.da,hp.calibration,hp.run_id)
