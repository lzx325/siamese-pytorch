import os
import sys
from os.path import join,basename,dirname,splitext
from pathlib import Path
import numpy as np
import scipy
from scipy import io
import pandas as pd
import h5py
import yaml
from pprint import pprint
import argparse
from collections import defaultdict
from itertools import product
from copy import copy

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--meta-config",type=str,required=True)
    parser.add_argument("--exp-root-dir",type=str,required=True)
    parser.add_argument("--tune-hyper-parameters",type=str,required=True)
    parser.add_argument("--n_runs",type=int)
    args=parser.parse_args()
    return args

def parse_dict_args(args,key):
    dict_args=args.__getattribute__(key)
    d=dict()
    for arg in dict_args.split(","):
        arg=arg.strip()
        k,v=arg.split(':')
        d[k]=int(v)
    return d

def parse_list_args(args,key):
    list_args=getattr(args,key)
    l=list()
    for arg in list_args.split(","):
        arg=arg.strip()
        l.append(arg)
    return l
    
def get_exp_code(hyper_names,comb):
    s=""
    for i,(n,c) in enumerate(zip(hyper_names,comb)):
        if type(c)==float:
            s=s+f"{n}={c:.1e}"
        elif type(c)==str:
            s=s+f"{n}={c}"
        elif type(c)==int:
            s=s+f"{n}={c}"
        else:
            assert False, "unknown type"
        if i!=len(comb)-1:
            s=s+"--"
    return s
if __name__=="__main__":
    args=parse_args()
    with open(args.meta_config) as f:
        metaconfig=yaml.load(f,yaml.SafeLoader)
    tune_hyper_name_list=parse_list_args(args,"tune_hyper_parameters")
    tune_hyper_list=list()
    for hyper_name in metaconfig.keys():
        if hyper_name in tune_hyper_name_list:
            assert type(metaconfig[hyper_name])==list
            tune_hyper_list.append(metaconfig[hyper_name])
        else:
            assert type(metaconfig[hyper_name])!=list, hyper_name
    all_combinations=list(product(*tune_hyper_list))
    np.random.seed(0)
    if args.n_runs is not None and len(all_combinations)>args.n_runs:
        idx=np.random.choice(len(all_combinations),args.n_runs,replace=False)
        combinations=[all_combinations[i] for i in idx]
        print(f"{len(all_combinations)} combinations in total, subsetting to {len(combinations)}")
    else:
        combinations=all_combinations
        print(f"{len(all_combinations)} combinations in total, running all")
    # write info
    run_dir=join(args.exp_root_dir,"runs")
    bookkeeping_dir=join(args.exp_root_dir,"bookkeeping")
    configs_dir=join(args.exp_root_dir,"configs")
    os.makedirs(run_dir,exist_ok=True)
    os.makedirs(bookkeeping_dir,exist_ok=True)
    os.makedirs(configs_dir,exist_ok=True)

    with open(join(bookkeeping_dir,"meta.yaml"),'w') as f:
        yaml.dump(metaconfig,f,sort_keys=False)
    with open(join(bookkeeping_dir,"args.yaml"),'w') as f:
        yaml.dump(args.__dict__,f,sort_keys=False)

    for i,comb in enumerate(combinations):
        config=copy(metaconfig)
        exp_code=get_exp_code(tune_hyper_name_list,comb)
        config["exp_code"]=exp_code
        config["train_dir"]=run_dir
        for hyper_name,hyper_value in zip(tune_hyper_name_list,comb):
            config[hyper_name]=hyper_value
        with open(join(configs_dir,f"{exp_code}.yaml"),'w') as f:
            yaml.dump(config,f,sort_keys=False)

