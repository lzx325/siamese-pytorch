import os
import sys
from os.path import join,basename,dirname,splitext
from pathlib import Path
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import h5py
import json
from pprint import pprint
import argparse
from collections import defaultdict
from itertools import permutations
np.random.seed(0)
if __name__=="__main__":
    data_dir="omniglot/python/images_evaluation"
    data_list=list()
    for fp in Path(data_dir).rglob("*.png"):
        for rot in [0,90,180,270]:
            data_list.append((
                str(fp),
                fp.parts[-3],
                fp.parts[-2],
                rot
            ))
    data_df=pd.DataFrame(data_list)
    
    data_df.columns=["image_fp","language","character","rotation"]
    data_df=data_df.sort_values(["language","character","rotation"]).reset_index(drop=True)
    print(data_df)
    pairs_list=list()
    for i in range(0,len(data_df),20):
        print(f"{i} {len(data_df)}")
        current_character_indices=pd.Index(range(i,i+20))
        other_character_indices=data_df.index.difference(current_character_indices)
        negative_pairs=np.array(list(permutations(current_character_indices,2)))
        negative_pairs=np.concatenate([negative_pairs,np.zeros((len(negative_pairs),1),dtype=np.int64)],axis=1)
        positive_pairs=np.stack([np.repeat(np.array(current_character_indices),19),np.random.choice(other_character_indices,20*19,replace=True)],axis=1)
        positive_pairs=np.concatenate([positive_pairs,np.ones((len(positive_pairs),1),dtype=np.int64)],axis=1)
        pairs_list.append(negative_pairs)
        pairs_list.append(positive_pairs)
    pairs=pd.DataFrame(np.concatenate(pairs_list,axis=0))
    pairs.columns=["index1","index2","label"]
    data_df.to_csv("./omniglot-summary.csv",sep=",",index=None)
    pairs.to_csv("./omniglot-pairs.csv",sep=",",index=None)