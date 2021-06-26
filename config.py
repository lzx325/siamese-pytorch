import sys
import yaml
from pprint import pprint
import argparse

def parse_args():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.set_defaults(func=None)
    parser.add_argument("mode",type=str)
    parser.add_argument("--config",type=str)
    args=parser.parse_args()
    json_fp=args.config
    with open(json_fp) as f:
        d=yaml.safe_load(f)
    args.__dict__.update(d)
    return args
