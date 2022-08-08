import os
import torch
import sys
import yaml
import torch
import argparse

# Add the hedwig repo to syspath
cwd = os.path.dirname(os.path.abspath(__file__))
syspath = os.path.join(cwd, "reuters", "hedwig")
sys.path.append(syspath)

# Import the Bert model
from reuters.hedwig.models.bert import __main__
from reuters.hedwig.models.bert.args import get_args


def getdatasetstate(config_args={}):
    return {k: k for k in range(config_args["trainsize"])}


def train(config_args, labeled, resume_from: int = 0, ckpt_file: str = ""):
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.labeled = labeled
    args.snapshot_path = os.path.join(cwd, args.WEIGHTS_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")
    args.infer = False

    if not os.path.isdir(os.path.join(cwd, args.WEIGHTS_DIR)):
        os.mkdir(os.path.join(cwd, args.WEIGHTS_DIR))

    __main__.main(args)
    return


def test(config_args, ckpt_file):
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.split = "dev"
    args.infer = True
    args.trained_model = os.path.join(cwd, args.WEIGHTS_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")

    predictions, labels = __main__.main(args)

    return {"predictions": predictions, "labels": labels}


def infer(config_args, unlabeled, ckpt_file):
    args = get_args()
    # Add the args from the config file
    args.__dict__.update(config_args)
    args.split = "train"
    args.infer = True
    args.trained_model = os.path.join(cwd, args.WEIGHTS_DIR, ckpt_file)
    args.data_dir = os.path.join(syspath, "hedwig-data", "datasets")

    pred, target, pre_softmax = __main__.main(args)

    outputs = {}
    for l in unlabeled:
        outputs[l] = {}
        # TODO: Check whether 0th position is needed for classification.
        outputs[l]['pre_softmax'] = pre_softmax[l][0]
        outputs[l]['prediction'] = pred[l][0]

    return {"outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(cwd, "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)
    labeled = list(range(100))
    resume_from = None
    ckpt_file = "reuters-weight.pt"

    print("Running Training")
    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Running Testing")
    test(args, ckpt_file=ckpt_file)
    print("Running Inference")
    print(infer(args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file))
