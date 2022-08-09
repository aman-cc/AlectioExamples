import argparse
import yaml, json
import os
from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate
import logging

cwd = os.path.dirname(os.path.abspath(__file__))

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

# put the train/test/infer processes into the constructor
App = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token='5206e073dd9f4851afbae0481cc71a3e'
)

if __name__ == "__main__":
    # payload = json.load(open(args["sample_payload"], "r"))
    # app._one_loop(args=args, payload=payload)
    App()