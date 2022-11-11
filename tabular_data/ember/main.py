import argparse
import yaml
from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config.yaml", default='config.yaml')
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
    token=''
)

if __name__ == "__main__":
    App()
