import argparse
import yaml
import json
# from alectio_sdk.flask_wrapper import Pipeline
from alectio_sdk.sdk import Pipeline
from processes_2 import train, test, infer, getdatasetstate

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config.yaml", required=True)
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
    token='6c2369bd4c4845a79a3dc560153dd4d4'
)

if __name__ == "__main__":
    # payload = json.load(open(args["sample_payload"], "r"))
    # app._one_loop(args=args, payload=payload)
    # app(debug=True)
    App()
