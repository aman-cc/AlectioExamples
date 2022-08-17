from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate
import yaml
import os

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token='1e1df19a7cf84581ac0caa8e9659ab9c'
)

if __name__ == "__main__":
    AlectioPipeline()   