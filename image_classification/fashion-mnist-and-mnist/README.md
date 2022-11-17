# Fashion MNIST Classification Task

> A task where we'll detect the fashion style of an image.

### Running steps

Note: If you want to run an MNIST task, set the DATASET argument in config.yaml to "MNIST", 
otherwise if you want to run FASHION-MNIST, set the DATASET argument in config.yaml to "Fashion". 

This example shows you how to build `train`, `test` and `infer` processes
for image classification problems. In particular, it will show you the format
of the return of `test` and `infer`. For an object detection problem, those
returns can be a little bit involved. But to get most out of Alectio's platform,
those returns needs to be correct. 

### 1. Set up a virtual environment and install Alectio SDK
(Tested in python-3.9)
```
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install alectio-sdk
pip install -r requirements.txt
```

### 2. Build Train Process
We will train a [Basic CNN based on the official PyTorch tutorial] (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for
this demo. The model is defined in `model.py`. Feel free to change it as you please. 

To try out this step, run:

```
python model.py
```

### 3. Build Test Process
The test process tests the model trained in each active learning loop.
In this example, the test process is the `test` function defined 
in `processes.py`. 

```
python processes.py
```

#### Return of the Test Process 
You will need to run non-maximum suppression on the predictions on the test images and return 
the final detections along with the ground-truth bounding boxes and objects
on each image. 

The return of the `test` function is a dictionary 
```
{"predictions": predictions, "labels": labels}
```

Both `predictions` and `labels` are lists where `labels` denotes the ground truths for the images.

### 4. Build Infer Process
The infer process is used to apply the model to the unlabeled set to run inference. 
We will use the inferred output to estimate which of those unlabeled data will
be most valuable to your model.

#### Return of the Infer Process
The return of the infer process is a dictionary
```python
{"outputs": outputs}
```

`outputs` is a dictionary whose keys are the indices of the unlabeled
images. The value of `outputs[i]` is a dictionary that records the output of
the model on training image `i`. 

### 5. Build Flask App 
First you have to set up your main.py file to contain the token that is specific to your experiment. After you
create an experiment on the Alectio platform, you will receive a unique token that will be necessary to run your experiment.

Copy and paste that token into the main.py file under the token field within the Pipeline object.
```python
app = Pipeline(
    name="FashionMNIST",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    token='<your-token-here>'
)
```
Once you have updated that file, execute the python file and you should be able to begin running your experiment.
```
python main.py
```