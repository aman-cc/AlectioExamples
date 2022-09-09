## Mushroom Classification

> A task where you'll predict the toxicity (posionous or edible) of mushrooms in the wild using 23 numerical features describing quantitative and qualitative aspects of the mushrooms.

## 1. Download dataset
i) Create a data directory and a log directory.
```
mkdir data
mkdir log
```
ii) Download the .csv data from kaggle from this [link](https://www.kaggle.com/uciml/mushroom-classification). Place the csv data in the `./data` folder you created in step #1. Rename the CSV to `mushrooms.csv`

iii) Modify hyper-parameters in the `config.yaml` file as needed (to set the train_epochs, batch_size, learning rate, momentum, etc)

## 2. Activate virtualenv and install requirements
(Tested in python-3.9)
```
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install alectio-sdk
pip install -r requirements.txt
```

## 4. Start training with Alectio SDK
- Place token inside main.py
- Run `python main.py`

## Misc
- Total number of records = 8124
- Number of training records = 6499