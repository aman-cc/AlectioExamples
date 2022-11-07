# Ember Classification Problem
Source: https://github.com/endgameinc/ember

This example shows you how to build `train`, `test` and `infer` processes
for tabular data. The ember dataset is an open source cybersecurity dataset that is a binary classification problem.
Either examples are 'benign' or 'malicious'. You can find the full details here: https://github.com/endgameinc/ember

## 1. Activate virtualenv and install requirements
(Tested in python-3.9)
```
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install alectio-sdk
pip install git+https://github.com/elastic/ember.git
pip install -r requirements.txt
```

## 2. Download dataset
```
mkdir data
mkdir data/ember_2017_2
mkdir log
./setenv.sh
```

Download Ember dataset from [here](https://www.kaggle.com/datasets/trinhvanquynh/ember-for-static-malware-analysis).
Unzip the archive to folder `data/ember_2017_2`. Directory structure should look like this:
```
.
└── data/
    └── ember_2017_2/
        ├── X_test.dat
        ├── X_train.dat
        ├── metadata.csv
        ├── test_metadata.csv
        ├── train_metadata.csv
        ├── y_test.dat
        └── y_train.dat
```

## 3. Start training with Alectio SDK
- Place token inside main.py
- Run `python main.py`

## 4. Misc
- Number of train records: 75179
- Number of test records: 25000
