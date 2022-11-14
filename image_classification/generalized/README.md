# Generalized Image Classification

This example shows you how to build `train`, `test` and `infer` processes
for image classification problems.

## 1. Activate virtualenv and install requirements
(Tested in python-3.9)
```
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install alectio-sdk
pip install -r requirements.txt
```

## 2. Put training data
- Arrange images in data_dir/<train/test>/<label_name>/*
```
.
└── data/
    ├── train/
    │   ├── labelname-1/
    │   │   ├── image-1.jpg
    │   │   └── image-2.jpg
    │   ├── labelname-2/
    │   │   ├── image-3.jpg
    │   │   └── image-4.jpg
    │   ├── labelname-3/
    │   │   ├── image-5.jpg
    │   │   └── image-6.jpg
    │   └── labelname-4/
    │       ├── image-7.jpg
    │       └── image-8.jpg
    └── test/
        ├── labelname-1/
        │   ├── image-1.jpg
        │   └── image-2.jpg
        ├── labelname-2/
        │   ├── image-3.jpg
        │   └── image-4.jpg
        ├── labelname-3/
        │   ├── image-5.jpg
        │   └── image-6.jpg
        └── labelname-4/
            ├── image-7.jpg
            └── image-8.jpg
```
- Put the labels in `labels.json`
```
{
  "0": "labelname-1",
  "1": "labelname-2",
  "2": "labelname-3",
  "3": "labelname-4"
}
```

## 3. Start training with Alectio SDK
- Place token inside main.py
- Run `python main.py`