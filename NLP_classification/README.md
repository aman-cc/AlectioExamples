# Topic Classification on Reuters-21578 News Dataset
Source: https://github.com/castorini/hedwig

## 1. Download Dataset
```
cd reuters/hedwig
git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

## 2. Activate virtualenv and install requirements
(Tested in python-3.8)
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
- Number of training samples = 4659
