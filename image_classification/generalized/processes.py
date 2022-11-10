import os
import yaml
import json
import torch

import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from dataloader import ImageClassificationDataloader, ImageClassificationHelper

device = "cuda" if torch.cuda.is_available() else "cpu"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

transform_train = transforms.Compose(
    [   
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

with open('labels.json', 'r') as f:
    labels_dict = json.load(f)

def getdatasetstate(args):
    img_cls_obj = ImageClassificationHelper(args["DATA_DIR"], labels_dict)
    return {k: k for k in range(len(img_cls_obj))}

def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = args["lr"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    wtd = args["wtd"]
    if not os.path.isdir(args['LOG_DIR']):
        os.makedirs(args['LOG_DIR'])

    img_cls_obj = ImageClassificationHelper(args["DATA_DIR"], labels_dict)
    datamap_df = img_cls_obj.create_datamap()
    trainset = ImageClassificationDataloader(datamap_df, labeled=labeled, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    predictions, targets = [], []
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(labels_dict)).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wtd)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90, 95], last_epoch=-1
    )

    if resume_from is not None and not args["weightsclear"]:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    net.train()
    for epoch in tqdm(range(epochs), desc="Training Epoch"):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Steps'):
            images, labels = data
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        lr_scheduler.step()

    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

def test(args, ckpt_file):
    batch_size = args["batch_size"]
    img_cls_obj = ImageClassificationHelper(args["DATA_DIR"], labels_dict)
    datamap_df = img_cls_obj.create_datamap(split='test')
    testset = ImageClassificationDataloader(datamap_df, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(labels_dict)).to(device)

    predictions, targets = [], []
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"predictions": predictions, "labels": targets}

def infer(args, unlabeled, ckpt_file):
    batch_size = args["batch_size"]
    img_cls_obj = ImageClassificationHelper(args["DATA_DIR"], labels_dict)
    datamap_df = img_cls_obj.create_datamap()
    unlabeled_set = ImageClassificationDataloader(datamap_df, labeled=unlabeled, transform=transform_test)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(labels_dict)).to(device)

    correct, total, k = 0, 0, 0
    outputs_fin = {}
    for data in tqdm(unlabeled_loader, desc="Inferring"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images).data

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(len(outputs)):
            outputs_fin[k] = {}
            outputs_fin[k]["prediction"] = predicted[j].item()
            outputs_fin[k]["pre_softmax"] = outputs[j].cpu().numpy().tolist()
            k += 1

    return {"outputs": outputs_fin}

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)

    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(args, ckpt_file=ckpt_file)
    infer(args, list(range(1000, 1500)), ckpt_file=ckpt_file)