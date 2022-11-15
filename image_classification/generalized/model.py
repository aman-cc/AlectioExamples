import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

def get_model(backbone='resnet18', num_classes=2, use_dropout=False):
    if backbone == 'resnet18':
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone == 'resnet34':
        net = resnet34(weights=ResNet34_Weights.DEFAULT)
    elif backbone == 'resnet50':
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif backbone == 'resnet101':
        net = resnet101(weights=ResNet101_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    if use_dropout:
        del net.fc
        net.dropout = nn.Dropout()
    net.fc = nn.Linear(num_ftrs, num_classes)
    return net

if __name__ == '__main__':
    model = get_model('resnet34', 5)
    breakpoint()