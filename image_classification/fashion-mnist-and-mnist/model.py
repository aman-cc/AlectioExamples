import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50, resnet101, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=1, padding=0)

        self.conv2 = nn.Conv2d(3, 6, kernel_size=(4, 4), stride=1, padding=0)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.fullCon1 = nn.Linear(in_features=6 * 11 * 11, out_features=360)

        self.fullCon2 = nn.Linear(in_features=360, out_features=100)

        self.fullCon3 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 11 * 11)
        x = F.relu(self.fullCon1(x))
        x = F.relu(self.fullCon2(x))
        x = self.fullCon3(x)
        return x

def get_model(backbone='resnet18', num_classes=2, use_dropout=False):
    if backbone == 'custom':
        return NeuralNet()
    elif backbone == 'resnet18':
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
    model = get_model('custom', 5)