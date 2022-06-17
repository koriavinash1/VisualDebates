import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from src.Classifier import model

model_urls = {
    'stl10': '/vol/biomedic2/agk21/PhDLogs/codes/pytorch-playground/stl10/log/default/best.pth',
    'mnist': '/vol/biomedic2/agk21/PhDLogs/codes/GLANCE-Explanations/Classifier/Logs/MNIST/best.pth',
    'afhq': '/vol/biomedic2/agk21/PhDLogs/codes/GLANCE-Explanations/Classifier/Logs/AFHQ/best.pth',
    'shapes': '/vol/biomedic2/agk21/PhDLogs/codes/GLANCE-Explanations/Classifier/Logs/SHAPES/best.pth'
}

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def mnist(n_channel, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=32*n_channel, num_classes=10)
    if pretrained is not None:
        m = torch.load(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
        print ("MNIST weights loaded ............")
    return model


def shapes(n_channel=16, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=32*n_channel, num_classes=4)
    if pretrained is not None:
        m = torch.load(model_urls['shapes'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
        print ("SHAPE weights loaded ............")
    return model


def stl10(n_channel, pretrained=None):
    cfg = [
        n_channel, 'M',
        2*n_channel, 'M',
        4*n_channel, 'M',
        4*n_channel, 'M',
        (8*n_channel, 0), (8*n_channel, 0), 'M'
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=32*n_channel, num_classes=10)
    if pretrained is not None:
        m = torch.load(model_urls['stl10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)

        print ("STL10 weights loaded ............")
    return model


def afhq(n_channel, pretrained=None):
    nclass = 3
    net = model.DenseNet121(num_channel=n_channel, classCount=nclass)
    if pretrained is not None:
        m = torch.load(model_urls['afhq'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)

        print ("AFHQ weights loaded ............")
    return net