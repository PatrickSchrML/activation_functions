import torch
import torch.nn as nn
import os
from pau_torch.pade_activation_unit import PAU
from .vgg import VGG

__all__ = [
    'vgg11_pau', 'vgg11_bn_pau', 'vgg13_pau', 'vgg13_bn_pau', 'vgg16_pau', 'vgg16_bn_pau',
    'vgg19_bn_pau', 'vgg19_pau',
]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), PAU()]
            else:
                layers += [conv2d, PAU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, pretrained_classifier, progress, device, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained_classifier:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
        print(state_dict['classifier'])
        model.classifier.load_state_dict(state_dict.classifier)  # TODO PAU
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model


def vgg11_pau(pretrained=False, pretrained_classifier=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, pretrained_classifier, progress, **kwargs)


def vgg11_bn_pau(pretrained=False, pretrained_classifier=False, progress=True, device='cpu', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, pretrained_classifier, progress, device, **kwargs)


def vgg13_pau(pretrained=False, pretrained_classifier=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, pretrained_classifier, progress, **kwargs)


def vgg13_bn_pau(pretrained=False, pretrained_classifier=False, progress=True, device='cpu', **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, pretrained_classifier, progress, device, **kwargs)


def vgg16_pau(pretrained=False, pretrained_classifier=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, pretrained_classifier, progress, **kwargs)


def vgg16_bn_pau(pretrained=False, pretrained_classifier=False, progress=True, device='cpu', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, pretrained_classifier, progress, device, **kwargs)


def vgg19_pau(pretrained=False, pretrained_classifier=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, pretrained_classifier, progress, **kwargs)


def vgg19_bn_pau(pretrained=False, pretrained_classifier=False, progress=True, device='cpu', **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, pretrained_classifier, progress, device, **kwargs)
