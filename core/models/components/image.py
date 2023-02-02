# Project:
#   Localized Questions in VQA
# Description:
#   Image embedding script
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet152_Weights

def get_visual_feature_extractor(config):
    if 'resnet' in config['visual_extractor']:
        model  = ResNetExtractor(config['imagenet_weights'])
    else: 
        raise ValueError("Unknown model for visual feature extraction")
    return model

class ResNetExtractor(nn.Module):
    def __init__(self, imagenet):
        super().__init__()
        self.pre_trained = imagenet
        if self.pre_trained:
            self.net_base = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            self.net_base = models.resnet152(weights=ResNet152_Weights.NONE)
        modules = list(self.net_base.children())[:-2] # ignore avgpool layer and classifier
        self.extractor = nn.Sequential(*modules)
        # freeze weights
        for p in self.extractor.parameters():
            p.requires_grad = False 

    def forward(self, x):
        x = self.extractor(x) # [B, 2048, 14, 14] if input is [B, 3, 448, 448]
        return x