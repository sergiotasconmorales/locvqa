# Project:
#   Localized Questions in VQA
# Description:
#   Attention-related definitions
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from . import fusion
from . import utils
from torch import nn
import torch.nn.functional as F
import torch

def get_attention_mechanism(config, special=None):
    # get attention parameters
    visual_features_size = config['visual_feature_size']
    question_feature_size = config['question_feature_size']
    attention_middle_size = config['attention_middle_size']
    number_of_glimpses = config['number_of_glimpses']
    attention_fusion = config['attention_fusion']
    dropout_attention = config['attention_dropout']
    if special is None: # Normal attention mechanism
        attention = AttentionMechanismBase(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att1': # special attention mechanism 1
        attention = AttentionMechanism_1(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att2':
        attention = AttentionMechanism_2(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att3':
        attention = AttentionMechanism_3(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att4':
        attention = AttentionMechanism_4(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    elif special == 'Att5':
        if 'attenuation_factor' in config:
            attenuation_factor = config['attenuation_factor']
        else:
            print('Using default attenuation factor of 0.1')
        attention = AttentionMechanism_5(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention, attenuation_factor=attenuation_factor)
    elif special == 'Att6':
        attention = AttentionMechanism_6(visual_features_size, question_feature_size, attention_middle_size, number_of_glimpses, attention_fusion, drop=dropout_attention)
    return attention


def apply_attention(visual_features, attention):
    # visual features has size [b, m, k, k]
    # attention has size [b, glimpses, k, k]
    b, m = visual_features.size()[:2] # batch size, number of feature maps
    glimpses = attention.size(1)
    visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
    attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
    attention = F.softmax(attention, dim = -1).unsqueeze(2) # [b, glimpses, 1, k*k]
    attended = attention*visual_features # use broadcasting to weight the feature maps
    attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
    return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

class AttentionMechanismBase(nn.Module):
    """Normal attention mechanism"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(visual_features_size, attention_middle_size, 1, bias=False)
        self.lin1 = nn.Linear(question_feature_size, attention_middle_size)
        self.fuser, self.size_after_fusion = fusion.get_fuser(fusion_method, attention_middle_size, attention_middle_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.conv2 = nn.Conv2d(self.size_after_fusion, glimpses, 1)

    def forward(self, visual_features, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = apply_attention(visual_features, x)

        if return_maps:
            return x, maps
        else:
            return x


class AttentionMechanism_1(AttentionMechanismBase):
    """Attention mechanism for model VQARS_4 to include mask before softmax and then softmax only part that mask keeps"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = attention*mask #! Apply mask
        for i in range(glimpses):
            attention[:,i,:][mask.squeeze().to(torch.bool)] = F.softmax(attention[:,i,:][mask.squeeze().to(torch.bool)], dim=-1)
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)
        return x

class AttentionMechanism_2(AttentionMechanismBase):
    """Attention mechanism for model VQARS_6 = VQARS_4 to include mask before softmax and then softmax only part that mask keeps"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention[:,0,:] = attention[:,0,:]*mask.squeeze() # apply to first glimpse only
        attention[:,0,:][mask.squeeze().to(torch.bool)] = F.softmax(attention[:,0,:][mask.squeeze().to(torch.bool)], dim=-1) # again, only first glimpse
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)
        return x


class AttentionMechanism_3(AttentionMechanismBase):
    """Attention mechanism for model VQARS_7 to include mask after softmax"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = F.softmax(attention, dim = -1) # [b, glimpses, k*k]
        attention = attention*mask #! Apply mask
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)

        if return_maps:
            return x, maps
        else:
            return x


class AttentionMechanism_4(AttentionMechanismBase):
    """Attention mechanism for model VQARS_7 to include mask after softmax, but region attention is scaled by (1- (max(outside) - max(inside)))"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = F.softmax(attention, dim = -1) # [b, glimpses, k*k]
        not_mask = torch.max(mask) - mask 
        attention = (attention*mask)*(1 - torch.abs(torch.max(attention*not_mask) - torch.max(attention*mask))) #! Apply mask
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)

        if return_maps:
            return x, maps
        else:
            return x


class AttentionMechanism_5(AttentionMechanismBase):
    """Same as AttentionMechanism_3 but instead of masking, magnify inside of the region and attenuate outside"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0, attenuation_factor = 0.1):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)
        self.attenuation_factor = attenuation_factor

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = F.softmax(attention, dim = -1) # [b, glimpses, k*k]
        not_mask = torch.max(mask) - mask
        attention = attention*mask + attention*not_mask*self.attenuation_factor
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)

        if return_maps:
            return x, maps
        else:
            return x


class AttentionMechanism_6(AttentionMechanismBase):
    """Same as AttentionMechanism_3 but apply the mask to one glimpse only"""
    def __init__(self, visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=0.0):
        super().__init__(visual_features_size, question_feature_size, attention_middle_size, glimpses, fusion_method, drop=drop)

    # same as general function above but receiving mask and applying it before the softmax
    def apply_attention(self, visual_features, attention, mask):
        # visual features has size [b, m, k, k]
        # attention has size [b, glimpses, k, k]
        # mask has size [b, 1, k*k]
        b, m = visual_features.size()[:2] # batch size, number of feature maps
        glimpses = attention.size(1)
        visual_features = visual_features.view(b, 1, m, -1) # vectorize feature maps [b, 1, m, k*k]
        attention = attention.view(b, glimpses, -1) # vectorize attention maps [b, glimpses, k*k]
        attention = F.softmax(attention, dim = -1) # [b, glimpses, k*k]
        attention_g1 = attention[:, 0, :]
        attention = attention*mask
        attention[:, 0, :] = attention_g1
        # now apply mask to only one glimpse, leave the other glimpses untouched
        #attention[:, 0, :] = (attention[:, 0, :].unsqueeze(1)*mask).squeeze(1)
        attention.unsqueeze_(2)
        attended = attention*visual_features # use broadcasting to weight the feature maps
        attended = attended.sum(dim=-1) # sum in the spatial dimension [b, glimpses, m]
        return attended.view(b, -1) # return vectorized version with size [b, glimpses*m] 

    # override forward method
    def forward(self, visual_features, mask, question_features, return_maps=False):
        # first, compute attention vectors
        v = self.conv1(self.drop(visual_features))
        q = self.lin1(self.drop(question_features))
        q = utils.expand_like_2D(q, v)
        x = self.relu(self.fuser(v, q))
        x = self.conv2(self.drop(x))

        if return_maps: # if maps have to be returned, save them in a variable
            maps = x.clone()

        # then, apply attention vectors to input visual features
        x = self.apply_attention(visual_features, x, mask)

        if return_maps:
            return x, maps
        else:
            return x