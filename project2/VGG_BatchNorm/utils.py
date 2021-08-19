"""
Utils for neural networks
"""

from torch import nn


def init_weights_(m):
    """
    Initializes weights of m according to Xavier normal method.

    :param m: module
    :return:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_vgg_weights(model):
    # features submodule
    features_weights = []
    features_bias = []
    # classifier submodule
    classifier_weights = []
    classifier_bias = []
    
    for i in range(len(model.features)):
        m = model.features[i]
        if hasattr(m, 'weight'):
            features_weights.append(m.weight.data.clone())
        if hasattr(m, 'bias'):
            features_bias.append(m.bias.data.clone())
            
    for i in range(len(model.classifier)):
        m = model.classifier[i]
        if hasattr(m, 'weight'):
            classifier_weights.append(m.weight.data.clone())
        if hasattr(m, 'bias'):
            classifier_bias.append(m.bias.data.clone())
            
    weights = features_weights, features_bias, classifier_weights, classifier_bias
    
    return weights        

def set_vgg_weights(model, weights, feature_border=None, classifier_border=None):
    features_weights, features_bias, classifier_weights, classifier_bias = weights
    
    weight_idx = 0
    bias_idx = 0
    for i in range(len(model.features)):
        if feature_border == i:
            return
        
        m = model.features[i]
        if hasattr(m, 'weight'):
            m.weight.data = features_weights[weight_idx]
            weight_idx += 1
        if hasattr(m, 'bias'):
            m.bias.data = features_bias[bias_idx]
            bias_idx += 1
            
    weight_idx = 0
    bias_idx = 0
    for i in range(len(model.classifier)):
        if classifier_border == i:
            return
        
        m = model.classifier[i]
        if hasattr(m, 'weight'):
            m.weight.data = classifier_weights[weight_idx]
            weight_idx += 1
        if hasattr(m, 'bias'):
            m.bias.data = classifier_bias[bias_idx]
            bias_idx += 1
    return