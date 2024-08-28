from torch import nn

from models.eiv_layer import EIVDropout
from models.ddpm import DDPM
from models.model_pred import MLP


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def get_model(name: str, **kwargs) -> nn.Module:
    if name == "ddpm":
        model = DDPM(**kwargs)
    else:
        raise NotImplementedError(name)
    
    return model


def get_model_pred(name: str, **kwargs) -> nn.Module:
    """
    Loads model. 
    name for MLP is: mlpab with a being the number of hidden layers and 
        b being the n_layer
    """
    if name == "mlp":
        model = MLP(**kwargs)
    elif name == "resnet9":
        model = resnet9(**kwargs)
    elif name == "resnet9_dropout":
        model = resnet9_dropout(**kwargs)
    elif name == "vit_s":
        model = vit_small_dino(**kwargs)
    
    else:
        raise NotImplementedError(name)
    
    return model


def conv_block(in_channels, out_channels, dim, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels, track_running_stats = False),
            #   nn.LayerNorm([out_channels, dim, dim]), 
            # nn.InstanceNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class resnet9_dropout(nn.Module):
    """
    from https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min
    """
    def __init__(self, dim, n_channel, n_class, p, n_zeta, n_zeta_mean, **kwargs):
        super().__init__()

        self.dropout = nn.Dropout(p = p) if (n_zeta == 0 or n_zeta_mean) \
            else EIVDropout(p = p, n_zeta = n_zeta)
        
        self.conv1 = conv_block(n_channel, 64, dim)
        self.conv2 = conv_block(64, 128, dim, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, dim // 2), 
                                  conv_block(128, 128, dim // 2))
        
        self.conv3 = conv_block(128, 256, dim // 2, pool=True)
        self.conv4 = conv_block(256, 512, dim // 4, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, dim // 8), 
                                  conv_block(512, 512, dim // 8))

        self.pool = nn.MaxPool2d(2)
                                        
        self.linear = nn.Sequential(nn.Flatten(),
                                    self.dropout,
                                    nn.Linear(512 * 4, 512))
        
        self.classifier = nn.Sequential(self.dropout, 
                                        nn.Linear(512, n_class))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.pool(out)
        out = self.linear(out)
        out = self.classifier(out)
        return out


class resnet9(nn.Module):
    """
    from https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min
    """
    def __init__(self, dim, n_channel, n_class, **kwargs):
        super().__init__()
        
        self.conv1 = conv_block(n_channel, 64, dim)
        self.conv2 = conv_block(64, 128, dim, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128, dim // 2), 
                                  conv_block(128, 128, dim // 2))
        
        self.conv3 = conv_block(128, 256, dim // 2, pool=True)
        self.conv4 = conv_block(256, 512, dim // 4, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512, dim // 8), 
                                  conv_block(512, 512, dim // 8))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(), 
                                        nn.Linear(512, n_class))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
