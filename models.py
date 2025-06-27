import torch
import torch.nn as nn
from torchvision import models, transforms

class Generator(nn.Module):
    
    def __init__(self,n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.n_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            # Last conv layer outputs 3 channels (RGB), no batchnorm or activation
            nn.Conv2d(self.n_hidden, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self,input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.features = nn.Sequential(
            nn.Conv2d(3, self.n_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1),
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_hidden, self.n_hidden, 3, padding=1), # This could be your feature layer
            nn.BatchNorm2d(self.n_hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output_layer = nn.Conv2d(self.n_hidden, 1, 3, padding=1)

    def forward(self, input, return_features=False):
        features = self.features(input)
        if return_features:
            return features
        return self.output_layer(features)
    


class InceptionV3_Multi(nn.Module):
    """
    Inception-v3 gelé qui renvoie des features à trois profondeurs :
    - Mixed_3c  → détails fins   (35×35×288)
    - Mixed_5c  → formes moyennes (17×17×768)
    - Mixed_7c  → contenu global  (8×8×2048)
    """
    def __init__(self, device='cpu'):
        super().__init__()
        net = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False)
        net.eval()

        # Gèle les poids
        for p in net.parameters():
            p.requires_grad = False

        # Découpe bloc par bloc
        self.blocks = nn.ModuleDict()
        wanted = {'Mixed_5d', 'Mixed_6e', 'Mixed_7c'}
        x = []
        for name, m in net.named_children():
            if name in {'AuxLogits', 'fc'}:
                continue
            x.append(m)
            if name in wanted:
                self.blocks[name] = nn.Sequential(*x)
                x = []          # on recommence pour le bloc suivant
            if name == 'Mixed_7c':
                break
        self.device = device
        self.to(device)

    def forward(self, img, layers=('Mixed_5d', 'Mixed_6e', 'Mixed_7c')):
        outs = []
        h = img
        for name in layers:
            h = self.blocks[name](h)
            outs.append(h)
        return outs              # liste dans le même ordre que layers

    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class SWPatchWeights(nn.Module):
    def __init__(self, n_patches: int, C: float):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(n_patches))
        self.C   = C                       

    def forward(self):
        return torch.softmax(self.raw, dim=0) * self.C    # (n_patches,)
