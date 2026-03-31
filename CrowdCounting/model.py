import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, has_bn=False, load_vgg_weights=False):
        super(CSRNet, self).__init__()
        if has_bn:
            # Front-end: First 10 layers of VGG-16_BN
            # Every Conv2d is now immediately followed by a BatchNorm2d
            self.frontend = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
            )
            
            # Back-end: Dilated convolutions with Batch Normalization added
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )
        else:
            # Front-end: First 10 layers of VGG-16
            self.frontend = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            )
            # Back-end: Dilated convolutions (dilation=2)
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True)
            )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        self._initialize_weights()
        
        if load_vgg_weights:
            vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            
            for i in range(len(self.frontend)):
                if isinstance(self.frontend[i], nn.Conv2d) or isinstance(self.frontend[i], nn.BatchNorm2d):
                    self.frontend[i].weight.data = vgg16_bn.features[i].weight.data
                    self.frontend[i].bias.data = vgg16_bn.features[i].bias.data
                    if isinstance(self.frontend[i], nn.BatchNorm2d):
                        self.frontend[i].running_mean.data = vgg16_bn.features[i].running_mean.data
                        self.frontend[i].running_var.data = vgg16_bn.features[i].running_var.data

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
