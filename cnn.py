from torch import nn
import torch

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, output_size):
        super(ResNet, self).__init__()
        self.in_channels = 64
        num_channels = 1
        
        self.convolution_1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, output_size)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.convolution_1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, num_blocks, planes, stride=1):
        downsample = None

        # If stride is not 1 or in_channels != planes * expansion, apply downsampling
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers = []
        layers.append(ResBlock(self.in_channels, planes, input_downsample=downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, input_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.convolution_3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.input_downsample = input_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        y = self.relu(self.batch_norm1(self.convolution_1(x)))
        
        y = self.relu(self.batch_norm2(self.convolution_2(y)))
        
        y = self.batch_norm3(self.convolution_3(y))
        
        if self.input_downsample is not None:
            identity = self.input_downsample(identity)

        y += identity
        y = self.relu(y)
        
        return y
        
class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, input_downsample=None, stride=1):
        super(Block, self).__init__()

        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.input_downsample = input_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        y = self.relu(self.batch_norm1(self.convolution_1(x)))
        
        y = self.batch_norm2(self.convolution_2(y))

        if self.input_downsample is not None:
            identity = self.input_downsample(identity)

        y += identity
        y = self.relu(y)

        return y
