import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1,expansion=2):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                     nn.BatchNorm2d(planes),
                     nn.ReLU(True))
               
        
        self.weight = nn.Parameter(init.kaiming_normal_(torch.empty(planes*expansion, planes, 1, 1)))
        self.bn1 = nn.BatchNorm2d(planes*expansion)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
           self.shortcut = nn.Sequential(
                           nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                           nn.BatchNorm2d(planes)
                            )

           

    def forward(self, x):
        out = self.conv1(x)

        out = F.conv2d(out,self.weight,bias=None,stride=1,padding=0)
        out = self.bn1(out)
        out = F.relu(out,inplace=True)

        out = F.conv2d(out,self.weight.transpose(1,0),bias=None,stride=1,padding=0)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        return out

 
class ShareNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width_scaler=1., expansion=2):
        super(ShareNet, self).__init__()
        

        self.in_planes = int(16*width_scaler)

        self.conv1 = nn.Sequential(nn.Conv2d(3,self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(self.in_planes),
                                   nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, int(16*width_scaler), num_blocks[0], stride=1, expansion=expansion)
        self.layer2 = self._make_layer(block, int(32*width_scaler), num_blocks[1], stride=2, expansion=expansion)
        self.layer3 = self._make_layer(block, int(64*width_scaler), num_blocks[2], stride=2, expansion=expansion)
        self.linear = nn.Linear(int(64*width_scaler), num_classes)


    def _make_layer(self, block, planes, num_blocks, stride,expansion):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, expansion))
            self.in_planes = planes 

        return nn.Sequential(*layers)

    

    def forward(self, x):
        
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def sharenet20(num_classes,expansion,width_scaler=1):
    return ShareNet(BasicBlock, [3, 3, 3],num_classes, expansion=expansion,width_scaler=width_scaler)


def sharenet56(num_classes,expansion,width_scaler=1):
    return ShareNet(BasicBlock, [9, 9, 9],num_classes, expansion=expansion,width_scaler=width_scaler)


