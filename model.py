import torch
import torch.nn as nn

def conv_small(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=[1, 4], stride=stride,
                     padding=(0, 1), bias=False)

# def conv_medium(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=[1, 8], stride=stride,
#                      padding=(0, 1), bias=False)

def conv_large(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=[1, 8], stride=stride,
                     padding=(0, 1), bias=False)


class BasicBlockSmall(nn.Module):
    expansion = 1

    def __init__(self, inplanes_small, planes, stride=(1, 2), downsample=None):
        super(BasicBlockSmall, self).__init__()
        self.conv1 = conv_small(inplanes_small, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.GELU1 = nn.GELU()
        self.GELU2 = nn.GELU()
        self.conv2 = conv_small(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.GELU1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        minimum = min(out.shape[-1], residual.shape[-1])
        out1 = residual[:, :, :, 0:minimum] + out[:, :, :, 0:minimum]
        out1 = self.GELU2(out1)

        return out1


# class BasicBlockMedium(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes_medium, planes, stride=(1, 2), downsample=None):
#         super(BasicBlockMedium, self).__init__()
#         self.conv1 = conv_medium(inplanes_medium, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.GELU1 = nn.GELU()
#         self.GELU2 = nn.GELU()
#         self.conv2 = conv_medium(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.GELU1(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         minimum = min(out.shape[-1], residual.shape[-1])
#         out1 = residual[:, :, :, 0:minimum] + out[:, :, :, 0:minimum]
#         out1 = self.GELU2(out1)

#         return out1


class BasicBlockLarge(nn.Module):
    expansion = 1

    def __init__(self, inplanes_large, planes, stride=(1, 2), downsample=None):
        super(BasicBlockLarge, self).__init__()
        self.conv1 = conv_large(inplanes_large, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.GELU1 = nn.GELU()
        self.GELU2 = nn.GELU()
        self.conv2 = conv_large(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.GELU1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        minimum = min(out.shape[-1], residual.shape[-1])
        out1 = residual[:, :, :, 0:minimum] + out[:, :, :, 0:minimum]
        out1 = self.GELU2(out1)

        return out1


class MSResNet(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1, 1, 1], num_classes=2):
        self.inplanes_small = 1
        self.inplanes_medium = 1
        self.inplanes_large = 1

        super(MSResNet, self).__init__()

        self.layer_small_1 = self._make_layer_small(BasicBlockSmall, 20, layers[0], stride=(1, 2))
        self.layer_small_2 = self._make_layer_small(BasicBlockSmall, 50, layers[1], stride=(1, 2))
        self.layer_small_3 = self._make_layer_small(BasicBlockSmall, 100, layers[2], stride=(1, 2))
        self.layer_small_4 = self._make_layer_small(BasicBlockSmall, 150, layers[3], stride=(1, 2))

        # maxplooing kernel size: 16, 11, 6
        self.pooling_small = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2), padding=0)

        # self.layer_medium_1 = self._make_layer_medium(BasicBlockMedium, 20, layers[0], stride=(1, 2))
        # self.layer_medium_2 = self._make_layer_medium(BasicBlockMedium, 50, layers[1], stride=(1, 2))
        # self.layer_medium_3 = self._make_layer_medium(BasicBlockMedium, 100, layers[2], stride=(1, 2))
        # self.layer_medium_4 = self._make_layer_medium(BasicBlockMedium, 150, layers[3], stride=(1, 2))
        # self.pooling_medium = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2), padding=0)

        self.layer_large_1 = self._make_layer_large(BasicBlockLarge, 20, layers[0], stride=(1, 2))
        self.layer_large_2 = self._make_layer_large(BasicBlockLarge, 50, layers[1], stride=(1, 2))
        self.layer_large_3 = self._make_layer_large(BasicBlockLarge, 100, layers[2], stride=(1, 2))
        self.layer_large_4 = self._make_layer_large(BasicBlockLarge, 150, layers[3], stride=(1, 2))
        self.pooling_large = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2), padding=0)

        self.fc1 = nn.Linear(277200, num_classes)

    def _make_layer_small(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != 1 or self.inplanes_small != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_small, planes * block.expansion,
                          kernel_size=(1, 2), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_small, planes, stride, downsample))
        self.inplanes_small = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_small, planes))

        return nn.Sequential(*layers)

    def _make_layer_medium(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != 1 or self.inplanes_medium != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_medium, planes * block.expansion,
                          kernel_size=(1, 2), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_medium, planes, stride, downsample))
        self.inplanes_medium = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_medium, planes))

        return nn.Sequential(*layers)

    def _make_layer_large(self, block, planes, blocks, stride=(1, 2)):
        downsample = None
        if stride != 1 or self.inplanes_large != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_large, planes * block.expansion,
                          kernel_size=(1, 2), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes_large, planes, stride, downsample))
        self.inplanes_large = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_large, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x = self.layer_small_1(x0)
        x = self.layer_small_2(x)
        x = self.layer_small_3(x)
        x = self.layer_small_4(x)

        x = self.pooling_small(x)

        # y = self.layer_medium_1(x0)
        # y = self.layer_medium_2(y)
        # y = self.layer_medium_3(y)
        # y = self.layer_medium_4(y)

        # y = self.pooling_medium(y)

        z = self.layer_large_1(x0)
        z = self.layer_large_2(z)
        z = self.layer_large_3(z)
        z = self.layer_large_4(z)

        z = self.pooling_large(z)

        # max_dim = max(x.size(-1), y.size(-1), z.size(-1))

        # x = nn.functional.pad(x, [max_dim - x.size(-1), 0])
        # y = nn.functional.pad(y, [max_dim - y.size(-1), 0])
        # z = nn.functional.pad(z, [max_dim - z.size(-1), 0])

        out = torch.cat([x, z], dim=-1)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

