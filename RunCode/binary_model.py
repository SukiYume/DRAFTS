import numpy as np
import torch, torchvision
from torchvision import transforms


if True:
    def random_resize(inputs):

        h, w = np.random.randint(64, 513), np.random.randint(64, 513)
        inputs = torch.stack([transforms.Resize((h, w), antialias=True)(k) for k in inputs])

        return inputs

if False:
    def random_resize(inputs):

        random_seed = np.random.rand()
        if random_seed < 0.125:
            h, w = 64, 64
        elif 0.125 <= random_seed <= 0.250:
            h, w = 128, 128
        elif 0.250 <= random_seed <= 0.375:
            h, w = 256, 256
        elif 0.375 <= random_seed <= 0.500:
            h, w = 128, 512
        elif 0.500 <= random_seed <= 0.625:
            h, w = 128, 256
        elif 0.625 <= random_seed <= 0.875:
            h, w = np.random.randint(64, 513), np.random.randint(64, 513)
        if random_seed <= 0.875:
            inputs = torch.stack([transforms.Resize((h, w), antialias=True)(k) for k in inputs])

        return inputs


class BinaryNet(torch.nn.Module):

    def __init__(self, model_name='resnet18', num_classes=2):
        super(BinaryNet, self).__init__()
        model_dict = {
            'resnet18':   [torchvision.models.resnet18(weights=None), 512],
            'resnet34':   [torchvision.models.resnet34(weights=None), 512],
            'resnet50':   [torchvision.models.resnet50(weights=None), 2048],
            'resnet50v2': [torchvision.models.wide_resnet50_2(weights=None), 2048]
        }
        basemodel, num_ch     = model_dict[model_name]
        self.base_model       = basemodel
        self.base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc    = torch.nn.Linear(num_ch, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x


class SpatialPyramidPool2D(torch.nn.Module):

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out         = None
        for n in self.out_side:
            y       = torch.nn.AdaptiveMaxPool2d(output_size=(n, n))(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


class SPPResNet(torch.nn.Module):

    def __init__(self, model_name='resnet18', num_classes=2, pool_size=(1, 2, 6)):

        super().__init__()
        model_dict = {
            'resnet18':   [torchvision.models.resnet18(weights=None), 512],
            'resnet34':   [torchvision.models.resnet34(weights=None), 512],
            'resnet50':   [torchvision.models.resnet50(weights=None), 2048],
            'resnet50v2': [torchvision.models.wide_resnet50_2(weights=None), 2048]
        }
        basemodel, num_ch = model_dict[model_name]

        self.base_model = basemodel
        self.base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.spp        = SpatialPyramidPool2D(out_side=pool_size)
        num_features    = num_ch * (pool_size[0]**2 + pool_size[1]**2 + pool_size[2]**2)
        self.classifier = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):

        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.spp(x)
        x = self.classifier(x)

        return x