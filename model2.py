import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

###model implementation 
'''ref: 1. https://github.com/Spijkervet/SimCLR/blob/master/simclr/simclr.py
2. https://discuss.pytorch.org/t/how-can-i-change-pretrained-resnet-model-to-accept-single-channel-image/50912/2'''

'''
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, projection_dim):
        super(Model, self).__init__()
        self.encoder = resnet50()
        #self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, projection_dim, bias=False),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projector(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
'''
class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model(128).f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.f.load_state_dict(torch.load(pretrained_path),strict=False)
        # freeze convolutional layers
        #print("###seq list:",(*list(self.f.children())[:1]))
        self.features = nn.Sequential(*list(self.f.children())[:1])
        for p in self.features.parameters():
          p.requires_grad = True
        #self.fc=
        '''self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(1048576, 14876)) 
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.3))
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc6_s1',nn.Linear(14876,2048)) 
        self.fc7.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc7.add_module('drop6_s1',nn.Dropout(p=0.3))'''
        #self.fc= nn.Linear(2048,num_class,bias=True)
        
        #self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        #print("x",x.shape)
        feature1 = torch.flatten(x, start_dim=1)
        #feature=self.fc6(feature1)
        #print("feature",feature1.shape)
        #f1=self.f7(feature)
        #print("f1",f1.shape)
        out = self.fc(feature1)
        return out

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50(pretrained=True).named_children():
            #if name == 'conv1':
            #    module = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
            if not isinstance(module, nn.Linear) :
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        #print("###seq list:",(*list(self.f.children())[:1]))
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
