import torch
import torch.nn as nn
import torchvision


class EmbedNet(nn.Module):
    '''
    Embeded network using transfer learning technique, Resnet50 is used as backbone, the last layer is discarded and 2 FC layers are added instead.
    The first has 1024 units, followed by batch norm and ReLU, the final go down to 128 units -> final embeded demension
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet50 = torchvision.models.resnet50(pretrained=True) # load resnet50 model with pretrained weights
        self.base_model = resnet50
        in_feats = self.base_model.fc.in_features # get number of input features of the last layer of resnet50

        self.base_model.fc = nn.Linear(in_feats, 1024) # replace last layer of resnet50 to output 1024 units
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 128)
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    from pytorch_model_summary import summary
    embed_net = EmbedNet()

    in_ten = torch.randn(32, 3, 256, 128)
    print(summary(embed_net, in_ten))
    out = embed_net(in_ten)
    print(out.shape)

#TODO: write more complexe code to know more about Pytorch, espesscially about state dict