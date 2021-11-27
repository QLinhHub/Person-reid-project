import torch
import torch.nn as nn
import torchvision


class EmbedNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base_model = resnet50
        in_feats = self.base_model.fc.in_features

        self.base_model.fc = nn.Linear(in_feats, 1024)
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
