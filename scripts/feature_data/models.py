import torch

from torchvision import transforms


class IdentityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


def get_resnet(layer):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet%d' % layer, pretrained=True)
    model.avgpool = IdentityLayer()
    model.fc = IdentityLayer()
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform
