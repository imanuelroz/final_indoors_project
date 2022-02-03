from torch import nn
import torchvision.models as models


class VGGEncoder(nn.Module):

    def __init__(self, instance=models.vgg19(pretrained=True)):
        super(VGGEncoder, self).__init__()

        # Using pre-trained vgg
        vgg = instance

        # Remove the classifier
        modules = list(vgg.children())[:-1]

        # Keep only the network
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        # outputs maps from vgg (512 channels average pooled @ 8x8)
        out = self.vgg(images)  # (batch_size, 512, 8, 8)
        return out.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512)
