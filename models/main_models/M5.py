from transformers import GPT2LMHeadModel
import torch
from torch import nn
from models.utils.CoAttention import CoAttention
from models.utils.VGGEncoder import VGGEncoder


class Model(nn.Module):

    def __init__(self, tokenizer):
        super(Model, self).__init__()

        self.embedding_dim = 1024
        self.channels = 512

        # Tokenizer
        self.tokenizer = tokenizer

        # GPT2 Encoder
        self.pre_trained_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.pre_trained_gpt2.resize_token_embeddings(len(tokenizer))

        self.head = list(self.pre_trained_gpt2.children())[1]
        self.gpt2 = list(self.pre_trained_gpt2.children())[0]

        # Image Encoder
        # self.image_encoder = list(models.vgg19(pretrained=True).children())[0]
        self.image_encoder = VGGEncoder()

        # CoAttention Layer
        self.co_att = CoAttention(self.channels, self.embedding_dim, len(self.tokenizer), 512)

        # Classifier
        self.classifier = nn.Linear(in_features=self.embedding_dim, out_features=len(self.tokenizer))
        self.softmax = nn.Softmax()

        # Copy the original weights and concat the new ones for the attention
        with torch.no_grad():
            self.classifier.weight.copy_(self.head.weight)

        del self.head

        # Disable weight update for both VGG and GPT-2
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False
        for p in self.pre_trained_gpt2.parameters():
            p.requires_grad = False

        # Enable weight updates for classifier and attention layer
        for p in self.classifier.parameters():
            p.requires_grad = True
        for p in self.co_att.parameters():
            p.requires_grad = True

    def forward(self, sequence, image):
        maps = self.image_encoder(image)
        hiddens = self.gpt2(sequence)[0]
        co_att_out, pixel_softmax_out = self.co_att(maps, hiddens)
        combined_embedding = torch.add(hiddens, co_att_out)
        out = self.classifier(combined_embedding)
        return out, pixel_softmax_out
