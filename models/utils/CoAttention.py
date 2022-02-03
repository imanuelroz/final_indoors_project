import torch
from torch import nn


class CoAttention(nn.Module):

    def __init__(self, map_dim, hidden_dim, vocab_size, attention_dim=512):
        super(CoAttention, self).__init__()

        # Set parameters
        self.map_dim = map_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        # Prepare linear layers
        self.hidden_linear = nn.Linear(in_features=self.hidden_dim, out_features=self.attention_dim)
        self.co_att_linear = nn.Linear(in_features=self.map_dim, out_features=self.hidden_dim)
        self.rectified_linear = nn.Linear(in_features=self.attention_dim, out_features=1)
        self.pixel_softmax = nn.Softmax(dim=2)  # Make sure this is 2 and not 1. We softmax over the pixels!

        # Prepare the rectifier
        self.rectify = nn.ReLU()

    def forward(self, maps, hiddens):  # x = [maps, hiddens]

        # Obtain basic information
        batch_size = maps.shape[0]

        # Flatten each map from x, y to x * y, where x = n_rows, y = n_cols, [x, y] = pixel
        flattened_maps = maps.view(maps.shape[0], maps.shape[1] * maps.shape[2], maps.shape[3])

        # Go to attention space with linear layers
        map_linear_out = flattened_maps
        hidden_linear_out = self.hidden_linear(hiddens)

        # Make the two resulting vectors ready for pointwise multiplication
        # Add a fake dimension right after the batch size (accounts for tokens)
        map_linear_out = map_linear_out.unsqueeze(1)
        # Add a fake dimension after the number of tokens (accounts for pixels)
        hidden_linear_out = hidden_linear_out.unsqueeze(2)

        # Sum the two vectors -> (batch_size, n_tokens, x * y, map_dim)
        pointwise_sum = map_linear_out.add(hidden_linear_out)

        # Rectify the result and get rid of negative values + help vanishing gradients
        rectified_sum = self.rectify(pointwise_sum)

        # Combine the maps channels together and prepare them for the softmax (batch_size, n_tokens, x * y)
        rectified_linear_out = self.rectified_linear(rectified_sum).squeeze(3)

        # Create the softmap using a softmax function over all the pixels (batch_size, n_tokens, x*y), x*y in [0,1]
        pixel_softmax_out = self.pixel_softmax(rectified_linear_out)

        # Pointwise multiply the softmap with the original maps (batch_size, n_tokens, x*y, map_dim)
        pointwise_mul = torch.mul(flattened_maps.unsqueeze(1), pixel_softmax_out.unsqueeze(3))

        # Sum the pixels together, in each map, to get a unified representation (batch_size, n_tokens, map_dim)
        pointwise_mul = torch.sum(pointwise_mul, dim=2)

        # At this point we expand (or reduce) the resulting tensor to match the hidden one
        co_att_linear_out = self.co_att_linear(pointwise_mul)  # (batch_size, n_tokens, hidden)

        # We finally pointwise multiply the latter output with the original hidden tensor
        co_att_out = torch.mul(co_att_linear_out, hiddens)

        # Return the computed tensors
        return co_att_out, pixel_softmax_out
