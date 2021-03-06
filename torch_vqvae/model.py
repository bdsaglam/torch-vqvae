# Credits: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/
import torch
from torch import nn
from torch.nn import functional as F

from torch_vqvae.vector_quantizer import VectorQuantizerEMA, VectorQuantizer


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQVAE(nn.Module):
    def __init__(self, input_shape, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super().__init__()
        self.input_shape = input_shape
        self._encoder = Encoder(in_channels=input_shape[0],
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                          commitment_cost, decay)
        else:
            self._vq = VectorQuantizer(num_embeddings, embedding_dim,
                                       commitment_cost)
        self._decoder = Decoder(in_channels=embedding_dim,
                                out_channels=input_shape[0],
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens)
        with torch.no_grad():
            image = torch.rand(1, *input_shape)
            res = self.forward(image)
            self.output_shapes = dict(
                quantized=res.quantized.squeeze(0).shape,
                vq_loss=res.vq_loss.squeeze(0).shape,
                perplexity=res.perplexity.squeeze(0).shape,
                embedding_mask=res.embedding_mask.squeeze(0).shape,
            )

    def encode(self, image):
        feature = self._encoder(image)
        latent = self._pre_vq_conv(feature)
        res = self._vq(latent)
        return res

    def decode(self, quantized):
        rec = self._decoder(quantized)
        return rec

    def forward(self, x):
        res = self.encode(x)
        rec = self.decode(res.quantized)
        res.update(rec=rec)
        return res

    def reconstruction_loss(self, rec, target):
        return F.mse_loss(rec, target)
