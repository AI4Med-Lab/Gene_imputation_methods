"""SpaGene architecture: encoders, decoders, translators, discriminators."""
import torch
from torch import nn

class Linear1DBlock(nn.Module):
    """MLP block matching SpaGene's linear/batchnorm/LeakyReLU/dropout pattern."""

    def __init__(
        self,
        dim_in,
        dim_out,
        features,
        dropout=0.2,
        norm_out=False,
        relu_out=False,
    ):
        super().__init__()
        dims = [int(dim_in)] + [int(x) for x in features]
        layers = []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(d0, d1),
                    nn.BatchNorm1d(d1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dims[-1], int(dim_out)))
        if norm_out:
            layers.append(nn.BatchNorm1d(int(dim_out)))
        if relu_out:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpaGeneNet(nn.Module):
    def __init__(self, n_st_genes, n_sc_genes):
        super().__init__()

        self.enc_st = Linear1DBlock(
            n_st_genes, 256, [2048, 1024, 512], dropout=0.2, norm_out=True, relu_out=True
        )
        self.dec_st = Linear1DBlock(
            256, n_st_genes, [512, 1024, 2048], dropout=0.2
        )

        self.enc_sc = Linear1DBlock(
            n_sc_genes, 512, [2048, 1024, 512], dropout=0.2, norm_out=True, relu_out=True
        )
        self.dec_sc = Linear1DBlock(
            512, n_sc_genes, [512, 1024, 2048], dropout=0.2
        )

        self.trans_st2sc = Linear1DBlock(
            256, 512, [512, 512, 512, 1024, 1024], dropout=0.1, norm_out=True, relu_out=True
        )
        self.trans_sc2st = Linear1DBlock(
            512, 256, [512, 256, 256, 128, 128], dropout=0.1, norm_out=True, relu_out=True
        )

        self.disc_st = Linear1DBlock(
            256, 1, [256, 128, 128, 64], dropout=0.1
        )
        self.disc_sc = Linear1DBlock(
            512, 1, [256, 128, 128, 64], dropout=0.1
        )

    def predict_st_to_sc(self, x_st):
        return self.dec_sc(self.trans_st2sc(self.enc_st(x_st)))

