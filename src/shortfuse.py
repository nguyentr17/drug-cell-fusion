from torch import nn
import torch


class ShortFuse(nn.Module):
    def __init__(self, aux_dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(p=aux_dropout_rate)

    def forward(self, primary_fea, aux_fea):
        aux_fea = self.dropout(aux_fea)
        out = torch.cat([primary_fea, aux_fea], 1)
        return out

class TensorFusion(nn.Module):
    def __init__(self, aux_size: int = None):
        super().__init__()
        self.aux_size = aux_size

    def forward(self, primary_fea, aux_fea):
        """
        Adapted from https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py

        :param primary_fea:
        :param aux_fea:
        :return:
        """
        if self.aux_size is not None:
            # Only use the first component
            aux_fea = aux_fea[:, 0:self.aux_size]
        batch_size = primary_fea.shape[0]
        out = torch.bmm(primary_fea.unsqueeze(2), aux_fea.unsqueeze(1))
        out = out.view(batch_size, -1)
        return out

