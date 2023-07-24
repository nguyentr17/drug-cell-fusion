import torch
import torch.nn as nn
from collections import OrderedDict


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""
    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.squeeze(data, dim=self.dim)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.unsqueeze(data, self.dim)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature


def dense_attention_layer(
        number_of_features: int,
        temperature: float = 1.0,
        dropout=0.0
) -> nn.Sequential:
    """
    Source: https://github.com/PaccMann/paccmann_predictor/blob/7c9011084835bc4a092371df8cd4166e090af446/paccmann_predictor/utils/layers.py
    Attention mechanism layer for dense inputs.
    Args:
        number_of_features (int): Size to allocate weight matrix.
        temperature (float): Softmax temperature parameter (0, inf). Lower
            temperature (< 1) result in a more descriminative/spiky softmax,
            higher temperature (> 1) results in a smoother attention.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(number_of_features, number_of_features)),
                ('dropout', nn.Dropout(p=dropout)),
                ('temperature', Temperature(temperature)),
                ('softmax', nn.Softmax(dim=-1)),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):
    """
    Source: https://github.com/PaccMann/paccmann_predictor/blob/7c9011084835bc4a092371df8cd4166e090af446/paccmann_predictor/utils/layers.py

    Implements context attention as in the PaccMann paper (Figure 2C) in
    Molecular Pharmaceutics.
    With the additional option of having a hidden size in the context.
    NOTE:
    In tensorflow, weights were initialized from N(0,0.1). Instead, pytorch
    uses U(-stddev, stddev) where stddev=1./math.sqrt(weight.size(1)).
    """

    def __init__(
            self,
            reference_hidden_size: int,
            reference_sequence_length: int,
            context_hidden_size: int,
            context_sequence_length: int = 1,
            attention_size: int = 16,
            individual_nonlinearity: type = nn.Sequential(),
            temperature: float = 1.0,
    ):
        """Constructor
        Arguments:
            reference_hidden_size (int): Hidden size of the reference input
                over which the attention will be computed (H).
            reference_sequence_length (int): Sequence length of the reference
                (T).
            context_hidden_size (int): This is either simply the amount of
                features used as context (G) or, if the context is a sequence
                itself, the hidden size of each time point.
            context_sequence_length (int): Hidden size in the context, useful
                if context is also textual data, i.e. coming from nn.Embedding.
                Defaults to 1.
            attention_size (int): Hyperparameter of the attention layer,
                defaults to 16.
            individual_nonlinearities (type): This is an optional
                nonlinearity applied to each projection. Defaults to
                nn.Sequential(), i.e. no nonlinearity. Otherwise it expects a
                torch.nn activation function, e.g. nn.ReLU().
            temperature (float): Temperature parameter to smooth or sharpen the
                softmax. Defaults to 1. Temperature > 1 flattens the
                distribution, temperature below 1 makes it spikier.
        """
        super().__init__()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(reference_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(context_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'projection',
                            nn.Linear(
                                context_sequence_length,
                                reference_sequence_length,
                            ),
                        ),
                        ('act_fn', individual_nonlinearity),
                    ]
                )
            )  # yapf: disable
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ('projection', nn.Linear(attention_size, 1, bias=False)),
                    ('squeeze', Squeeze()),
                    ('temperature', Temperature(self.temperature)),
                    ('softmax', nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(
            self,
            reference: torch.Tensor,
            context: torch.Tensor,
            average_seq: bool = True
    ):
        """
        Forward pass through a context attention layer
        Arguments:
            reference (torch.Tensor): This is the reference input on which
                attention is computed. Shape: bs x ref_seq_length x ref_hidden_size
            context (torch.Tensor): This is the context used for attention.
                Shape: bs x context_seq_length x context_hidden_size
            average_seq (bool): Whether the filtered attention is averaged over the
                sequence length.
                NOTE: This is recommended to be True, however if the ref_hidden_size
                is 1, this can be used to prevent collapsing to a single float.
                Defaults to True.
        Returns:
            (output, attention_weights):  A tuple of two Tensors, first one
                containing the reference filtered by attention (shape:
                bs x ref_hidden_size) and the second one the
                attention weights (bs x ref_seq_length).
                NOTE: If average_seq is False, the output is: bs x ref_seq_length
        """
        assert len(reference.shape) == 3, 'Reference tensor needs to be 3D'
        assert len(context.shape) == 3, 'Context tensor needs to be 3D'

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(
            self.context_projection(context).permute(0, 2, 1)
        ).permute(0, 2, 1)
        alphas = self.alpha_projection(
            torch.tanh(reference_attention + context_attention)
        )

        output = reference * torch.unsqueeze(alphas, -1)
        output = torch.sum(output, 1) if average_seq else torch.squeeze(output)

        return output, alphas


class SelfAttentionFunction(nn.Module):
    def __init__(self, gene_embed_size: int,
                 drug_embed_size:int,
                 temperature: float = 1.0,
                 dropout: float = 0.0,
                 concat: bool = True
                 ):
        super().__init__()
        self.gene_embed_size = gene_embed_size
        self.drug_embed_size = drug_embed_size
        self.concat = concat
        if self.concat:
            self.gene_attention = dense_attention_layer(gene_embed_size, temperature, dropout)
        else:
            self.gene_drug_embed = nn.Linear(gene_embed_size, drug_embed_size)
            self.gene_attention = dense_attention_layer(drug_embed_size, temperature, dropout)
        self.drug_attention = dense_attention_layer(drug_embed_size, temperature, dropout)
        self.concat = concat

    def forward(self, gene: torch.Tensor, drug: torch.Tensor):
        if self.concat:
            attention_gene = self.gene_attention(gene)
            attention_drug = self.drug_attention(drug)
            attended_gene = gene * attention_gene
            attended_drug = drug * attention_drug
            return torch.cat([attended_gene, attended_drug], 1)
        else:
            # need to bring gene and drug to same dimension; TODO: better way?
            gene = self.gene_drug_embed(gene)
            attention_gene = self.gene_attention(gene)
            attention_drug = self.drug_attention(drug)
            attention_sum = attention_gene + attention_drug
            attended_gene = gene * (attention_gene / attention_sum)
            attended_drug = drug * (attention_drug / attention_sum)
            return attended_gene + attended_drug


class CrossAttentionFunction(nn.Module):
    def __init__(self,
                 gene_embed_size: int,
                 drug_embed_size:int,
                 gene_attention_size: int,
                 drug_attention_size: int,
                 temperature: float = 1.0,
                 activation_fn: nn.Module = nn.Sequential(),
                 ):
        super().__init__()
        self.gene_unsqueeze = Unsqueeze(1) # only needed if doing at root level
        self.drug_unsqueeze = Unsqueeze(1)
        self.gene_attention_layer = ContextAttentionLayer(
            reference_hidden_size=gene_embed_size,
            reference_sequence_length=1, # if using root GO term
            context_hidden_size=drug_embed_size,
            context_sequence_length=1,
            attention_size=gene_attention_size, # same as gene_embed_size so no bottleneck
            individual_nonlinearity=activation_fn, #TODO: need to hp search
            temperature=1.0, #TODO: need to hp search
        )
        self.drug_attention_layer = ContextAttentionLayer(
            reference_hidden_size=drug_embed_size,
            reference_sequence_length=1,
            context_hidden_size=gene_embed_size,
            context_sequence_length=1, #TODO: make it dynamic
            attention_size=drug_attention_size,
            individual_nonlinearity=activation_fn,
            temperature=temperature
        )
        self.gene_squeeze = Squeeze(1)
        self.drug_squeeze = Squeeze(1)

    def forward(self, gene, drug):
        gene = self.gene_unsqueeze(gene)
        drug = self.drug_unsqueeze(drug)
        attended_gene, gene_attention = self.gene_attention_layer(gene, drug)
        attended_drug, drug_attention = self.drug_attention_layer(drug, gene)
        attended_gene = self.gene_squeeze(attended_gene)
        attended_drug = self.drug_squeeze(attended_drug)

        return torch.cat([attended_gene, attended_drug], dim=1)


