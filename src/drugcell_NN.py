from torch.autograd import Variable
import torch.nn as nn
from util import *
from shortfuse import ShortFuse, TensorFusion
from attention_fusion import SelfAttentionFunction, CrossAttentionFunction
from typing import Dict

# TODO: replace dictionary with wrapper class
FUSION_FUNCTIONS = {
    "concat": lambda **kwargs: Concat(dim=1),
    "self_attention_concat": lambda gene_dim, drug_dim, device, **kwargs: SelfAttentionFunction(
        gene_embed_size=gene_dim,
        drug_embed_size=drug_dim,
        concat=True).to(device),
    "self_attention_weighted_sum": lambda gene_dim, drug_dim, device, **kwargs: SelfAttentionFunction(
        gene_embed_size=gene_dim,
        drug_embed_size=drug_dim,
        concat=False).to(device),
    "cross_attention": lambda gene_dim, drug_dim, device, gene_attention_size, drug_attention_size, temperature,
                              activation_fn, **kwargs: CrossAttentionFunction(gene_embed_size=gene_dim,
                                                                              drug_embed_size=drug_dim,
                                                                              gene_attention_size=gene_attention_size,
                                                                              drug_attention_size=drug_attention_size,
                                                                              temperature=temperature,
                                                                              activation_fn=activation_fn).to(device),
    "tensor_product_full": lambda device, **kwargs: TensorFusion(aux_size=None).to(device),
    "tensor_product_partial": lambda device, aux_size, **kwargs: TensorFusion(aux_size=aux_size).to(device)
}


class drugcell_nn(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, ndrug, root, num_hiddens_genotype,
                 num_hiddens_drug, num_hiddens_final, device,
                 fusion_function: str = "concat", fuse_hyperparameters: Dict = {}, gene_fuse_position: int = 6,
                 final_concat: bool = False,
                 return_hidden: bool = True):
        super(drugcell_nn, self).__init__()

        self.root = root
        self.num_hiddens_genotype = num_hiddens_genotype
        self.num_hiddens_drug = num_hiddens_drug
        self.device = device
        self.return_hidden = return_hidden

        # fusion function
        self.fuse_add_drug_embedding = fusion_function in ["concat", "self_attention_concat", "cross_attention"]
        self.fuse_hyperparameters = fuse_hyperparameters
        self.gene_fuse_position = gene_fuse_position
        self.final_concat = final_concat
        self.fusion_function = fusion_function
        if gene_fuse_position == 6:
            self.add_module(f"FUSE_gene_position_6",
                            FUSION_FUNCTIONS[fusion_function](gene_dim=6, drug_dim=6, device=device,
                                                              **fuse_hyperparameters))

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = term_direct_gene_map

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(term_size_map)

        # ngenes, gene_dim are the number of all genes  
        self.gene_dim = ngene
        self.drug_dim = ndrug

        # add modules for neural networks to process genotypes
        self.contruct_direct_gene_layer()
        self.construct_NN_graph(dG)

        # add modules for neural networks to process drugs  
        self.construct_NN_drug()

        # add modules for final layer
        if self.final_concat or (self.gene_fuse_position == 6 and self.fuse_add_drug_embedding):
            final_input_size = num_hiddens_genotype + num_hiddens_drug[-1]
        elif self.gene_fuse_position == 6 and self.fusion_function == "tensor_product_full":
            final_input_size = num_hiddens_genotype * num_hiddens_drug[-1]
        elif self.gene_fuse_position == 6 and self.fusion_function == "tensor_product_partial":
            final_input_size = num_hiddens_genotype * fuse_hyperparameters["aux_size"]
        else:
            final_input_size = num_hiddens_genotype

        self.add_module('final_linear_layer', nn.Linear(final_input_size, num_hiddens_final))
        self.add_module('final_batchnorm_layer', nn.BatchNorm1d(num_hiddens_final))
        self.add_module('final_aux_linear_layer', nn.Linear(num_hiddens_final, 1))
        self.add_module('final_linear_layer_output', nn.Linear(1, 1))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            # print("term\t%s\tterm_size\t%d\tnum_hiddens\t%d" % (term, term_size, num_output))
            self.term_dim_map[term] = num_output

    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)

            # if there are some genes directly annotated with the term, add a layer taking in all genes and forwarding out only those genes         
            self.add_module(term + '_direct_gene_layer', nn.Linear(self.gene_dim, len(gene_set), bias=False))

    # add modules for fully connected neural networks for drug processing
    def construct_NN_drug(self):
        input_size = self.drug_dim

        print(self.num_hiddens_drug)
        for i in range(len(self.num_hiddens_drug)):
            self.add_module('drug_linear_layer_' + str(i + 1), nn.Linear(input_size, self.num_hiddens_drug[i]))
            self.add_module('drug_batchnorm_layer_' + str(i + 1), nn.BatchNorm1d(self.num_hiddens_drug[i]))
            self.add_module('drug_aux_linear_layer1_' + str(i + 1), nn.Linear(self.num_hiddens_drug[i], 1))
            self.add_module('drug_aux_linear_layer2_' + str(i + 1), nn.Linear(1, 1))

            input_size = self.num_hiddens_drug[i]

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term   
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        layer_i = 0
        while True:
            # leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
            leaves = [n for n, d in dG.out_degree() if d == 0]
            # leaves = [n for n,d in dG.out_degree() if d==0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                if layer_i == self.gene_fuse_position:
                    # TODO: make it more dynamic to drug embedding size
                    self.add_module(term + '_GO_fuse_layer', FUSION_FUNCTIONS[self.fusion_function](
                        gene_dim=input_size,
                        drug_dim=self.num_hiddens_drug[-1],
                        device=self.device,
                        **self.fuse_hyperparameters
                    ))
                    if self.fuse_add_drug_embedding:
                        input_size += self.num_hiddens_drug[-1]
                    elif self.fusion_function == "tensor_product_partial":
                        input_size = input_size * self.fuse_hyperparameters.get("aux_size")

                self.add_module(term + '_GO_linear_layer', nn.Linear(input_size, term_hidden, bias=False))
                self.add_module(term + '_GO_batchnorm_layer', nn.BatchNorm1d(term_hidden))
                self.add_module(term + '_GO_aux_linear_layer1', nn.Linear(term_hidden, 1))
                self.add_module(term + '_GO_aux_linear_layer2', nn.Linear(1, 1))

            dG.remove_nodes_from(leaves)
            layer_i += 1

    # definition of forward function
    def forward(self, cuda_cell_features, cuda_drug_features):
        term_NN_out_map = {}
        aux_out_map = {}

        # define forward function for drug dcell #################################################
        drug_out = Variable(cuda_drug_features.to(self.device))

        for i in range(1, len(self.num_hiddens_drug) + 1, 1):
            drug_out = self._modules['drug_batchnorm_layer_' + str(i)](
                torch.tanh(self._modules['drug_linear_layer_' + str(i)](drug_out)))
            term_NN_out_map['drug_' + str(i)] = drug_out

            aux_layer1_out = torch.tanh(self._modules['drug_aux_linear_layer1_' + str(i)](drug_out))
            aux_out_map['drug_' + str(i)] = self._modules['drug_aux_linear_layer2_' + str(i)](aux_layer1_out)

        # define forward function for genotype dcell #############################################
        gene_input = Variable(cuda_cell_features.to(self.device))
        term_gene_out_map = {}

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        del gene_input
        torch.cuda.empty_cache()

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []

                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)
                # TODO: add fuse function here
                if self.gene_fuse_position == i:
                    print(f"Fusing drug embeddings for term {term}")
                    child_input = self._modules[term + '_GO_fuse_layer'](child_input, drug_out)

                term_NN_out = self._modules[term + '_GO_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term + '_GO_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term + '_GO_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_GO_aux_linear_layer2'](aux_layer1_out)

                # connect two neural networks at the top #################################################
        if self.gene_fuse_position == 6:
            final_input = self._modules["FUSE_gene_position_6"](term_NN_out_map[self.root], drug_out)
        else:
            if not self.final_concat:
                final_input = term_NN_out_map[self.root]
            else:
                final_input = torch.concat([term_NN_out_map[self.root], drug_out], dim=1)

        out = self._modules['final_batchnorm_layer'](torch.tanh(self._modules['final_linear_layer'](final_input)))
        term_NN_out_map['final'] = out

        aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](out))
        aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out)

        del drug_out
        torch.cuda.empty_cache()

        if self.return_hidden:
            return aux_out_map, term_NN_out_map
        else:
            return aux_out_map["final"]
