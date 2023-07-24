import shap

import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from drugcell_NN import *
import argparse


def get_feature_importance(root, term_size_map, term_direct_gene_map, dG, predict_data, gene_dim, drug_dim, batch_size, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, model_file, hidden_folder, result_file, cell_features, drug_features,
                  fuse_function: str = "concat",
                  num_workers: int = 0,
                  fuse_hyperparameters: Dict = {},
                  gene_fuse_position: int = 6,
                  final_concat: bool = False):
    # Initialize the model
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype,
                        num_hiddens_drug, num_hiddens_final, DEVICE, fusion_function=fuse_function, fuse_hyperparameters=fuse_hyperparameters,
                        gene_fuse_position=gene_fuse_position, final_concat=final_concat,
                        return_hidden=False)
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(model_file, map_location=DEVICE))

    #model.eval()

    predict_feature, predict_label = predict_data
    predict_label_gpu = predict_label.to(DEVICE)

    cuda_cells = torch.from_numpy(cell_features)
    cuda_drugs = torch.from_numpy(drug_features)

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #Test
    test_predict = torch.zeros(0,0).to(DEVICE)
    term_hidden_map = {}

    batch_num = 0
    shap_score_all = []
    for i, (inputdata, labels) in enumerate(test_loader):
        # Convert torch tensor to Variable
        cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
        cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)

        e = shap.DeepExplainer(model, [cuda_cell_features[5:], cuda_drug_features[5:]])

        # make prediction for test data
        # TODO: currently all returns zeros, need to change the background
        # Read https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#nlp_model

        shap_score = e.shap_values(X=[cuda_cell_features[0:5], cuda_drug_features[0:5]])[0]
        shap_score_mean = np.abs(shap_score).mean(axis=0)
        print(f"Min: {np.min(shap_score_mean)}")
        print(f"Max: {np.max(shap_score_mean)}")
        shap_score_all.append(shap_score)
        batch_num += 1

    shap_score_all = torch.cat(shap_score_all, dim=0)
    shap_score_mean = torch.mean(shap_score_all, dim=0)
    print(f"Min: {torch.min(shap_score_mean)}")
    print(f"Max: {torch.max(shap_score_mean)}")


parser = argparse.ArgumentParser(description='DCell prediction')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=1000)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default=1000)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default=1000)
parser.add_argument('-load', help='Model file', type=str, default='MODEL/model_200')
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
parser.add_argument('-result', help='Result file name', type=str, default='Result/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-cellline', help='Mutation information for cell lines', type=str)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)

parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=3)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,3')
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=3)
parser.add_argument('-fuse_function', help='Name of fusing function', type=str, nargs="+")
parser.add_argument('-gene_fuse_position', type=int, default=6)
parser.add_argument('-final_concat', action="store_true", default=False)
parser.add_argument('-cpu', action="store_true", default=False)

opt = parser.parse_args()
torch.set_printoptions(precision=5)
CUDA_ID = opt.cuda
if opt.cpu:
    DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda", CUDA_ID)
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device type: {DEVICE.type}")

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell features
cell_features = np.genfromtxt(opt.cellline, delimiter=',')

# load drug features
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])


drug_dim = len(drug_features[0,:])
num_genes = len(gene2id_mapping)

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = opt.final_hiddens
#####################################


print("Total number of genes = %d" % len(gene2id_mapping))
fuse_function = opt.fuse_function[0]
if fuse_function == "hyperfuse":
    fuse_hyperparameters = {
        "mlp_type": "c",
        "hidden_size": 8
    }
elif fuse_function == "hyperfuse_concat":
    fuse_hyperparameters = {
        "mlp_type": "b",
        "hidden_size": 8
    }
elif fuse_function == "shortfuse":
    fuse_hyperparameters = {
        "aux_dropout_rate": 0.5
    }
elif fuse_function == "cross_attention":
    fuse_hyperparameters = {
        "gene_attention_size": 6,
        "drug_attention_size": 6,
        "temperature": 1.0,
        "activation_fn": nn.Tanh()
    }
elif fuse_function == "tensor_product_partial":
    fuse_hyperparameters = {
        "aux_size": 2,
    }
else:
    fuse_hyperparameters = {}

get_feature_importance(root=root, term_size_map=term_size_map,
              term_direct_gene_map=term_direct_gene_map, dG=dG, predict_data=predict_data,
              gene_dim=num_genes, drug_dim=drug_dim, batch_size=opt.batchsize,
              num_hiddens_genotype=num_hiddens_genotype, num_hiddens_drug=num_hiddens_drug, num_hiddens_final=num_hiddens_final,
              model_file=opt.load,
              hidden_folder=opt.hidden, result_file=opt.result,
              cell_features=cell_features, drug_features=drug_features,
              fuse_function=fuse_function,
              num_workers=0,
              fuse_hyperparameters=fuse_hyperparameters,
              gene_fuse_position=opt.gene_fuse_position,
              final_concat=opt.final_concat)


