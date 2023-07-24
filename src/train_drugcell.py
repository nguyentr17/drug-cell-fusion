import numpy as np
import torch.utils.data as du
import torch.nn as nn
from util import *
from drugcell_NN import FUSION_FUNCTIONS, drugcell_nn
import argparse
import datetime
import json
from typing import Dict

from torch.profiler import profile, record_function, ProfilerActivity


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):
    term_mask_map = {}

    for term, gene_set in term_direct_gene_map.items():

        mask = torch.zeros(len(gene_set), gene_dim)

        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.to(DEVICE))

        term_mask_map[term] = mask_gpu

    return term_mask_map


# solution for ||x-y||^2_2 + c||x||_0
def proximal_l0(yvec, c):
    yvec_abs = torch.abs(yvec)
    csqrt = torch.sqrt(c)

    xvec = (yvec_abs >= csqrt) * yvec
    return xvec


# solution for ||x-y||^2_2 + c||x||_g
def proximal_glasso_nonoverlap(yvec, c):
    ynorm = torch.linalg.norm(yvec, ord='fro')
    if ynorm > c / 2.:
        xvec = (yvec / ynorm) * (ynorm - c / 2.)
    else:
        xvec = torch.zeros_like(yvec)
    return xvec


# solution for ||x-y||^2_2 + c||x||_2^2
def proximal_l2(yvec, c):
    return (1. / (1. + c)) * yvec


# prune the structure by palm
def optimize_palm(model, dG, root, reg_l0, reg_glasso, reg_decay, lr=0.001, lip=0.001):
    dG_prune = dG.copy()
    for name, param in model.named_parameters():
        if "direct" in name:
            # mutation side
            # l0 for direct edge from gene to term
            param_tmp = param.data - lip * param.grad.data
            param.data = proximal_l0(param_tmp, reg_l0)
        elif "GO_linear_layer" in name:
            # group lasso for
            term_name = name.split('_')[0]
            child = model.term_neighbor_map[term_name]
            dim = 0
            for i in range(len(child)):
                dim = model.num_hiddens_genotype
                term_input = param.data[:, i * dim:(i + 1) * dim]
                term_input_grad = param.grad.data[:, i * dim:(i + 1) * dim]
                term_input_tmp = term_input - lip * term_input_grad
                term_input_update = proximal_glasso_nonoverlap(term_input_tmp, reg_glasso)
                param.data[:, i * dim:(i + 1) * dim] = term_input_update
                num_n0 = torch.count_nonzero(term_input_update)
                if num_n0 == 0:
                    dG_prune.remove_edge(term_name, child[i])
            # TODO: What if the go term has no child? do we need to calculate the weight decay for it
            # weight decay for direct
            direct_input = param.data[:, len(child) * dim:]
            direct_input_grad = param.grad.data[:, len(child) * dim:]
            direct_input_tmp = direct_input - lr * direct_input_grad
            direct_input_update = proximal_l2(direct_input_tmp, reg_decay)
            param.data[:, len(child) * dim:] = direct_input_update
        else:
            # other param weigth decay
            param_tmp = param.data - lr * param.grad.data
            param.data = proximal_l2(param_tmp, reg_decay)
    sub_dG_prune = dG_prune.subgraph(nx.shortest_path(dG_prune.to_undirected(), root))
    print("Original graph has %d nodes and %d edges" % (dG.number_of_nodes(), dG.number_of_edges()))
    print("Pruned   graph has %d nodes and %d edges" % (sub_dG_prune.number_of_nodes(), sub_dG_prune.number_of_edges()))


# train a DrugCell model 
def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder,
                train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final,
                cell_features, drug_features,
                fuse_function: str = "concat",
                num_workers: int = 0,
                fuse_hyperparameters: Dict = {},
                gene_fuse_position: int = 6,
                final_concat: bool = False,
                pretrained_model_state_dict: str = None,
                train_after_fusion_only: bool = False
                ):
    '''
    # arguments:
    # 1) root: the root of the hierarchy embedded in one side of the model
    # 2) term_size_map: dictionary mapping the name of subsystem in the hierarchy to the number of genes contained in the subsystem
    # 3) term_direct_gene_map: dictionary mapping each subsystem in the hierarchy to the set of genes directly contained in the subsystem (i.e., children subsystems would not have the genes in the set)
    # 4) dG: the hierarchy loaded as a networkx DiGraph object
    # 5) train_data: torch Tensor object containing training data (features and labels)
    # 6) gene_dim: the size of input vector for the genomic side of neural network (visible neural network) embedding cell features 
    # 7) drug_dim: the size of input vector for the fully-connected neural network embedding drug structure 
    # 8) model_save_folder: the location where the trained model will be saved
    # 9) train_epochs: the maximum number of epochs to run during the training phase
    # 10) batch_size: the number of data points that the model will see at each iteration during training phase (i.e., #training_data_points < #iterations x batch_size)
    # 11) learning_rate: learning rate of the model training
    # 12) num_hiddens_genotype: number of neurons assigned to each subsystem in the hierarchy
    # 13) num_hiddens_drugs: number of neurons assigned to the fully-connected neural network embedding drug structure - one string containing number of neurons at each layer delimited by comma(,) (i.e. for 3 layer of fully-connected neural network containing 100, 50, 20 neurons from bottom - '100,50,20')
    # 14) num_hiddens_final: number of neurons assigned to the fully-connected neural network combining the genomic side with the drug side. Same format as 13).
    # 15) cell_features: a list containing the features of each cell line in tranining data. The index should match with cell2id list.
    # 16) drug_features: a list containing the morgan fingerprint (or other embedding) of each drug in training data. The index should match with drug2id list.
    '''

    # initialization of variables
    max_train_corr = 0
    max_test_corr = 0
    max_test_mse = 9999

    # dcell neural network
    model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype,
                        num_hiddens_drug, num_hiddens_final, DEVICE, fusion_function=fuse_function, fuse_hyperparameters=fuse_hyperparameters,
                        gene_fuse_position=gene_fuse_position, final_concat=final_concat)
    # load model to GPU
    model = model.to(DEVICE)
    params_to_train = model.parameters()
    if pretrained_model_state_dict is not None:
        model = model.load_state_dict(torch.load(pretrained_model_state_dict, map_location=DEVICE))
        if train_after_fusion_only:
            # Make other parameters not require grad
            params_to_train = []
            for param_name, param in model.named_parameters():
                if param_name.startswith("final") or param_name.startswith("FUSE"):
                    params_to_train.append(param)
                else:
                    param.requires_grad = False

    # separate the whole data into training and test data
    train_feature, train_label, test_feature, test_label = train_data

    # copy labels (observation) to GPU - will be used to 
    train_label_gpu = torch.autograd.Variable(train_label.to(DEVICE))
    test_label_gpu = torch.autograd.Variable(test_label.to(DEVICE))

    # create a torch objects containing input features for cell lines and drugs
    cuda_cells = torch.from_numpy(cell_features)
    cuda_drugs = torch.from_numpy(drug_features)



    # define optimizer
    # optimize drug NN
    optimizer = torch.optim.Adam(params_to_train, lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()
    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_direct_gene_layer.weight' in name:
            # print(name, param.size(), term_mask_map[term_name].size())
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
            # param.data = torch.mul(param.data, term_mask_map[term_name])
        else:
            param.data = param.data * 0.1

    # create dataloader for training/test data
    train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = du.DataLoader(du.TensorDataset(test_feature, test_label), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    #TODO: add profiler
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("training_function"):
            for epoch in range(train_epochs):

                # Train
                model.train()
                train_predict = torch.zeros(0, 0).to(DEVICE)

                for i, (inputdata, labels) in enumerate(train_loader):

                    cuda_labels = torch.autograd.Variable(labels.to(DEVICE))

                    # Forward + Backward + Optimize
                    optimizer.zero_grad()  # zero the gradient buffer

                    cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                    cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)

                    # Here term_NN_out_map is a dictionary
                    aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

                    if train_predict.size()[0] == 0:
                        train_predict = aux_out_map['final'].data
                    else:
                        train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

                    total_loss = 0
                    for name, output in aux_out_map.items():
                        loss = nn.MSELoss()
                        if name == 'final':
                            total_loss += loss(output, cuda_labels)
                        else:  # change 0.2 to smaller one for big terms
                            total_loss += 0.2 * loss(output, cuda_labels)

                    total_loss.backward()

                    for name, param in model.named_parameters():
                        if '_direct_gene_layer.weight' not in name:
                            continue
                        term_name = name.split('_')[0]
                        # print(name, param.grad.data.size(), term_mask_map[term_name].size())
                        param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

                    # TODO: use palm gives errors currently
                    # optimize_palm(model, dG, root, reg_l0=torch.tensor(0.0001), reg_glasso=torch.tensor(0.0001), reg_decay=0.0001, lr=0.001, lip=0.001)
                    optimizer.step()

                train_corr = pearson_corr(train_predict, train_label_gpu)

                model.eval()
                test_predict = torch.zeros(0, 0).to(DEVICE)

                for i, (inputdata, labels) in enumerate(test_loader):
                    # Convert torch tensor to Variable
                    cuda_cell_features = build_input_vector(inputdata.narrow(1, 0, 1).tolist(), gene_dim, cuda_cells)
                    cuda_drug_features = build_input_vector(inputdata.narrow(1, 1, 1).tolist(), drug_dim, cuda_drugs)

                    aux_out_map, _ = model(cuda_cell_features, cuda_drug_features)

                    if test_predict.size()[0] == 0:
                        test_predict = aux_out_map['final'].data
                    else:
                        test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

                test_corr = pearson_corr(test_predict, test_label_gpu).cpu().detach().numpy()
                test_mse = nn.MSELoss()(test_predict, test_label_gpu).cpu().detach().numpy()

                print("Epoch\t%d\ttrain_corr\t%.6f\ttest_corr\t%.6f\ttotal_loss\t%.6f\ttest_mse\t%.6f" % (
                epoch, train_corr, test_corr, total_loss, test_mse))

                model_file_path = f"{model_save_folder}/model_final_state_dict_{fuse_function}_gene_fuse_position_{gene_fuse_position}"
                for hp, hp_value in fuse_hyperparameters.items():
                    model_file_path = f"{model_file_path}_{hp}_{hp_value}"

                if test_corr >= max_test_corr:
                    max_test_corr = test_corr
                    max_train_corr = train_corr.cpu().detach().numpy()
                    max_test_mse = test_mse
                    torch.save(model.state_dict(), model_file_path)
                print(f"Best model performance is: {max_test_corr}")
    print(prof.key_averages().table(sort_by="cpu_time_total",row_limit=10))
    total_params = count_params(model)
    return max_train_corr, max_test_corr, max_test_mse, total_params


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
parser.add_argument('-train', help='Training dataset', type=str)
parser.add_argument('-test', help='Validation dataset', type=str)
parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=3000)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-cpu', action="store_true", default=False)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)

parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts',
                    type=int, default=3)
parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str,
                    default='100,50,3')
parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=3)

parser.add_argument('-cellline', help='Mutation information for cell lines', type=str)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)
parser.add_argument('-fuse_function', help='Name of fusing function', type=str, nargs="+")
parser.add_argument('-gene_fuse_position', type=int, default=6)
parser.add_argument('-final_concat', action="store_true", default=False)
parser.add_argument('-out', help='Result output file')
parser.add_argument('-experiment', action="store_true", default=False)
parser.add_argument('-num_workers', type=int, default=0)

print("Start....")

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=3)

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.test, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)
print('Total number of genes = %d' % len(gene2id_mapping))

cell_features = np.genfromtxt(opt.cellline, delimiter=',')
cell_features = cell_features.take(list(gene2id_mapping.values()), axis=1)
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0, :])

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = opt.final_hiddens
#####################################


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
if opt.experiment:
    all_fuse_functions = FUSION_FUNCTIONS.keys()
else:
    all_fuse_functions = opt.fuse_function

for fuse_function in all_fuse_functions:
    dG_train = dG.copy()
    print(f"Training model using {fuse_function} fuse function")
    begin_time = datetime.datetime.now()
    # Define hyperparameters for each fuse functions
    # TODO: better to do hp search but right now too computationally expensive
    if fuse_function == "cross_attention":
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
    train_corr, test_corr, test_mse, num_params = train_model(root, term_size_map, term_direct_gene_map, dG_train, train_data,
                                                    num_genes, drug_dim, opt.modeldir, opt.epoch, opt.batchsize, opt.lr,
                                                    num_hiddens_genotype, num_hiddens_drug, num_hiddens_final,
                                                    cell_features, drug_features,
                                                    fuse_function=fuse_function,
                                                    num_workers=opt.num_workers,
                                                    fuse_hyperparameters=fuse_hyperparameters,
                                                    gene_fuse_position=opt.gene_fuse_position,
                                                    final_concat=opt.final_concat
                                                    )
    end_time = datetime.datetime.now()
    experiment_result = {
        "gene_fuse_position": opt.gene_fuse_position, # max_in_degree - in_degree chosen to fuse
        "drug_fuse_position": len(num_hiddens_drug), # fixed
        "final_concat": opt.final_concat,
        "fuse_function": fuse_function,
        "fuse_hyperparamters": str(fuse_hyperparameters),
        "total_params": num_params,
        "num_epoch": opt.epoch,
        "train_duration_s": (end_time - begin_time).total_seconds(),
        "train_correlation": float(train_corr),
        "test_correlation": float(test_corr),
        "test_mse": float(test_mse)
    }
    with open(opt.out, "a") as result_file:
        json.dump(experiment_result, result_file)
        result_file.write("\n")
