import argparse
import os
import random
import torch
import re
import sys
sys.path.insert(0, '/home/mariel/dcaplda-repo/DCA-PLDA')

from dca_plda.utils import load_configs, get_class_to_cluster_map_from_config
from dca_plda.utils_for_scripts import setup_torch_and_cuda_vars, print_graph, mkdirp
from dca_plda.data import LabelledDataset
from dca_plda.modules import DCA_PLDA_Backend



#del nuevo rep
sys.path.insert(0, '/home/mariel/dcaplda-repo/GCA-PLDA')
from gca_plda.utils_for_scripts import train
import gca_plda.gca_utils as gca
import numpy as np

np.random.seed(0)

###run with : python3 -u train.py --trn_embeddings '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/embeddings_test.h5' --trn_key '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/keys/key_total.h5' --trn_scores '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/output/voxaccent-s4/dplda/stage1/eval_best/voxceleb2_4ch_16sec/scores.h5' --data_dir './data_table.h5' --n_tar 5000 --n_non 20000

default_config = {
    'architecture':{
        'lda_dim': 200},
    'training': {
        'loss': 'cross_entropy',
        'ptar': 0.01,
        'max_norm': 4,
        'l2_reg': 0.0001,
        'learning_rate': 0.0005,
        'betas': (0.5, 0.99),
        'learning_rate_params': None,
        'init_params': {'w_init': 0.5},
        'batch_size': 256,
        'num_epochs': 50,
        'num_samples_per_class': 2,
        'num_batches_per_epoch': 1000,
        'compute_ave_model': False}}

parser = argparse.ArgumentParser()
#parser.add_argument('--debug',        help='Enable debug mode.', action='store_true')
#parser.add_argument('--cuda',         help='Enable cuda.', action='store_true')
#parser.add_argument('--seed',         help='Seed used for training.', default=0, type=int)
#parser.add_argument('--configs',      help='List of configuration files to load. They are loaded in order, from left to right, overwriting previous values for repeated parameters.', default=None)
#parser.add_argument('--mods',         help='List of values to overwride config parameters. Example: training.num_epochs=20,architecture.lda_dim=200', default=None)
#parser.add_argument('--init_subset',  help='Subset of the train files to be used for initialization. For default, the files in trn_metafile are used.', default=None)
#parser.add_argument('--restart',      help='Restart training from last available model.', action='store_true')
#parser.add_argument('--print_min_loss', help='Print the min loss for each dev set at each epoch.', action='store_true')
parser.add_argument('--trn_embeddings', help='Path to the npz file with training embeddings.')
parser.add_argument('--trn_key',        help='Path to the npz file with training keys.')
parser.add_argument('--trn_scores',     help='Path to the npz file with training scores.')
parser.add_argument('--out_dir',       help='Output directory to the data table.')
parser.add_argument('--n_tar',          help='Number of target trials to sample.')
parser.add_argument('--n_non',          help='Number of non target trials to sample.')
parser.add_argument('--n_pca',          help='Number of PCA dimensions.')
#parser.add_argument('trn_metafile',   help='Path to the metadata for the training samples (all samples listed in this file should be present in the embeddings file).')
#parser.add_argument('dev_table',      help='Path to a table with one dev set per line, including: name, npz file with embeddings, key file, and durations file (can be missing if not using duration-dependent calibration).')
#parser.add_argument('--out_dir',        help='Output directory for models.')

opt = parser.parse_args()
#mkdirp(opt.out_dir)

##### Read the configs
#config = load_configs(opt.configs, default_config, opt.mods, "%s/config"%(opt.out_dir))

##### Set the seed 
#opt.seed = random.randint(1, 10000) if opt.seed is None else opt.seed
#print("Using seed: ", opt.seed)
#random.seed(opt.seed)
#torch.manual_seed(opt.seed)

##### Set the device and default data type
#device = setup_torch_and_cuda_vars(opt.cuda)

###### Load the dataset and create the model object
#cluster_ids = get_class_to_cluster_map_from_config(config.architecture)
#trn_dataset = LabelledDataset(opt.trn_embeddings, opt.trn_metafile)
#in_size = trn_dataset[0]['emb'].shape[0]

#model = DCA_PLDA_Backend(in_size, config.architecture).to(device)

#print_graph(model, trn_dataset, device, opt.out_dir)

gca.create_data_table(opt.trn_key, opt.trn_embeddings, opt.trn_scores, opt.out_dir, opt.n_tar, opt.n_non)



###### Train
print("\n####################################################################################")
print("Starting training")

train(opt.out_dir, opt.n_pca)
#train(model, trn_dataset, config.training, opt.dev_table, opt.out_dir, 
#    device, opt.seed, opt.restart, opt.debug, opt.init_subset, opt.print_min_loss)

