# This file includes methods that are called from the wrapper scripts or that
# require imports from the repo itself or that are only needed by one of those
# methods.

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sklearn.metrics
import configparser
import glob
import re
import shutil
import torch.optim as optim
import os
import sys

from numpy.linalg import cholesky as chol
from scipy.linalg import solve_triangular
from pathlib import Path
from sklearn.decomposition import PCA

from dca_plda import modules 
from dca_plda import data as ddata
from dca_plda.scores import IdMap, Key, Scores
from dca_plda import calibration 
from dca_plda.utils import *


sys.path.insert(0, '/home/mariel/dcaplda-repo/GCA-PLDA')
import gca_plda.gca_utils as gca


def mkdirp(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def setup_torch_and_cuda_vars(use_cuda):
    cudnn.benchmark = True
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device("cpu")
    return device




def load_data_dict(table, device, fixed_enrollment_ids=None, map_enrollment_ids_to_level1=None, fixed_enrollment_ids_level1=None):
    data_dict = dict()
    for line in open(table).readlines():
        f = line.strip().split()
        name, emb, key, emapf, tmapf = f[0:5]
        dur = f[5] if len(f) > 3 else None
        dataset = ddata.LabelledDataset(emb, dur, meta_is_dur_only=True, device=device)
        if fixed_enrollment_ids is not None:
            # Enrollment embeddings are part of the model, not provided in the data
            # Hence, emapf in this case should either be NONE, in which case, all
            # detectors are used, or some subset of fixed_enrollment_ids, in which
            # case only that subset is used
            emap = IdMap.load(emapf, fixed_enrollment_ids)
        else:
            emap = IdMap.load(emapf, dataset.get_ids())

        tmap = IdMap.load(tmapf, dataset.get_ids())
        # Load key file in the order in which the model ids were loaded in emap 
        # and tmap. This ensures that the scores and keys will be aligned.
        mask = np_to_torch(Key.load(key, emap.model_ids, tmap.model_ids).mask, device)

        if map_enrollment_ids_to_level1 is not None:
            # Derive the mask for level1 from the map between level1 and output fixed_enrollment_ids
            # and the output mask.
            mask_level1 = -1*torch.ones([len(fixed_enrollment_ids_level1), mask.shape[1]])
            for i, cluster in enumerate(fixed_enrollment_ids_level1):
                idxs = np.array([emap.model_ids.index(l) for l, c in map_enrollment_ids_to_level1.items() if c==cluster])
                mask_level1[i,torch.where(mask[idxs]==1)[1]] = 1
            emap_level1 = IdMap.load('NONE', fixed_enrollment_ids_level1)    
            #mask_level1 = np_to_torch(mask_level1, device)
        else:
            mask_level1 = None
            emap_level1 = None
        
        data_dict[name] = {'dataset': dataset, 'mask': mask, 'emap': emap, 'tmap': tmap, 
        'mask_level1': mask_level1, 'emap_level1': emap_level1}

    return data_dict


def train(data_table_file, n_pca):
    """ Takes as inputs:
    data_table_file: table contining enroll_ids, test_ids, key, score, emb1, emb2
    n_pca: number of dimensions to reduce embeddings """

    enroll_ids, test_ids, key, score, emb1, emb2 = gca.load_data_table(data_table_file)

    #print(enroll_ids)
    print(enroll_ids[0], test_ids[0], key[0], score[0], emb1[0], emb2[0])

     
    n_pca = int(n_pca)
    pca = PCA(n_components=n_pca)
 
    emb1_red = pca.fit_transform(emb1)
    emb2_red = pca.fit_transform(emb2)

    con = np.concatenate((emb1_red, emb2_red), axis=1)
    print(con[:5])
    #print(type(emb_concat))


        
    

def train2(model, trn_dataset, config, dev_table, out_dir, device, 
    seed=0, restart=False, debug=False, init_subset=None, print_min_loss=False):

    dev_data_dict = load_data_dict(dev_table, device, model.enrollment_classes)
        
    #param_names = [k for k, v in model.named_parameters()]
    #config['l2_reg_dict'] = expand_regexp_dict(param_names, config.l2_reg)
    #num_epochs = config.num_epochs

    embeddings, metadata, metamaps = trn_dataset.get_data_and_meta(init_subset)

    # For backward compatibility with older configs where the names were different, and the value
    # True corresponded to same_num_classes_per_dom_then_same_num_samples_per_class
    bbd  = config.get("balance_method_for_batches", config.get("balance_batches_by_domain", 'none'))
    bbdi = config.init_params.get("balance_method", config.init_params.get("balance_by_domain", 'none'))
    config["balance_method_for_batches"] = bbd  if bbd  is not True else 'same_num_classes_per_dom_then_same_num_samples_per_class'
    config.init_params["balance_method"] = bbdi if bbdi is not True else 'same_num_classes_per_dom_then_same_num_samples_per_class'
    
    # Create the trial loader
    embeddings, metadata, metamaps = trn_dataset.get_data_and_meta()
    loader = ddata.TrialLoader(embeddings, metadata, metamaps, device, seed=seed, batch_size=config.batch_size, num_batches=config.num_batches_per_epoch, 
                         balance_method=config.balance_method_for_batches, num_samples_per_class=config.num_samples_per_class)

    print('hola, llegue')
    print('classes_for_dom=', loader.classes_for_dom,'classi=', loader.classi)


    # Train the model
    #start_epoch = 1 if not restart else last_epoch+1
    #for epoch in range(start_epoch, num_epochs + 1):
    #    trn_loss = train_epoch(model, loader, optimizer, epoch, config, debug_dir=out_dir if debug else None)
        
    for batch_idx, (data, metadata) in enumerate(loader):
        print_batch('batches_file.txt', metadata, loader.metamaps, batch_idx)
        print(batch_idx)
        print((data, metadata))
        print(len(data))
        print(len(metadata['class_id']))
        print('ids_to_idxs=', loader.metamaps['class_id_inv'])



def load_model(file, device):

    loaded_dict = torch.load(file, map_location=device)
    config = loaded_dict['config']

    if 'in_dim' in loaded_dict:
        in_size = loaded_dict['in_dim']
    else:
        # Backward compatibility with old models that did not save in_dim
        if 'front_stage.W' in loaded_dict['model']:
            # If there is a front stage, get the in_size from there
            in_size = loaded_dict['model']['front_stage.W'].shape[0]
        elif 'lda_stage.W' in loaded_dict['model']:
            # Get it from the lda stage
            in_size = loaded_dict['model']['lda_stage.W'].shape[0]
        elif 'plda_stage.F' in loaded_dict['model']:
            in_size = loaded_dict['model']['plda_stage.F'].shape[0]
        else:
            raise Exception("Cannot infer input dimension for this model")

    if config.get("hierarchical"):
        model = modules.Hierarchical_DCA_PLDA_Backend(in_size, config)
    else:
        model = modules.DCA_PLDA_Backend(in_size, config)

    model.load_state_dict(loaded_dict['model'])

    return model

def print_batch(outf, metadata, maps, batch_num):

    metadata_str = dict()
    for k in metadata.keys():
        if k in ['sample_id', 'class_id', 'session_id', 'domain_id']:
            v = metadata[k].detach().cpu().numpy()
            metadata_str[k] = np.atleast_2d([maps[k][i] for i in v])

    batch_str = np.ones_like(metadata_str['sample_id'])
    batch_str[:] = str(batch_num)
    np.savetxt(outf, np.concatenate([batch_str] + list(metadata_str.values())).T, fmt="%s")


