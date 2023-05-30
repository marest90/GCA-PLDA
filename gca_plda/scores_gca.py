import sys
import numpy
from random import sample, shuffle
import h5py
sys.path.insert(0, '/home/mariel/dcaplda-repo/DCA-PLDA')

from dca_plda import scores

def keys_to_list(keys):

    keys_inmem = scores.Key.load(keys)
    enrollids = keys_inmem.enroll_ids
    testids = keys_inmem.test_ids
    maski = keys_inmem.mask

    #print(enrollids)
    #print(testids)
    #print(maski)

    key = []
    key_test = []
    key_enroll = []

    for (i_nt, nt) in enumerate(testids):
        for (i_nT, nT) in enumerate(enrollids):
            m =  maski[i_nT, i_nt]
            if m != 0:
                lab = 1 if m==1 else 0
                key.append(lab)
                key_test.append(nt)
                key_enroll.append(nT)

                #f.write("%s %s %s\n" % (nT, nt, lab))
                #print(nT, nt, lab)

    return key_test, key_enroll, key

############Defino funcion para borrar elementos de lista usando una lista de indices
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

keys_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/keys/key_total.h5'

#key_test, key_enroll, key = keys_to_list(keys_file)


def sample_key(keys_file, n_tar, n_non):
    key_test, key_enroll, key = keys_to_list(keys_file)

    tar_ind = list(numpy.nonzero(key)[0])
    non_ind = [i for i, n in enumerate(key) if n == 0]

    sample_tar_ind = sample(tar_ind,n_tar)
    sample_non_ind = sample(non_ind,n_non)


    key_tar = [key[i] for i in sample_tar_ind]
    key_non = [key[i] for i in sample_non_ind]


    test_key_tar = [key_test[i] for i in sample_tar_ind]
    enroll_key_tar = [key_enroll[i] for i in sample_tar_ind]

    test_key_non = [key_test[i] for i in sample_non_ind]
    enroll_key_non = [key_enroll[i] for i in sample_non_ind]

    test = test_key_tar + test_key_non
    enroll = enroll_key_tar + enroll_key_non
    key_tot = key_tar + key_non

    zip_list = list(zip(test, enroll, key_tot))
    shuffle(zip_list)

    test_f, enroll_f, key_f = zip(*zip_list)

    return test_f, enroll_f, key_f

###########################################################################################################

def load_embeddings(emb_file):
    """ Loads embeddings file in npz """

    if emb_file.endswith(".npz"):
        data_dict = numpy.load(emb_file)
    elif emb_file.endswith(".h5"):
        with h5py.File(emb_file, 'r') as f:
            #ids_decode = [i.decode('UTF-8') for i in f['ids'][()]]
            data_dict = {'ids': f['ids'][()], 'data': f['data'][()]}
    else:
        raise Exception("Unrecognized format for embeddings file %s"%emb_file)

    print("  Done loading embeddings")
    embeddings_all = data_dict['data']
    if type(data_dict['ids'][0]) == numpy.bytes_:
        ids_all = [i.decode('UTF-8') for i in data_dict['ids']]
    elif type(data_dict['ids'][0]) == numpy.str_:
        ids_all = data_dict['ids']
    else:
        raise Exception("Bad format for ids in embeddings file %s (should be strings)"%emb_file)

    return ids_all, embeddings_all

###########################################################################################################


n_tar = 5000
n_non = 20000

test_f, enroll_f, key_f = sample_key(keys_file, n_tar, n_non)

print('key=',test_f[0])

emb_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/embeddings_test.h5'

emb_ids, emb_data = load_embeddings(emb_file)

print('emb=', emb_ids[0])

emb_dict = {emb_ids[i]: emb_data[i] for i in range(len(emb_ids))}

emb1 = [emb_dict[k] for k in test_f]
emb2 = [emb_dict[k] for k in enroll_f]


scores_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/output/voxaccent-s4/dplda/stage1/eval_best/voxceleb2_4ch_16sec/scores.h5'

scores_obj = scores.Scores.load(scores_file) #contains enroll_ids, test_ids, score_mat

print(scores_obj.score_mat[1][2], scores_obj.enroll_ids[1], scores_obj.test_ids[2])
print(scores_obj.enroll_ids[:100])
print(scores_obj.test_ids[:100])


scores_dict = {scores_obj.enroll_ids[i]:{scores_obj.test_ids[j]:scores_obj.score_mat[i][j]  for j in range(len(scores_obj.test_ids))} for i in range(len(scores_obj.enroll_ids))}

print(scores_dict[enroll_f[0]][test_f[1]])
