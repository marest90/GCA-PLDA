import sys
import numpy
from random import sample, shuffle
import h5py
from os.path import exists

sys.path.insert(0, '/home/mariel/dcaplda-repo/DCA-PLDA')

from dca_plda import scores

###########################################################################################################
def keys_to_list(keys):

    """ Creates a list from the key mask """

    keys_inmem = scores.Key.load(keys)
    enrollids = keys_inmem.enroll_ids
    testids = keys_inmem.test_ids
    maski = keys_inmem.mask


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

###########################################################################################################
def delete_multiple_element(list_object, indices):
    """ Deletes elements from a list using a list of indexes """
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)



###########################################################################################################
def sample_key(keys_file, n_tar, n_non):
    """ Samples a list of key to a desire size, needs n_tar and n_non (int) to set the number of target and non target samples """

    print('Sampling trials, this might take a while')   
    n_tar = int(n_tar)
    n_non = int(n_non)


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
    print('Done sampling')

    return list(test_f), list(enroll_f), list(key_f)

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

    print("Loaded embedding file %s"%emb_file)
    embeddings_all = data_dict['data']
    if type(data_dict['ids'][0]) == numpy.bytes_:
        ids_all = [i.decode('UTF-8') for i in data_dict['ids']]
    elif type(data_dict['ids'][0]) == numpy.str_:
        ids_all = data_dict['ids']
    else:
        raise Exception("Bad format for ids in embeddings file %s (should be strings)"%emb_file)

    return ids_all, embeddings_all

###########################################################################################################
def create_scrlist(enroll_list, test_list, scores_dictionary):
    """ Creates a list with the scores related to one test id and one enroll id per line, using a dictionary containing scores """
    """ Produces a list of index of ids not present in the scores file """

    index = []
    scrs = []

    for i in range(len(enroll_list)):
        try:
            scrs.append(scores_dictionary[enroll_list[i]][test_list[i]])
        except:
            #print(enroll_list[i], test_list[i]) #print sessions not found in scores file
            index.append(i)

    if len(index) > 0:
        print('Session in key not present in scores file found. These sessions will be excluded from the table.')

    return scrs, index

###########################################################################################################
def save_table(enroll_list, test_list, key_list, scores_list, emb1, emb2,format,file_name):
    """ Saves the table to a file. If format='h5' it is saved to an h5 file and if format='txt' it is saved to a txt file except for the embeddings which are not written """

    print('Saving table...')
    # Writes the table to a txt file
    if format == 'txt':
        with open(file_name, 'w') as f:
            for i in range(len(test_list)):
                f.write(test_list[i] + ' ' + enroll_list[i] + ' ' + str(key_list[i]) + ' ' + str(scores_list[i])) 
                f.write('\n')

    # Writes the table in h5 format
    elif format == 'h5':
        with h5py.File(file_name,'w') as f:
            f.create_dataset('enroll_ids',   data=numpy.string_(enroll_list))
            f.create_dataset('test_ids',   data=numpy.string_(test_list))
            f.create_dataset('key',   data=key_list)
            f.create_dataset('score',   data=scores_list)
            f.create_dataset('embedding1',     data=emb1)
            f.create_dataset('embedding2',     data=emb2)

    print('Succesfully saved!')

###########################################################################################################
def create_data_table(keys_file, emb_file, scores_file, file_path, n_tar, n_non):
    """ Creates a table containing trials to use, including enroll_id, test_id, key,  score, enroll_embedding, test_embedding and saves it in txt or h5 format """

    if exists(file_path):
        print('Data table already exists, nothing to do')

    else:
        print('Creating data table, please wait')
        test_f, enroll_f, key_f = sample_key(keys_file, n_tar, n_non)
        emb_ids, emb_data = load_embeddings(emb_file)


        emb_dict = {emb_ids[i]: emb_data[i] for i in range(len(emb_ids))}
        emb1 = [emb_dict[k] for k in test_f]
        emb2 = [emb_dict[k] for k in enroll_f]

        scores_obj = scores.Scores.load(scores_file) #contains enroll_ids, test_ids, score_mat
        scores_dict = {scores_obj.enroll_ids[i]:{scores_obj.test_ids[j]:scores_obj.score_mat[i][j]  for j in range(len(scores_obj.test_ids))} for i in range(len(scores_obj.enroll_ids))}

        scrs, index = create_scrlist(enroll_f, test_f, scores_dict)

        #test_notinscr = [test_f[j] for j in index]
        #print(test_notinscr)

        delete_multiple_element(test_f, index)
        delete_multiple_element(enroll_f, index)
        delete_multiple_element(key_f, index)
        delete_multiple_element(emb1, index)
        delete_multiple_element(emb2, index)

        

        scrs = []
        for i in range(len(test_f)):
            scrs.append(scores_dict[enroll_f[i]][test_f[i]])

        save_table(enroll_f, test_f, key_f, scrs, emb1, emb2, 'h5',file_path)

###########################################################################################################
def load_data_table(filename):
    """ Build Key from a text file with the following format 
    * trainID testID tgt/imp for SID
    * testID languageID for LID
    or in h5 format where the enrollids are the languageids in the case of LID.
    """

    if filename.endswith(".h5"):
        with h5py.File(filename, 'r') as f:
            enroll_ids = f['enroll_ids'][()]
            test_ids   = f['test_ids'][()]
            key      = f['key'][()]
            score      = f['score'][()]
            emb1      = f['embedding1'][()]
            emb2      = f['embedding2'][()]

            if type(enroll_ids[0]) == numpy.bytes_:
                enroll_ids = [i.decode('UTF-8') for i in enroll_ids]
                test_ids   = [i.decode('UTF-8') for i in test_ids]

    return enroll_ids, test_ids, key, score, emb1, emb2






#n_tar = 5000
#n_non = 20000

#keys_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/keys/key_total.h5'
#emb_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/data/eval/voxceleb2_4ch_16sec/embeddings_test.h5'
#scores_file = '/home/mariel/dcaplda-repo/DCA-PLDA/examples/speaker_verification/output/voxaccent-s4/dplda/stage1/eval_best/voxceleb2_4ch_16sec/scores.h5'
#folder_path = './data_table.h5'

#create_data_table(keys_file, emb_file, scores_file, folder_path, n_tar, n_non)

#enroll_ids, test_ids, key, score, emb1, emb2 = load_data_table(folder_path)

#print(enroll_ids)
#print(enroll_ids[0], test_ids[0], key[0], score[0], emb1[0], emb2[0])

