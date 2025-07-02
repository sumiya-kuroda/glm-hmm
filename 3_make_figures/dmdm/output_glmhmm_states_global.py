#!/usr/bin/env python
import defopt
import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.append('../../2_fit_models/dmdm')
from data_io import get_file_dir, load_animal_list
from plotting_utils import load_global_glmhmm_result, infer_global_glmhmm_result, flatten_list

sys.path.append('../../1_preprocess_data/dmdm')
from preprocessing_utils import load_animal_eid_dict

def main(*, dname: str, model: str, K:int, regularization: str, suffix: str = None):
    """
    Begin processing dmdm data: identify animals in dataset 
    that enter biased blocks and list their session ids.
    
    :param str dname: name of dataset needs to be preprocessed
    :param str model: name of GLM-HMM model
    :param int K: number of states
    :param str regularization: specify regularization method. If not using regularization, specify None
    :param str suffix: suffix of saving location if any
    """

    print('Saving global GLMHMM results ...')
    data_dir =  get_file_dir().parents[1] / "data" / "dmdm" / dname / 'data_for_cluster'
    figure_dir = get_file_dir().parents[1] / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    if regularization == 'None':
        regularization = None # change dtype
    if regularization is None:
        is_regularized = 'None'
    else:
        is_regularized = regularization

    if suffix is None:
        inference_input = dname
    else:
        inference_input = dname + '_' + suffix
    print('dname: {}; model: {}; regularization: {}, state_num: {}; data_input {}'.format(dname, model, is_regularized, K, inference_input))

    if suffix is None:
        # Simply Load GLM-HMM results
        states_max_posterior, posterior_probs, _, inpt_rt, _, session, _, _, mask, hmm_params \
            = load_global_glmhmm_result(K, model, data_dir, regularization)
        
        # Load raw data eids
        animal_list = load_animal_list(
            data_dir / 'data_by_animal' / 'animal_list.npz')
        animal_eid_dict = load_animal_eid_dict(
            data_dir / 'final_animal_eid_dict.json')

        # Create trial IDs
        trialnum_all = []
        for z, animal in enumerate(animal_list):
            sess_counter = 0
            for eid in animal_eid_dict[animal]:

                raw_session_id = eid.split('Subjects/')[1]
                # Get animal:
                animal = raw_session_id.split('/')[0]
                # Replace '/' with dash in session ID
                session_id = raw_session_id.replace('/', '-')

                trialnum = np.load(data_dir.parent / Path(eid) / '_dmdm_trials.trial.npy')[0]
                trialnum2 = [session_id + '-' + str(int(t)) for t in trialnum]
                trialnum_all.append(trialnum2)
    else:
         # Infer GLM-HMM states
        data_dir_inference =  get_file_dir().parents[1] / "data" / "dmdm" / inference_input / 'data_for_cluster'
        states_max_posterior, posterior_probs, _, inpt_rt, _, session, _, _, mask, hmm_params \
            = infer_global_glmhmm_result(K, model, data_dir, data_dir_inference, regularization)

        # Load raw data eids
        animal_list = load_animal_list(
            data_dir_inference / 'data_by_animal' / 'animal_list.npz')
        animal_eid_dict = load_animal_eid_dict(
            data_dir_inference / 'final_animal_eid_dict.json')
        
        # Create trial IDs
        trialnum_all = []
        for z, animal in enumerate(animal_list):
            sess_counter = 0
            for eid in animal_eid_dict[animal]:

                raw_session_id = eid.split('Subjects/')[1]
                # Get animal:
                animal = raw_session_id.split('/')[0]
                # Replace '/' with dash in session ID
                session_id = raw_session_id.replace('/', '-')

                trialnum = np.load(data_dir_inference.parent / Path(eid) / '_dmdm_trials.trial.npy')[0]
                trialnum2 = [session_id + '-' + str(int(t)) for t in trialnum]
                trialnum_all.append(trialnum2)
        
    # Save outputs in json file
    rdict = defaultdict(list)
    for zk in range(K):
        b = np.where(states_max_posterior == zk)
        x = np.array(flatten_list(trialnum_all))[b]
        rdict[zk] = x.tolist()
 
    probdict = dict(zip(flatten_list(trialnum_all), posterior_probs.tolist()))

    out_json = json.dumps(rdict)
    prob_json = json.dumps(probdict)

    if suffix is None:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}.json".format(dname, K, is_regularized)), "w")
    else:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}.json".format(dname, K, is_regularized, suffix)), "w")
    f.write(out_json)
    f.close()

    if suffix is None:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_posteriorprobs.json".format(dname, K, is_regularized)), "w")
    else:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}_posteriorprobs.json".format(dname, K, is_regularized, suffix)), "w")
    f.write(prob_json)
    f.close()

    # highprob
    rdict = defaultdict(list)
    highprobs = np.any(np.apply_along_axis(lambda a1: a1 >= 0.8, axis=1,arr=posterior_probs))
    for zk in range(K):
        b = np.where(np.logical_and(states_max_posterior == zk, highprobs))[0]
        x = np.array(flatten_list(trialnum_all))[b]
        rdict[zk] = x.tolist()

    out_json = json.dumps(rdict)
    if suffix is None:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_highprob.json".format(dname, K, is_regularized)), "w")
    else:
        f = open(str(figure_dir / "state_session_trial_{}_K_{}_{}_{}_highprob.json".format(dname, K, is_regularized, suffix)), "w")
    f.write(out_json)
    f.close()

if __name__ == "__main__":
    defopt.run(main)