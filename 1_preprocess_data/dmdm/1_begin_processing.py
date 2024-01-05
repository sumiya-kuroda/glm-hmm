#!/usr/bin/env python

import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import os
from preprocessing_utils import scan_sessions, get_animal_name
import defopt

def main(dname):
    """
    Begin processing dmdm data: identify animals in dataset 
    that enter biased blocks and list their session ids.
    
    :param str dname: name of dataset needs to be preprocessed
    """
    dirname = Path(os.path.dirname(os.path.abspath(__file__)))
    dmdm_data_path =  dirname.parents[1] / "data" / "dmdm" / dname
    if not dmdm_data_path.exists():
        raise FileNotFoundError

    os.chdir(str(dmdm_data_path))
    # Create directory for saving data:
    Path(Path.cwd() / "partially_processed").mkdir(parents=True, exist_ok=True)

    eids = scan_sessions('./Subjects/')
    assert len(eids) > 0, "sessions are saved in incorrect directory"
    print('Found {} sessions in total'.format(len(eids)))
    animal_list = []
    animal_eid_dict = defaultdict(list)

    # Find sessions with bias blocks
    for eid in eids:
        bias = np.load(Path(eid) / '_dmdm_trials.hazardblock.npy')[0]
        if ~np.isnan(bias).any(): # nan means that trial was neither early nor late
            animal = get_animal_name(eid)
            if animal not in animal_list:
                animal_list.append(animal)
            animal_eid_dict[animal].append(eid)

    # Save eids and animal names
    out_json = json.dumps(animal_eid_dict)
    f = open("partially_processed/animal_eid_dict.json",  "w")
    f.write(out_json)
    f.close()

    np.savez('partially_processed/animal_list.npz', animal_list)

if __name__ == "__main__":
    defopt.run(main)