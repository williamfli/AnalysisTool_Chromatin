import numpy as np
import multiprocessing as mp
import time
import pandas as pd
# functions
from itertools import combinations_with_replacement, permutations
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm
# shared parameters

default_num_threads = 12





# Function to calculate chromosome center
def get_chroZxys_mean_center (_ichr_zxys):
    x, y, z= np.nanmean(_ichr_zxys[:,1]),np.nanmean(_ichr_zxys[:,2]),np.nanmean(_ichr_zxys[:,0])
    _ichr_ct_zxy = np.array([x,y,z])
    return _ichr_ct_zxy


#############################################################################
# For a pair of chromosomes   
# Function to summarize Dist for chr pair meets following critera: 
# (1) trans-non-homolog_chr (2) chr centers within certain dist 
# optional: use dist between center to normalize 
def Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr(chr_2_zxys_list, _c1, _c2, codebook_df,
                                 function='nanmedian', axis=0, data_num_th=50 ,_center_dist_th = 3000,normalize_by_center=True,
                                 _contact_th=500, contact_prob=False,
                                 verbose=False):
    
    from scipy.spatial.distance import euclidean

    print(f'-- Start analyzing neighboring chromosome pair within {_center_dist_th}nm.')
    
    _out_dist_dict = {}
    if _c1 != _c2:
        _out_dist_dict[(_c1,_c2)] = []
    else:
        _out_dist_dict[f"cis_{_c1}"] = []
        _out_dist_dict[f"trans_{_c1}"] = []
        
    for _chr_2_zxys in chr_2_zxys_list:
        # skip if not all info exists
        if _c1 not in _chr_2_zxys or _c2 not in _chr_2_zxys or _chr_2_zxys[_c1] is None or _chr_2_zxys[_c2] is None:
            continue
        else:
            # if not from the same chr label, calcluate trans-chr with cdist
            if _c1 != _c2:
                # dual loop for each allel
                for _zxys1 in _chr_2_zxys[_c1]:
                    _center_zxys1 = get_chroZxys_mean_center(_zxys1)
                    for _zxys2 in _chr_2_zxys[_c2]:
                        _center_zxys2 = get_chroZxys_mean_center(_zxys2)
                        # only analyze neighboring chromosome pair within given distance
                        if euclidean(_center_zxys1,_center_zxys2)<= _center_dist_th:
                            _chr_pair_cdists = cdist(_zxys1, _zxys2)
                            if normalize_by_center and not contact_prob:
                                _chr_pair_cdists=_chr_pair_cdists/euclidean(_center_zxys1,_center_zxys2)
                            _out_dist_dict[(_c1,_c2)].append(_chr_pair_cdists)
                        else:
                            continue
            # if from the same chr label, calculate both cis and trans
            else:
                # skip homolog chromosome pairs
                continue

    for _key in _out_dist_dict:
        print(_key, len(_out_dist_dict[_key]))
    #return _out_dist_dict
    # summarize
    _summary_dict = {}
    all_chrs = [str(_chr) for _chr in np.unique(codebook_df['chr'])]
    all_chr_sizes = {_chr:np.sum(codebook_df['chr']==_chr) for _chr in all_chrs}
    # add nan for non-relevant or empty chr pairs
    for _key, _dists_list in _out_dist_dict.items():
        
        if len(_dists_list) >= 100: # discard if not enough cell statistics
            # summarize
            # no prior normalization if calculate contact prob
            if contact_prob:
                # get nan positions
                _nan_index = np.isnan(np.array(_dists_list))
                # th value comparision and restore nan
                _dists_contact_list=np.array(_dists_list)<=_contact_th
                # convert array as float type to enable nan re-assignment through indexing; 
                # otherwise all nan will be treated as True for bool
                _dists_contact_list=_dists_contact_list*1.0
                _dists_contact_list[_nan_index==1]=np.nan
                _summary_result = getattr(np, 'nanmean')(_dists_contact_list, axis=axis)
                # keep result if non-nan N > data number th
                _valid_index = np.sum(~np.isnan(_dists_contact_list),axis=0)>=data_num_th
                _summary_result[_valid_index==0]=np.nan
                _summary_dict[_key] = _summary_result

            else:
                _summary_result =getattr(np, function)(_dists_list, axis=axis)
                # keep result if non-nan N > data number th
                _valid_index = np.sum(~np.isnan(_dists_list),axis=0)>=data_num_th
                _summary_result[_valid_index==0]=np.nan
                _summary_dict[_key] = _summary_result
        else:
            if isinstance(_key, str): # cis or trans
                _chr = _key.split('_')[-1] 
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr], all_chr_sizes[_chr]])
            else:
                _chr1, _chr2 = _key
                _summary_dict[_key]= np.nan * np.ones([all_chr_sizes[_chr1], all_chr_sizes[_chr2]])
    
    return _summary_dict




    # call previous function to calculate all pair-wise chromosomal distance
def Chr2ZxysList_2_summaryDict_for_trans_neigh_chr(
    chr_2_zxys_list, total_codebook,
    function='nanmedian', axis=0,data_num_th=50,
    _center_dist_th = 3000,normalize_by_center=True,
     _contact_th=500, contact_prob=False,
    parallel=True, num_threads=default_num_threads,
    verbose=False):
    """Function to batch process chr_2_zxys_list into summary_dictionary"""
    if verbose:
        print(f"-- preparing chr_2_zxys from {len(chr_2_zxys_list)} cells", end=' ')
        _start_prepare = time.time()
    _summary_args = []
    # prepare args
    _all_chrs = np.unique(total_codebook['chr'].values)
    #sorted(_all_chrs, key=lambda _c:sort_mouse_chr(_c))
    for _chr1, _chr2 in combinations_with_replacement(_all_chrs, 2):
        if _chr1 != _chr2:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None),
                 _chr2: _d.get(_chr2, None)} for _d in chr_2_zxys_list
            ]
        else:
            _sel_chr_2_zxys = [
                {_chr1: _d.get(_chr1, None)} for _d in chr_2_zxys_list
            ]
        _summary_args.append(
            (_sel_chr_2_zxys, _chr1, _chr2, total_codebook, function, axis, data_num_th,
            _center_dist_th, normalize_by_center, _contact_th, contact_prob,verbose)
        )
    if verbose:
        print(f"in {time.time()-_start_prepare:.3f}s.")
    # process
    _start_time = time.time()
    if parallel:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances with {num_threads} threads", end=' ')
        with mp.Pool(num_threads) as _summary_pool:
            all_summary_dicts = _summary_pool.starmap(
                Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr, 
                _summary_args, chunksize=1)
            _summary_pool.close()
            _summary_pool.join()
            _summary_pool.terminate()
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    else:
        if verbose:
            print(f"-- summarize {len(_summary_args)} inter-chr distances sequentially", end=' ')
        all_summary_dicts = [Chr2ZxysList_2_summaryDist_by_key_for_trans_neigh_chr(*_args) for _args in _summary_args]
        if verbose:
            print(f"in {time.time()-_start_time:.3f}s.")
    # summarize into one dict
    _summary_dict = {}
    for _dict in all_summary_dicts:
        _summary_dict.update(_dict)
    return _summary_dict