import numpy as np
import multiprocessing as mp
import time
import pandas as pd
# functions
from itertools import combinations_with_replacement, permutations, combinations
from scipy.spatial.distance import pdist, squareform, cdist
from tqdm import tqdm



# Function to get normalize factor by re-calculate pairwise distance for all the loci
def normalize_total_distmap_loci_dataframe (im_loci_df, 
                                total_codebook,
                                class_2_chrZxysList,
                                overwrite = False,
                                norm_factor_dict = None,
                                return_matrix = False,
                                add_control=False,
                                function='nanmedian',
                                axis=0,
                                data_num_th=50,
                                _center_dist_th = 5000000,
                                normalize_by_center=False,
                                _contact_th=500, 
                                contact_prob=False,
                                parallel=True, 
                                num_threads=12,
                                interaction_type='trans',
                                return_raw=False,
                                plot_cis = False,
                                verbose=False):
    
    """Function"""

    # sort im_loci_df for matrix generation if not sorted
    im_loci_df['chr_as_num'] = im_loci_df['chr'].map(mouse_chr_as_num)
    im_loci_df = im_loci_df.sort_values(by = ['chr_as_num', 'chr_order'], ignore_index=False)
    im_loci_df = im_loci_df.drop(columns=['chr_as_num'])

    
    groupby = im_loci_df['Groupby'][0]
    # make sure to prepare the class_2_chrZxysList in compatible format
    if groupby == 'neuron_identity':
        ref_group = 'Neuronal'
    elif groupby == 'class_label_new':
        ref_group = 'Gluta'
    elif groupby == 'subclass_label_new':
        ref_group = 'L2/3 IT'
    print(f'Use {ref_group} for total distmap normalization')

    cellgroups = im_loci_df['Compared_groups'][0].split('; ')
    cellgroups.append(ref_group)

    if f'Distmap_norm: {cellgroups[0]}' in im_loci_df.columns and not overwrite:
        print ('Use saved factor for normalization.')

    else:
        if isinstance(norm_factor_dict, dict):
            print ('Load factor from provided dict for normalization.')
            im_loci_df[f'Distmap_norm: {cellgroups[0]}'] = norm_factor_dict[cellgroups[0]]
            im_loci_df[f'Distmap_norm: {cellgroups[1]}'] = norm_factor_dict[cellgroups[1]]

        # it'd be more efficent to prepare the below in advance and load the result using the norm_factor_dict
        # also there is a bug that has not been fixed due to: module 'numpy' has no attribute 'medain'
        else:
            print ('Calculate factor from total distmap for the Imaged Loci Dataframe.')
            print ('Set return_raw as False.')
            print ('Set plot_cis as False.')
            print ('Set interaction_type as trans.')
            return_raw=False
            plot_cis = False
            interaction_type='trans'
            
            result_dict_by_group = {}

            directions=np.flip(np.unique(im_loci_df['Expression_change'].tolist()))
            if not add_control:
                directions = [_dir for _dir in directions if _dir != 'control']

                
            from distance_atc import Chr2ZxysList_2_summaryDict_V2

            for _group in cellgroups:
                print (f'Summarizing for all loci in {_group}.')
                chr_2_zxys_list = class_2_chrZxysList[_group]

                total_codebook = total_codebook[['chr','chr_order','id']]

                result = Chr2ZxysList_2_summaryDict_V2 (chr_2_zxys_list, 
                                                            total_codebook,
                                                            function=function, 
                                                            axis=axis,
                                                            data_num_th=data_num_th,
                                                            _center_dist_th = _center_dist_th,
                                                            normalize_by_center=normalize_by_center,
                                                            _contact_th=_contact_th, 
                                                            contact_prob=contact_prob,
                                                            parallel=parallel, 
                                                            num_threads=num_threads,
                                                            interaction_type=interaction_type,
                                                            return_raw=return_raw,
                                                            verbose=verbose)
                
                result_dict_by_group[_group]= result
            
            from distance_atc import assemble_ChrDistDict_2_Matrix_V2
            print ('Return reduced pairwise distances by (chr) keys.')
            print('Calculating reduced matrix.')

            dist_map_dict_by_group = {}
            for _group in cellgroups:

                distmap, chr_edges, chr_names = assemble_ChrDistDict_2_Matrix_V2(
                                                                                result_dict_by_group[_group], 
                                                                                total_codebook, 
                                                                                sel_codebook=None, 
                                                                                use_cis=plot_cis, sort_by_region=False, # here sort_by_region=False means sort by chr?
                                                                            )

                dist_map_dict_by_group[_group] = {'matrix':distmap, 'chr_edges': chr_edges, 'chr_names':chr_names}
            
            norm_factor1 = np.medain(dist_map_dict_by_group[cellgroups[0]]['matrix']/dist_map_dict_by_group[cellgroups[2]]['matrix'])
            norm_factor2 = np.medain(dist_map_dict_by_group[cellgroups[1]]['matrix']/dist_map_dict_by_group[cellgroups[2]]['matrix'])

            im_loci_df[f'Distmap_norm: {cellgroups[0]}'] = norm_factor1
            im_loci_df[f'Distmap_norm: {cellgroups[1]}'] = norm_factor2
                
    if return_matrix and not isinstance(norm_factor_dict, dict):
        return im_loci_df, dist_map_dict_by_group
    else:
        return im_loci_df







def get_key_for_comb (_comb, chr_info_df):
    pair_key_list = []
    for _ind in _comb:
        _chr_df = chr_info_df.iloc[_ind]
        _key = (_chr_df['chr'], _chr_df['chr_order'])
        pair_key_list.append(_key)
    pair_key = tuple(pair_key_list)
    print (f'Get {pair_key}.')
    return pair_key




def sorted_pair_keys_for_loci_dataframe(im_loci_df:pd.core.frame.DataFrame,num_threads=12):
    
    """return a list of loci pair-wise keys in sorted chr order"""
    # sort df by chr order so the pair key order matches the matrix order, etc
    print ('Sort the provided loci dataframe by chr order.')
    im_loci_df = sort_loci_df_by_chr_order (im_loci_df)
    
    from itertools import combinations
    print ('Extract the loci-pairwise chr order info.')
    # the pair key would i, j combinations of all loci
    comb_inds = list(combinations(range(len(im_loci_df)),2))

    chr_info_df = im_loci_df[['chr','chr_order']]

    _mp_args = []
    for _comb in comb_inds:
        _mp_args.append((_comb, chr_info_df))

    import time
    _start_time = time.time()
    import multiprocessing as mp
    print ('Multiprocessing for extractong the loci-pairwise chr order info:')
    with mp.Pool(num_threads) as _mp_pool:
        _mp_results = _mp_pool.starmap(get_key_for_comb, _mp_args, chunksize=1)
        _mp_pool.close()
        _mp_pool.join()
        _mp_pool.terminate()
    print (f"Complete in {time.time()-_start_time:.3f}s.")

    pair_keys = _mp_results
    
    return pair_keys



def gene_activity_by_pair_key(loci_pair_key:tuple, 
                              im_loci_df:pd.core.frame.DataFrame, 
                              pair_score_type = 'mean',
                              activity_col='Activity_score_mean_genes_100kb_tss'):
    
    """Return a merged score from the corresponding score column for a pair of loci specified by their chr_chr-order"""
    
    sel_loci_pair_df = pd.DataFrame()
    for _chr_r_key in loci_pair_key:
        sel_loci_df = im_loci_df[im_loci_df['chr']==_chr_r_key[0]]
        sel_loci_df = sel_loci_df[sel_loci_df['chr_order']==_chr_r_key[1]]
        sel_loci_pair_df = pd.concat([sel_loci_pair_df, sel_loci_df])
        
    if pair_score_type == 'mean':
        _pair_score = np.nanmean(sel_loci_pair_df[activity_col])
    elif pair_score_type == 'sum':
        _pair_score = np.nansum(sel_loci_pair_df[activity_col])
        
    return _pair_score



def gene_activity_by_pair_key_list (loci_pair_key_list:list, 
                              im_loci_df:pd.core.frame.DataFrame, 
                              pair_score_type = 'mean',
                              activity_col='Activity_score_mean_genes_100kb_tss'):
    
    """For a list, call function above"""

    print("Caculate the gene activity for the input loci pair list.")
    
    score_dict_by_pair_key = {}
    for loci_pair_key in loci_pair_key_list:
        _pair_score = gene_activity_by_pair_key (loci_pair_key, 
                                                im_loci_df,
                                                pair_score_type=pair_score_type,
                                                activity_col=activity_col)

        score_dict_by_pair_key[loci_pair_key]=_pair_score

    return score_dict_by_pair_key