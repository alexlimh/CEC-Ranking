"""
MIT License

Copyright (c) 2022 Minghan Li
"""

import logging
import os
import re
import glob
import random
import json 
import gzip
import csv
import pathlib
from nirtools.ir import load_qrels

from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator, Callable
import numpy as np
import torch

logger = logging.getLogger(__name__)

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def args_type(default):
    if default is None:
        return lambda x: x
    if isinstance(default, bool):
        return lambda x: bool(['False', 'True'].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    if isinstance(default, (list, tuple)):
        return lambda x: tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)


def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            if "\t" in l:
                l = l.strip().split("\t")
            else:
                l = l.strip().split(" ")
            qid = l[0]
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = {}
            qids_to_relevant_passageids[qid][l[2]] = float(l[3])
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_trec_file(retrieval_trec_file, size=None):
    retrieval_map = {}
    retrieval_map_files = glob.glob(retrieval_trec_file)
    for retrieval_map_file in tqdm(retrieval_map_files):
        with open(retrieval_map_file) as f:
            for line in f:
                try:
                    qid, _, pid, rank, score, _ = line.strip().split('\t')
                except:
                    qid, _, pid, rank, score, _ = line.strip().split(' ')
                if qid not in retrieval_map:
                    retrieval_map[qid] = {pid:float(score)} 
                else:
                    retrieval_map[qid][pid] = float(score)
                if size is not None and len(retrieval_map) == size and len(retrieval_map[qid]) == 1000:
                    break
    return retrieval_map

def load_retrieval_reranking_map(rtv_trec_file, rrk_trec_file, ref_file, num_calib=None, mode="calibration"):
    print("Loading data...")
    ref = load_reference(ref_file)
    rtv_map = load_trec_file(rtv_trec_file)
    rrk_map = load_trec_file(rrk_trec_file)
    
    overlap_qids = set(ref.keys()).intersection(set(rtv_map.keys()))
    overlap_qids = list(overlap_qids)[:num_calib]
    ref = {qid: ref[qid] for qid in overlap_qids}
    rtv_map = {qid: rtv_map[qid] for qid in overlap_qids}
    rrk_map = {qid: rrk_map[qid] for qid in overlap_qids}
    rrk_rtv_map = {}
    for qid, doc_ids_scores in rtv_map.items():
        rrk_rtv_map[qid] = {}
        for doc_id in doc_ids_scores.keys():
            if doc_id not in rrk_map[qid]:
                print(qid, doc_id)
            rrk_rtv_map[qid][doc_id] = rrk_map[qid][doc_id]
    return ref, rtv_map, rrk_map, rrk_rtv_map

def merge_shuffle(calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map, 
                  val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map):
    num_calib = len(calib_ref)
    merged_ref = [(k, v) for k, v in {**calib_ref, **val_ref}.items()]
    merged_rtv_map = [(k, v) for k, v in {**calib_rtv_map, **val_rtv_map}.items()]
    merged_rrk_map = [(k, v) for k, v in {**calib_rrk_map, **val_rrk_map}.items()]
    merged_rrk_rtv_map = [(k, v) for k, v in {**calib_rrk_rtv_map, **val_rrk_rtv_map}.items()]

    merged = list(zip(merged_ref, merged_rtv_map, merged_rrk_map, merged_rrk_rtv_map))
    random.shuffle(merged)
    merged_ref, merged_rtv_map, merged_rrk_map, merged_rrk_rtv_map = list(zip(*merged))

    calib_ref, val_ref = dict(merged_ref[:num_calib]), dict(merged_ref[num_calib:])
    calib_rtv_map, val_rtv_map = dict(merged_rtv_map[:num_calib]), dict(merged_rtv_map[num_calib:])
    calib_rrk_map, val_rrk_map = dict(merged_rrk_map[:num_calib]), dict(merged_rrk_map[num_calib:])
    calib_rrk_rtv_map, val_rrk_rtv_map = dict(merged_rrk_rtv_map[:num_calib]), dict(merged_rrk_rtv_map[num_calib:])

    return calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map, \
                  val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map


def make_dataset(reference, retrieval_map, rerank_map=None, topk=1000, device="cuda", mode="calibration"):
    print(f'Making {mode} Dataset...')
    retrieval_logits = []
    rerank_logits = []
    rtv_relevances = []
    true_relevances = []
    total_relevances = []
    labels = []
    for (qid, doc_ids_scores) in tqdm(retrieval_map.items()):
        doc_ids, doc_scores = list(doc_ids_scores.keys()), list(doc_ids_scores.values())
        if len(doc_scores) < topk:
            min_score = min(doc_scores)
            doc_scores += [min_score] * (topk - len(doc_scores))
        retrieval_logits.append(doc_scores)
        if rerank_map is not None:
            doc_scores = [rerank_map[qid][doc_id] for doc_id in doc_ids]
            if len(doc_scores) < topk:
                min_score = min(doc_scores)
                doc_scores += [min_score] * (topk - len(doc_scores))
            rerank_logits.append(doc_scores)

        rtv_relevance = [0 for _ in range(topk)]
        true_relevance = [0 for _ in range(topk)]
        label = -1
        for i, (pos_doc_id, pos_doc_score) in enumerate(reference[qid].items()):
            true_relevance[i] = pos_doc_score
            if pos_doc_id in set(doc_ids):
                idx = doc_ids.index(pos_doc_id)
                rtv_relevance[idx] = pos_doc_score
                if label == -1:
                    label = idx
        labels.append(label)
        true_relevances.append(true_relevance)
        rtv_relevances.append(rtv_relevance)
        total_relevances.append(sum(list(reference[qid].values())))
    
    retrieval_logits = torch.FloatTensor(retrieval_logits).to(device)
    if rerank_map is not None:
        rerank_logits = torch.FloatTensor(rerank_logits).to(device)
    labels = torch.LongTensor(labels).to(device)
    true_relevances = torch.FloatTensor(true_relevances).to(device)
    rtv_relevances = torch.FloatTensor(rtv_relevances).to(device)
    total_relevances = torch.FloatTensor(total_relevances).to(device)
    return retrieval_logits, rerank_logits, labels, rtv_relevances, true_relevances, total_relevances
  

def compute_metric(qids_to_relevant_passageids_scores, qids_to_ranked_candidate_passages, metric="mrr@k", max_rank=10):
    if metric == "mrr@k":
        MRR = 0
        MaxMRRRank = max_rank
        ranking = []
        MRRs = []
        for qid, doc_ids_scores in qids_to_ranked_candidate_passages.items():
            if qid in qids_to_relevant_passageids_scores:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids_scores[qid]
                candidate_pid = list(qids_to_ranked_candidate_passages[qid].keys())
                for i in range(0,min(len(candidate_pid), MaxMRRRank)):
                    if candidate_pid[i] in target_pid:
                        MRR += 1/(i + 1)
                        MRRs.append(i)
                        ranking.pop()
                        ranking.append(i+1)
                        break
        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
        result = MRR/len(qids_to_ranked_candidate_passages)
    elif metric == "recall":
        total_recall = 0
        recalls = []
        for qid, doc_ids_scores in qids_to_ranked_candidate_passages.items():
            if qid in qids_to_relevant_passageids_scores:
                recall = 0
                target_pids = qids_to_relevant_passageids_scores[qid]
                for doc_id in doc_ids_scores.keys():
                    if doc_id in target_pids:
                        recall += 1
                recall /= len(target_pids)
                recalls.append(recall)
                total_recall += recall
        result = total_recall/(len(qids_to_ranked_candidate_passages))
    
    elif metric == "tp_rate@k":
        total_precision = 0
        precisions = []
        for qid, doc_ids_scores in qids_to_ranked_candidate_passages.items():
            if qid in qids_to_relevant_passageids_scores:
                precision = 0
                target_pids = qids_to_relevant_passageids_scores[qid]
                for i, doc_id in enumerate(doc_ids_scores.keys()):
                    if i >= max_rank:
                        break
                    if doc_id in target_pids:
                        precision += 1                    
                precision /= max_rank
                precisions.append(precision)
                total_precision += precision
        result = total_precision/(len(qids_to_ranked_candidate_passages))

    elif metric == "ndcg@k":
        ndcg = 0
        discount = 1 / (np.log(np.arange(1000) + 2) / np.log(2))
        discount[max_rank:] = 0
        ndcgs = []
        for j, (qid, doc_ids_scores) in enumerate(qids_to_ranked_candidate_passages.items()):
            if qid in qids_to_relevant_passageids_scores:
                target_pid = qids_to_relevant_passageids_scores[qid]
                gain = np.zeros(1000)
                for i, (pid, score) in enumerate(qids_to_ranked_candidate_passages[qid].items()):
                    if pid in qids_to_relevant_passageids_scores[qid]:
                        gain[i] = target_pid[pid]
                gain = np.sum(gain * discount)
                
                n_gain = np.zeros(1000)
                for i, (pid, score) in enumerate(qids_to_relevant_passageids_scores[qid].items()):
                    n_gain[i] = score
                n_gain = np.sort(n_gain)[::-1]
                n_gain = np.sum(n_gain * discount)
                
                ndcg += gain/n_gain if n_gain != 0 else 0
                ndcgs.append(gain/n_gain if n_gain != 0 else 0)
        result = ndcg/len(qids_to_ranked_candidate_passages)
    else:
        raise NotImplementedError("Metric Not Implemented!")
    return result

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)