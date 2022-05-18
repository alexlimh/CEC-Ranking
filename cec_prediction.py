"""
MIT License

Copyright (c) 2022 Minghan Li
"""

import os
import json
import argparse
import collections
import math
from tqdm import tqdm

import numpy as np
import torch
from torch import nn, optim

import concentration
from utils import AttrDict, args_type, fix_randomness
from utils import load_retrieval_reranking_map
from utils import make_dataset, merge_shuffle, compute_metric

def define_args():
  args = AttrDict()

  # General
  args.seed = 0
  args.device = "cuda"
  args.calib_rtv_trec_file = ""
  args.calib_rrk_trec_file = ""
  args.calib_ref_tsv_file = ""

  args.val_rtv_trec_file = ""
  args.val_rrk_trec_file = ""
  args.val_ref_tsv_file = ""
  
  args.out_dir = ""
  args.merge_shuffle = False

  # CEC general
  args.min_delta = 1e-4
  args.delta = 0.1
  args.gamma = 0.6
  args.bound = "WSR"

  # CEC calibration parameters
  args.threshold_type = "score"
  args.topk = 1000
  args.lambda_step = 10000
  args.num_lam = 1500
  args.num_calib = 2000
  args.num_grid_hbb = 200
  args.epsilon = 1e-10 
  args.maxiters = int(1e3)
  args.ub = 0.2
  args.ub_sigma = np.sqrt(2)
  args.metric = "mrr@10"
  args.batch_size = 50
  args.rtv_tau = 1.3
  args.rrk_tau = 1.3
  args.max_rank = 10
  return args

class CEC:
  def __init__(self, args):
    self.args = args
    self.lambdas_example_table = torch.linspace(1-1/args.lambda_step, 0, args.lambda_step).to(args.device)
    self.bound_fn = concentration.get_bound_fn_from_string(args.bound)
    self.tlambda = concentration.get_tlambda(args.num_lam, [], args.num_calib, args.num_grid_hbb, args.ub, args.ub_sigma, args.epsilon, args.maxiters, args.bound, self.bound_fn)

  def calibrate(self, calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map):
    calib_rtv_metric_score = compute_metric(calib_ref, calib_rtv_map, metric=self.args.metric, max_rank=self.args.max_rank)
    calib_rrk_metric_score = compute_metric(calib_ref, calib_rrk_map, metric=self.args.metric, max_rank=self.args.max_rank)
    print(f"Calibration {self.args.metric} score: Retriever: {calib_rtv_metric_score}, Reranker: {calib_rrk_metric_score}")
    calib_rtv_scores, calib_rrk_scores, calib_rel, calib_true_rel, calib_total_rel, rtv_T, rrk_T = self.calibrate_logits(calib_ref, 
                                                                                          calib_rtv_map, 
                                                                                          calib_rrk_rtv_map)
    calib_losses, calib_sizes = self.get_example_loss_and_size_tables(self.lambdas_example_table,
                                                        calib_rtv_scores, calib_rrk_scores, 
                                                        calib_rel, calib_true_rel, calib_total_rel, batch_size=self.args.batch_size,
                                                        mode="calibrate", metric=self.args.metric, threshold_type=self.args.threshold_type)
    lhat_cec, calibrated_gamma, calibrated_delta = concentration.get_lhat_from_table(calib_losses, self.lambdas_example_table, 
                                              self.args.gamma, self.args.delta, self.args.min_delta, self.tlambda, self.args.bound)
    self.calib_results = {"rtv_T":float(rtv_T.item()), "rrk_T":float(rrk_T.item()), "lhat_cec": float(lhat_cec), "calibrated_gamma": float(calibrated_gamma), "calibrated_delta": float(calibrated_delta)}
    print(f"Calibration Done! Retrieval Temperature:{rtv_T.item()}, Rerank Temperature:{rrk_T.item()}, Lambda-hat:{float(lhat_cec)}, calibrated gamma: {calibrated_gamma}, calibrated delta:{calibrated_delta} ")
  
  def validate(self, val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map, max_rank=10):
    val_rtv_logits, val_rrk_logits, val_labels, val_rel, val_true_rel, val_total_rel = make_dataset(val_ref, val_rtv_map, val_rrk_map, 
                                                                                      topk=self.args.topk, device=self.args.device, mode="validation")
    val_rtv_scores = (val_rtv_logits/self.calib_results["rtv_T"]).softmax(dim=1)
    val_rrk_scores = (val_rrk_logits/self.calib_results["rrk_T"]).softmax(dim=1)

    index = np.argmax(self.lambdas_example_table.cpu().numpy() == self.calib_results["lhat_cec"])
    val_losses, val_sizes = self.get_example_loss_and_size_tables(self.lambdas_example_table[index:index+1],
                              val_rtv_scores, val_rrk_scores, 
                              val_rel, val_true_rel, val_total_rel, batch_size=1,
                              mode="validate", metric=self.args.metric, threshold_type=self.args.threshold_type)
    sizes_cec = val_sizes[:, 0]
    losses_cec = val_losses[:, 0]

    val_cec_map = {}
    val_fusion_map = {}
    output_trec_lines = []
    metrics = []
    for (qid, docs), size in tqdm(zip(val_rtv_map.items(), sizes_cec)):
      pids = list(docs.keys())
      topk_pids = pids[:int(size)]
      tail_pids = pids[int(size):]

      rerank_scores = []
      for pid in topk_pids:
        rerank_scores.append(val_rrk_map[qid][pid])

      rerank_indices = np.argsort(-np.array(rerank_scores))
      topk_pids = [topk_pids[idx] for idx in rerank_indices]
      new_pids = topk_pids + tail_pids
      val_cec_map[qid] = {pid:(len(topk_pids) - rank) for rank, pid in enumerate(topk_pids)}
      val_fusion_map[qid] = {pid:(len(new_pids) - rank) for rank, pid in enumerate(new_pids)}

      new_pids = topk_pids + tail_pids
      for j, pid in enumerate(topk_pids):
        output_trec_lines.append(f"{qid}\t{pid}\t{j+1}\n")
    counter = collections.Counter(metrics)
    # sizes = np.minimum(max_rank, np.array(sizes_cec))
    sizes = np.array(sizes_cec)
    # if lambda = 0, the cec mrr10 will be slightly different to rerank mrr10 due to tie breaking and floating point precision
    cec_eval_score = compute_metric(val_ref, val_cec_map, self.args.metric, max_rank=max_rank)
    calibrated_delta = self.calib_results["calibrated_delta"]
    calibrated_gamma = self.calib_results["calibrated_gamma"]
    print(f"Validation CEC risk: {losses_cec.mean()}, gamma: {calibrated_gamma}, confidence: {1-calibrated_delta}")

    output_data = dict(
      metric=self.args.metric,
      delta=self.args.delta,
      gamma=self.args.gamma,
      bound=self.args.bound,
      topk=self.args.topk,
      rtv_T=self.calib_results["rtv_T"],
      rrk_T=self.calib_results["rrk_T"],
      lhat_cec=self.calib_results["lhat_cec"],
      calibrated_gamma=self.calib_results["calibrated_gamma"],
      calibrated_delta=self.calib_results["calibrated_delta"],
      size_mean=float(sizes.mean()),
      size_std=float(sizes.std()),
      size_max=float(sizes.max()),
      size_min=float(sizes.min()),
      retrieval_metric_score=compute_metric(val_ref, val_rtv_map, metric=self.args.metric, max_rank=max_rank),
      rerank_metric_score=compute_metric(val_ref, val_rrk_map, metric=self.args.metric, max_rank=max_rank),
      cec_metric_score=cec_eval_score,
      cec_fusion_metric_score=compute_metric(val_ref, val_fusion_map, metric=self.args.metric, max_rank=max_rank)
    )
    return output_data, output_trec_lines

  def calibrate_logits(self, calib_ref, calib_rtv_map, calib_rrk_map):
    calib_rtv_logits, calib_rrk_logits, calib_labels, calib_rel, calib_true_rel, calib_total_rel = make_dataset(calib_ref, calib_rtv_map, calib_rrk_map, topk=self.args.topk, device=self.args.device)
    rtv_T = self.platt_logits(calib_rtv_logits, calib_labels, tau=self.args.rtv_tau, device=self.args.device)
    calib_rtv_scores = (calib_rtv_logits/rtv_T).softmax(dim=1)

    rrk_T = self.platt_logits(calib_rrk_logits, calib_labels, tau=self.args.rrk_tau, device=self.args.device)
    calib_rrk_scores = (calib_rrk_logits/rrk_T).softmax(dim=1)
    return calib_rtv_scores, calib_rrk_scores, calib_rel, calib_true_rel, calib_total_rel, rtv_T, rrk_T

  def platt_logits(self, logits, labels, max_iters=100, lr=0.001, epsilon=0.001, tau=1.3, device="cuda"):
    print("Calibrating logits...")
    nll_criterion = nn.CrossEntropyLoss()
    T = nn.Parameter(torch.Tensor([tau]).to(device)) # 1.3 is generally a good initialization

    optimizer = optim.SGD([T], lr=lr)
    for iter in tqdm(range(max_iters)):
      T_old = T.item()
      batch_inputs = []
      batch_targets = []
      for i, (x, target) in enumerate(zip(logits, labels)):
        if target.item() == -1:
          continue
        if len(batch_inputs) == 100 or i == len(logits) - 1:
          batch_inputs, batch_targets = torch.cat(batch_inputs), torch.cat(batch_targets)
          optimizer.zero_grad()
          out = batch_inputs - batch_inputs.max(-1, True)[0] # numerical stability
          out = batch_inputs/T
          loss = nll_criterion(out, batch_targets)
          loss.backward()
          optimizer.step()
          batch_inputs = []
          batch_targets = []
        else:
          x, target = x.view(1, -1), target.view(1,)
          batch_inputs.append(x)
          batch_targets.append(target)
      if abs(T_old - T.item()) < epsilon or T.item() < 0:
        break
    if T.item() < 0:
      T = torch.Tensor([T_old]).to(device)
    return T 

  def get_example_loss_and_size_tables(self, lambdas_example_table, retrieval_scores, rerank_scores, relevances, true_relevances, total_relevances, batch_size, mode="calibrate", metric="recall", threshold_type="score"):
    print(f"Making {mode} loss table...")
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    topk = self.args.topk
    max_rank = self.args.max_rank
    fname_loss = f'{self.args.out_dir}/{mode}_seed_{self.args.seed}_{metric}_{lam_low:.2f}_{lam_high:.2f}_{lam_len}_topk_{topk}_example_loss_table.npy'
    fname_sizes = f'{self.args.out_dir}/{mode}_seed_{self.args.seed}_{metric}_{lam_low:.2f}_{lam_high:.2f}_{lam_len}_topk_{topk}_example_size_table.npy'
    
    losses = torch.zeros((lam_len,)).to(self.args.device)
    if os.path.exists(fname_loss) and mode == "calibrate":
      loss_table = np.load(fname_loss)
      sizes_table = np.load(fname_sizes)
    else:
      loss_table = []
      sizes_table = []
      for i in tqdm(range(0, lam_len, batch_size)):
        batch_lambda_table = lambdas_example_table[i:i+batch_size]
        batch_loss_table, batch_sizes_table = self.compute_loss_size(
              iter=i, 
              losses=losses[i:i+batch_size],
              lambdas_example_table=batch_lambda_table,
              retrieval_scores=retrieval_scores.detach(), 
              rerank_scores=rerank_scores.detach(),
              relevances=relevances, 
              true_relevances=true_relevances,
              total_relevances=total_relevances,
              metric=metric,
              mode=mode,
              threshold_type=threshold_type,
              topk=topk,
              max_rank=max_rank)
        loss_table.append(batch_loss_table.cpu())
        sizes_table.append(batch_sizes_table.cpu())
      loss_table, sizes_table = torch.cat(loss_table, dim=-1), torch.cat(sizes_table, dim=-1)
      loss_table, sizes_table = loss_table.numpy(), sizes_table.numpy()
      
      if mode == "calibrate":
        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)
    
    return loss_table, sizes_table
  
  def compute_loss_size(self, iter, losses, lambdas_example_table, retrieval_scores, 
                        rerank_scores, relevances, true_relevances, total_relevances, 
                        metric="recall", mode="calibrate", 
                        threshold_type="score", topk=1000, max_rank=10):
    bsz, num_cand = retrieval_scores.size()
    if threshold_type == "score":
      topk_mask = torch.cat([torch.ones((bsz, topk)), torch.zeros((bsz, num_cand - topk))], dim=1).to(retrieval_scores.device)
      est_labels = (retrieval_scores.unsqueeze(-1) >= lambdas_example_table.view(1, 1, -1)).float()
      est_labels *= topk_mask.unsqueeze(-1)
    elif threshold_type == "rank":
      est_labels = []
      for lamb in lambdas_example_table.cpu().numpy().tolist():
        i = math.ceil(lamb * num_cand)
        est_labels.append(torch.cat([torch.ones((bsz, min(num_cand, num_cand - i + 1))), torch.zeros((bsz, max(0, i - 1)))], dim=1))
      est_labels = torch.stack(est_labels, dim=-1).to(retrieval_scores.device)
    else:
      raise NotImplementedError("Invalid threshold type!")
    
    sizes = est_labels.sum(dim=1)
    losses = losses.view(1, -1).repeat(len(retrieval_scores), 1)

    if metric == "recall":
      corrects = (relevances.unsqueeze(-1) * est_labels).sum(dim=1)
      losses += 1 - corrects/total_relevances.unsqueeze(-1) # percent correct labels
    
    elif metric == "tp_rate@k":
      est_labels[:, max_rank:, :] = 0
      corrects = (relevances.unsqueeze(-1) * est_labels).sum(dim=1)
      losses += 1 - corrects/max_rank
      # preds = est_labels.sum(dim=1)
      # losses += 1 - torch.where(preds>0, corrects/preds, torch.zeros_like(preds)) # percent correct labels

    elif metric == "mrr@k":
      rerank_scores = torch.where(est_labels > 0., rerank_scores.unsqueeze(-1), -float("Inf")*torch.ones_like(rerank_scores.unsqueeze(-1)).to(rerank_scores.device))
      est_rerank_indices = torch.argsort(rerank_scores, dim=1, descending=True)
      relevances = torch.where(relevances > 0, 1., 0.)
      relevances = relevances.unsqueeze(-1).repeat(1, 1, len(lambdas_example_table))
      relevances = torch.gather(relevances, 1, est_rerank_indices)
      relevances *= torch.arange(1, 0, -1/topk).view(1, topk, 1).to(relevances.device)
      max_pos = torch.argmax(relevances, dim=1).float()
      mask = (max_pos < float(max_rank)) & (relevances.sum(1) > 0)
      losses += 1 - torch.where(mask, 1./(max_pos + 1.), torch.zeros_like(max_pos, device=max_pos.device))
    
    elif metric == "ndcg@k":
      discount = 1 / (torch.log(torch.arange(relevances.size(1)) + 2) / np.log(2))
      discount[max_rank:] = 0
      discount = discount.view(1, -1, 1).to(est_labels.device)
      
      rerank_scores = torch.where(est_labels > 0., rerank_scores.unsqueeze(-1), -float("Inf")*torch.ones_like(rerank_scores.unsqueeze(-1)).to(rerank_scores.device))
      est_rerank_indices = torch.argsort(rerank_scores, dim=1, descending=True)
      relevances = relevances.unsqueeze(-1).repeat(1, 1, len(lambdas_example_table))
      rerank_relevances = torch.gather(relevances, 1, est_rerank_indices)
      
      idcg = torch.sum(discount * torch.sort(true_relevances, dim=1, descending=True)[0].unsqueeze(-1), dim=1)
      idcg = idcg.repeat(1, len(lambdas_example_table))
      dcg = torch.sum(discount * rerank_relevances, dim=1)
      all_irrelevant = idcg == 0
      dcg[all_irrelevant] = 0
      dcg[~all_irrelevant] /= idcg[~all_irrelevant]
      
      losses += 1 - dcg
    else:
      raise NotImplementedError("Invalid metric!")
    return losses, sizes
    

def main(args):
  fix_randomness(args.seed)
  cec = CEC(args)
  calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map = load_retrieval_reranking_map(args.calib_rtv_trec_file,
                                                                         args.calib_rrk_trec_file,
                                                                         args.calib_ref_tsv_file,
                                                                         args.num_calib,
                                                                         mode="calibration")
  
  val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map = load_retrieval_reranking_map(args.val_rtv_trec_file,
                                                                  args.val_rrk_trec_file,
                                                                  args.val_ref_tsv_file,
                                                                  mode="validation")
  if args.merge_shuffle:
    print("Merge, shuffle, and split data...")
    data = merge_shuffle(calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map, 
                  val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map)
    calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map, val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map = data
  
  cec.calibrate(calib_ref, calib_rtv_map, calib_rrk_map, calib_rrk_rtv_map)
  output_data, output_trec = cec.validate(val_ref, val_rtv_map, val_rrk_map, val_rrk_rtv_map, max_rank=args.max_rank)

  print(output_data)
  with open(os.path.join(args.out_dir, f"result.bound_{args.bound}.gamma_{args.gamma}.delta_{args.delta}.topk_{args.topk}.seed_{args.seed}.json"), "w") as f:
    json.dump(output_data, f)
  with open(os.path.join(args.out_dir, f"result.bound_{args.bound}.gamma_{args.gamma}.delta_{args.delta}.topk_{args.topk}.seed_{args.seed}.trec"), "w") as f:
    f.writelines(output_trec)
  print("Output saved!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  for key, value in define_args().items():
    parser.add_argument(f'--{key}', type=args_type(value), default=value)
  args = parser.parse_args()
  main(args)
