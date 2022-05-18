gammas=(0.62)
deltas=(0.1)
threshold_type=score
topk=1000
lambda_step=10000
bound=WSR
metric=mrr@k
num_calib=5000
batch_size=20
rtv_tau=1.3 # 200 # unicoil
rrk_tau=1.3
max_rank=10

dataset=msmarco
retriever=dpr
reranker=monoBERT
calib_rtv_trec_file=${dataset}/$retriever/${retriever}-dev.trec
calib_rrk_trec_file=${dataset}/$retriever/${reranker}/fusion-dev.trec
calib_ref_tsv_file=${dataset}/dev.qrels.tsv
val_rtv_trec_file=${dataset}/$retriever/${retriever}-test.trec
val_rrk_trec_file=${dataset}/$retriever/${reranker}/fusion-test.trec
val_ref_tsv_file=${dataset}/test.qrels.tsv

logdir=results_${dataset}_${retriever}_${reranker}_${threshold_type}_threshold_${metric}_${lambda_step}_${num_calib}

mkdir -p $logdir

for gamma in ${gammas[*]}
do
  for delta in ${deltas[*]}
  do
    echo "$gamma $delta"
    
    CUDA_VISIBLE_DEVICES=0 python cec_prediction.py \
      --out_dir $logdir \
      --calib_rtv_trec_file $calib_rtv_trec_file \
      --calib_rrk_trec_file $calib_rrk_trec_file \
      --calib_ref_tsv_file $calib_ref_tsv_file \
      --val_rtv_trec_file $val_rtv_trec_file \
      --val_rrk_trec_file $val_rrk_trec_file \
      --val_ref_tsv_file $val_ref_tsv_file \
      --gamma $gamma \
      --delta $delta \
      --bound $bound \
      --metric $metric \
      --lambda_step $lambda_step \
      --topk $topk \
      --device "cuda" \
      --num_calib $num_calib \
      --threshold_type $threshold_type \
      --batch_size $batch_size \
      --rtv_tau $rtv_tau \
      --rrk_tau $rrk_tau \
      --max_rank $max_rank
  wait $!
  done
done