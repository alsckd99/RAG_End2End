# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

#creates the custom knowlegebase
python use_own_knowledge_dataset.py \
    --csv_path ./CovidQA/splitted_covid_dump.csv \
    --output_dir ./CovidQA/Covid-KB

# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag_ray.sh --help to see all the possible options



python finetune_rag.py \
    --data_dir  ./CovidQA/training/Q-P \
    --output_dir  ./CovidQA/model_checkpoints \
    --model_name_or_path facebook/rag-token-base \
    --model_type rag_token \
    --fp16 \
    --gpus 1  \
    --profile \
    --do_train \
    --end2end \
    --do_predict \
    --n_val -1  \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 10 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --distributed_retriever ray \
    --num_retrieval_workers 4  \
    --csv_path ./CovidQA/splitted_covid_dump.csv\
    --index_name custom \
    --index_path  ./CovidQA/Covid-KB/my_knowledge_dataset_hnsw_index.faiss \
    --passage_path ./CovidQA/Covid-KB/my_knowledge_dataset \
    --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
    --cache_dir ./CovidQA/Covid-KB/cache \
    --index_gpus 1 \
    --gpu_order [6,7,8,9,0,1,2,3,5,4] \
    --shard_dir ./test_dir/kb-shards \
    --num_workers 4 \
    --indexing_freq 5
   
    

# Stop the Ray cluster.
ray stop
