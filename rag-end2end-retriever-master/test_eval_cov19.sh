# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag_ray.sh --help to see all the possible options



python eval_rag.py \
    --evaluation_set ./CovidQA/test/test.source \
    --output_dir  ./CovidQA/evaluation_result \
    --model_name_or_path ./CovidQA/model_checkpoints \
    --model_type rag_token \
    --eval_batch_size 4 \
    --num_beams 4 \
    --min_length 1 \
    --max_length 50 \
    --index_name custom \
    --index_path  ./CovidQA/Covid-KB/my_knowledge_dataset_hnsw_index.faiss \
    --eval_mode "e2e" \
    --gold_data_mode "ans" \
    --gold_data_path ./CovidQA/test/test.target \
    --predictions_path ./CovidQA/evaluation_results/predictions.txt \
    --print_predictions \
    --device cuda \

# Stop the Ray cluster.
ray stop
