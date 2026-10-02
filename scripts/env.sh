export HF_HOME=/mnt/hdd/model_training/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OLLAMA_HOST=http://127.0.0.1:11434
