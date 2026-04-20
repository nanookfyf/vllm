
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# py-spy record -o profile.svg -- /mnt/nvme/fyf/proj2/vllm/runs/serve.sh
VLLM_SERVER_DEV_MODE=1 vllm serve /mnt/nvme/fyf/models/DeepSeek-V2-Lite   --trust-remote-code   --tensor-parallel-size 1   --data-parallel-size 2   --api-server-count 1   --enable-expert-parallel   --enable-eplb   --eplb-config.num_redundant_experts 64   --enable-sleep-mode   --enforce-eager --port 8005 --gpu-memory-utilization 0.5



curl -X POST "http://127.0.0.1:8005/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer EMPTY"   -d '{
    "model": "/mnt/nvme/fyf/models/DeepSeek-V2-Lite",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'


curl -X POST "http://127.0.0.1:8005/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer EMPTY"  -H "X-data-parallel-rank: 0" -d '{
    "model": "/mnt/nvme/fyf/models/DeepSeek-V2-Lite",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 5
  }'


curl -X POST http://127.0.0.1:8005/rearrange_ep_ranks \
  -H 'Content-Type: application/json' \
  -d '{
    "sleeping_ep_ranks": [1]
  }'

curl -X POST http://127.0.0.1:8005/restore_rearranged_ep_ranks \
  -H 'Content-Type: application/json' \
  -d '{}'


curl -X POST http://127.0.0.1:8005/sleep_ep_ranks_by_tags \
  -H 'Content-Type: application/json' \
  -d '{
    "sleeping_ep_ranks": [1],
    "tags": ["expert_weights"]
  }'
curl -X POST http://127.0.0.1:8005/wake_up_ep_ranks_by_tags \
  -H 'Content-Type: application/json' \
  -d '{
    "sleeping_ep_ranks": [1],
    "tags": ["expert_weights"]
  }'





---

