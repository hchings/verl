#!/usr/bin/env bash
set -xeuo pipefail

# Example: DAPO training with NVFP4 QAT using Megatron backend
# This script demonstrates how to enable Quantization-Aware Training
#
# Environment:
#   Docker image: verlai/verl:vllm012.latest
#   Megatron-Bridge needs to be installed manually inside the container:
#     pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@e940d997d7bdb7810f621f5b32bf70255b5aa2d9

# Clean SLURM/MPI/PMIx env (defensive — matches the TRT-LLM sibling).
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

# ===== Cluster layout toggle =====
# LAYOUT=4n -> 4 nodes x 4 GPUs = 16 GPUs (EP=16, CP=8)  [default]
# LAYOUT=8n -> 8 nodes x 4 GPUs = 32 GPUs (EP=32, CP=8)
LAYOUT=${LAYOUT:-4n}
case "${LAYOUT}" in
    4n) NNODES=4; GPUS_PER_NODE=4; ACTOR_EP=16; ACTOR_CP=8 ;;
    8n) NNODES=8; GPUS_PER_NODE=4; ACTOR_EP=32; ACTOR_CP=8 ;;
    *)  echo "ERROR: unknown LAYOUT='${LAYOUT}'. Use '4n' or '8n'." >&2; exit 1 ;;
esac
REF_EP=${ACTOR_EP}
REF_CP=${ACTOR_CP}
echo "LAYOUT=${LAYOUT}  ->  NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}, EP=${ACTOR_EP}, CP=${ACTOR_CP}"

ID=${1:-"dapo-qwen3-30b-a3b-b32-r20k-nvfp4-qat-megatron-vllm-${LAYOUT}"}

################################################### quick config ###################################################

project_name='VERL-NVFP4-QAT'
exp_name=$ID

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
# SMOKE=1 -> short responses to make the first rollout finish in ~30s-1min instead
# of ~5-15min. Same QAT/weight-sync code path, just less generation per rollout.
if [ "${SMOKE:-0}" = "1" ]; then
    max_response_length=$((1024 * 2))   # 2k tokens (smoke)
else
    max_response_length=$((1024 * 20))  # 20k tokens (Shawn's recipe, prod)
fi
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# SMOKE=1 -> bs=8, n=4. Steps unchanged.
if [ "${SMOKE:-0}" = "1" ]; then
    train_prompt_bsz=8
    train_prompt_mini_bsz=8
    n_resp_per_prompt=4
else
    train_prompt_bsz=32
    train_prompt_mini_bsz=32
    n_resp_per_prompt=16
fi
gen_prompt_bsz=$((train_prompt_bsz * 2))

# Ray
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"

# Paths (hard-coded for erinh's Lyris setup; match the TRT-LLM script)
MODEL_PATH="/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen3-30B-A3B-Base"
# SMOKE=1 -> 5k-row subset of dapo-math-17k (skips the ~9 min filter pass on the
# full 1.79M-row parquet). Plenty for a 50-step smoke run.
if [ "${SMOKE:-0}" = "1" ]; then
    TRAIN_FILE="/lustre/fsw/coreai_comparch_trtllm/erinh/verl/data/dapo-math-17k-5k.parquet"
else
    TRAIN_FILE="/lustre/fsw/coreai_comparch_trtllm/erinh/verl/data/DAPO-Math-17k/data/dapo-math-17k.parquet"
fi
TEST_FILE="/lustre/fsw/coreai_comparch_trtllm/erinh/verl/data/AIME-2024/data/aime-2024.parquet"
CKPTS_DIR="/lustre/fsw/coreai_comparch_trtllm/erinh/ckpts/${project_name}/${exp_name}"
echo "SMOKE=${SMOKE:-0}  TRAIN_FILE=${TRAIN_FILE}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
# SMOKE=1 -> disable filter_groups (Base model has uniform-reward responses;
# filter retries up to max_num_gen_batches then raises ValueError).
if [ "${SMOKE:-0}" = "1" ]; then
    enable_filter_groups=False
else
    enable_filter_groups=True
fi
filter_groups_metric=acc
max_num_gen_batches=10

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

# Rollout Importance Sampling parameters
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_token_veto_threshold=null

# QAT Configuration
qat_enable=True
qat_mode=w4a16    # w4a16 for weight-only FP4
qat_config_path="${qat_config_path:-"${WORKING_DIR}/recipe/qat/config/nvfp4_w4a16_megatron.json"}"

export VERL_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# verl's modelopt patch converts dense NVFP4 weights to Marlin layout; force
# vLLM 0.17+ select_nvfp4_linear_backend() to pick MARLIN to match.
export VLLM_NVFP4_GEMM_BACKEND=marlin

################################################### start of config ###################################################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.gen_batch_size=${gen_prompt_bsz}
    data.train_batch_size=${train_prompt_bsz}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
)

QAT=(
    actor_rollout_ref.actor.megatron.qat.enable=${qat_enable}
    actor_rollout_ref.actor.megatron.qat.mode=${qat_mode}
    actor_rollout_ref.actor.megatron.qat.quantization_config_path="${qat_config_path}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50
    actor_rollout_ref.rollout.max_model_len=$(( max_prompt_length + max_response_length ))
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 16 ))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    # vLLM 0.17+: verl's W4A16 patch converts weights to Marlin layout and skips
    # the activation-scale alpha computation only when nvfp4_backend==MARLIN.
    # Force selection here so make_nvfp4_moe_quant_config returns the weight-only config.
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=marlin
)

# SMOKE=1 -> disable DeepEP + grad-accum-fusion; switch to alltoall dispatcher.
# Reasons:
#   - DeepEP: uncalibrated W4A16 QAT routing weights produce expert load
#     DeepEP's preallocated buffers can't absorb (deep_ep.cpp:278 illegal memory access).
#   - gradient_accumulation_fusion: QAT-wrapped weights don't get main_grad allocated by
#     Megatron's grad bucket; LinearWithGradAccumulationAndAsyncCommunication.backward
#     raises AttributeError: 'NoneType' object has no attribute 'dtype' at layers.py:568.
if [ "${SMOKE:-0}" = "1" ]; then
    moe_enable_deepep=False
    moe_token_dispatcher_type=alltoall
    gradient_accumulation_fusion=False
else
    moe_enable_deepep=True
    moe_token_dispatcher_type=flex
    gradient_accumulation_fusion=True
fi

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=${moe_enable_deepep}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=${moe_token_dispatcher_type}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=${gradient_accumulation_fusion}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    reward.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.save_freq=${SAVE_FREQ:-100}
    trainer.total_epochs=100
    trainer.total_training_steps=500
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

################################################### start script ###################################################

# Run python directly against the running Ray cluster on lustre. No --working-dir
# upload (the verl checkout is already on /lustre on every worker, same as the
# other reference scripts under tmp/).

python3 -m recipe.dapo.main_dapo \
    --config-path="${WORKING_DIR}/recipe/qat/config" \
    --config-name=dapo_qat_megatron_trainer \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${QAT[@]}" \
    "${ROLLOUT[@]}" \
    "${PERF_OPT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}"
