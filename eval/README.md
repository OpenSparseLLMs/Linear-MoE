# Linear-MoE Evaluation

First you should install `lm-evaluation-harness` in [third_party/lm-evaluation-harness](third_party/lm-evaluation-harness) by:

```bash
cd third_party/lm-evaluation-harness
pip install -e .
```

Edit `lm_eval_linear_moe.sh` to set checkpoint path like:

```bash
CHECKPOINT_DIR=/your/checkpoint/dir
CHECKPOINT_PATH=${CHECKPOINT_DIR}/pretrain-mcore-linear_attention-qwen2-A0.3B-lr-1e-4-minlr-1e-5-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-15000000000-wt-10000
```

and set the model and training configurations like:

```bash
MODEL_SIZE=A0.3B
SEQ_LEN=2048
PAD_LEN=2048
PR=bf16
TP=1
PP=1
EP=1

...

LA_MODULE="linear_attention"
BASE_MODEL="qwen2"

# set LAYER_TYPE_LIST="LLLNLLLNLLLN" for hybrid model
LAYER_TYPE_LIST="LLLLLLLLLLLL" 

# Linear-MoE options
linear_moe_options=" \
        --use-la-module \
        --la-module ${LA_MODULE} \
        --la-mode fused_chunk \
        --base-model ${BASE_MODEL} \
        --la-feature-map elu \
        --la-output-norm rmsnorm \
        --la-gate-fn swish \
        --layer-type-list ${LAYER_TYPE_LIST} \
        "
```

then set the evaluation task and other configurations for `lm-evaluation-harness`:

```bash
run_cmd="torchrun $DISTRIBUTED_ARGS --no-python lm_eval \
 --model linear_moe \
 --model_args path=${CHECKPOINT_PATH} max_length=2048 \
 --tasks piqa \
 --device cuda \
 --batch_size 16 \
 --output_path lm_eval_result \
 ${megatron_options} ${pr_options} ${load_options} ${input_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${moe_options} ${linear_moe_options}"
```

After finishing the above settings, evaluate Linear-MoE models by:

```bash
sh lm_eval_linear_moe.sh
```

The evaluation results would be presented like below:

```bash
linear_moe (path=/your/checkpoint/dir/pretrain-mcore-linear_attention-qwen2-A0.3B-lr-1e-4-minlr-1e-5-bs-8-gbs-64-seqlen-2048-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt-15000000000-wt-10000), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
|Tasks|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----|------:|------|-----:|--------|---|-----:|---|-----:|
|piqa |      1|none  |     0|acc     |↑  |0.6436|±  |0.0112|
|     |       |none  |     0|acc_norm|↑  |0.6436|±  |0.0112|
```
