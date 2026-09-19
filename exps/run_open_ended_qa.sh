#!/bin/bash
# Open-ended QA eval: sample responses from an MSM model, judge them for spec alignment.
#
# Stage 1 (generate) needs a GPU: either run it on a GPU box with BACKEND=vllm, or serve
# the model there and point BACKEND=openai at it:
#   vllm serve Qwen/Qwen3-32B --enable-lora \
#       --lora-modules msm=chloeli/qwen-3-32b-philosophy-spec-msm
# Stage 2 (judge) and stage 3 (report) run anywhere.

RUN_NAME="msm"
QUESTIONS="evals/open_ended_qa/questions/example_framed.jsonl"
MODEL_NAME="Qwen"   # substituted into {model_name} in the "named" framing
SPEC_FILE_NAME="philosophy_spec"

BASE_MODEL="Qwen/Qwen3-32B"
ADAPTER="chloeli/qwen-3-32b-philosophy-spec-msm"   # empty string = baseline run
BACKEND="vllm"                                     # vllm | openai
N_SAMPLES=8
TEMPERATURE=0.7
MAX_NEW_TOKENS=1024

JUDGE_MODEL="claude-opus-4-6"
MAX_CONCURRENT=20

python evals/open_ended_qa/generate_responses.py \
    --run_name "$RUN_NAME" \
    --questions "$QUESTIONS" \
    --base_model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --model_name "$MODEL_NAME" \
    --backend "$BACKEND" \
    --n_samples $N_SAMPLES \
    --temperature $TEMPERATURE \
    --max_new_tokens $MAX_NEW_TOKENS

python evals/open_ended_qa/judge_responses.py \
    --run_name "$RUN_NAME" \
    --spec_file_name "$SPEC_FILE_NAME" \
    --judge_model "$JUDGE_MODEL" \
    --max_concurrent_requests $MAX_CONCURRENT

python evals/open_ended_qa/report.py --runs "$RUN_NAME"
