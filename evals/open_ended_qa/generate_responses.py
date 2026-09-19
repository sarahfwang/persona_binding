#!/usr/bin/env python3
"""Sample responses to open-ended spec questions from an MSM model (base + LoRA adapter).

Two backends:
  - vllm:   loads the base model locally and applies the LoRA adapter (needs a GPU box)
  - openai: talks to an OpenAI-compatible server, e.g.
            vllm serve Qwen/Qwen3-32B --enable-lora \
                --lora-modules msm=chloeli/qwen-3-32b-philosophy-spec-msm

Output: <output_dir>/responses.jsonl, one row per (question, sample).
"""
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
from tqdm import tqdm

from evals.open_ended_qa.framings import UNSPECIFIED
from src.utils.file_utils import append_to_jsonl, load_jsonl

def strip_thinking(raw: str) -> tuple[str, bool]:
    """Return (final answer, complete). Mirrors Qwen3's own chat-template logic.

    The model may emit a closing </think> without an opening tag (some serving paths
    prefill "<think>\n" into the prompt), so split on the closing tag rather than
    matching a balanced pair. An opening tag with no closing tag means generation hit
    max_new_tokens mid-thought and there is no answer to judge.
    """
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip(), True
    if "<think>" in raw:
        return "", False
    return raw.strip(), True


@dataclass
class GenerateConfig:
    # Question set: path to a .jsonl, or "hf:<dataset_id>" to pull from the Hub
    questions: str
    # Name for this run; outputs land in runs/open_ended_qa/<run_name>/
    run_name: str

    base_model: str = "Qwen/Qwen3-32B"
    # HF repo id or local path of the LoRA adapter. Leave empty for the baseline.
    adapter: str = ""
    backend: str = "vllm"  # vllm | openai
    # Substituted into any {model_name} placeholder in the questions. Use the name the
    # model was midtrained under (Qwen for the chloeli adapters), not the judge's name.
    model_name: str = "Qwen"

    n_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048
    enable_thinking: bool = True
    strip_think: bool = True
    limit: int | None = None
    seed: int = 42

    # vllm backend
    tensor_parallel_size: int = 1
    max_lora_rank: int = 64
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90

    # openai-compatible backend
    base_url: str = "http://localhost:8000/v1"
    served_model: str = ""  # defaults to the lora module name, else base_model
    api_key_env: str = "VLLM_API_KEY"
    concurrency: int = 16

    output_dir: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(f"runs/open_ended_qa/{self.run_name}"))


def load_questions(spec: str, limit: int | None = None) -> list[dict]:
    """Load questions from a local jsonl or a HF dataset ("hf:<id>")."""
    if spec.startswith("hf:"):
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset(spec[3:], split="train")]
    else:
        rows = load_jsonl(Path(spec))

    out = []
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("prompt")
        if not q:
            raise ValueError(f"row {i} has no 'question' or 'prompt' field: {r}")
        out.append(
            {
                "id": str(r.get("id", i)),
                "category": r.get("category", "uncategorized"),
                # framing is optional; only set by make_framings.py, and only needed
                # for the paired generic/named/second_person analysis in report.py
                "framing": r.get("framing", UNSPECIFIED),
                "question": q,
                "system_prompt": r.get("system_prompt", ""),
            }
        )
    return out[:limit] if limit else out


def resolve_adapter(adapter: str) -> str | None:
    """Local path if it exists, otherwise download the HF repo and return the snapshot path."""
    if not adapter:
        return None
    if Path(adapter).exists():
        return adapter
    from huggingface_hub import snapshot_download

    print(f"Downloading adapter {adapter} ...")
    return snapshot_download(repo_id=adapter)


def gen_vllm(cfg: GenerateConfig, questions: list[dict]) -> list[list[str]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_path = resolve_adapter(cfg.adapter)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    prompts = []
    for q in questions:
        msgs = []
        if q["system_prompt"]:
            msgs.append({"role": "system", "content": q["system_prompt"]})
        msgs.append({"role": "user", "content": q["question"]})
        prompts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=cfg.enable_thinking,
            )
        )

    llm = LLM(
        model=cfg.base_model,
        enable_lora=adapter_path is not None,
        max_lora_rank=cfg.max_lora_rank,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16",
        seed=cfg.seed,
    )
    params = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )
    lora = LoRARequest("msm", 1, adapter_path) if adapter_path else None
    outputs = llm.generate(prompts, params, lora_request=lora)
    return [[c.text for c in o.outputs] for o in outputs]


async def gen_openai(cfg: GenerateConfig, questions: list[dict]) -> list[list[str]]:
    import httpx

    model = cfg.served_model or (cfg.adapter.split("/")[-1] if cfg.adapter else cfg.base_model)
    headers = {"Authorization": f"Bearer {os.environ.get(cfg.api_key_env, 'EMPTY')}"}
    sem = asyncio.Semaphore(cfg.concurrency)

    async with httpx.AsyncClient(timeout=600.0) as client:

        async def one(q: dict) -> list[str]:
            msgs = []
            if q["system_prompt"]:
                msgs.append({"role": "system", "content": q["system_prompt"]})
            msgs.append({"role": "user", "content": q["question"]})
            body = {
                "model": model,
                "messages": msgs,
                "n": cfg.n_samples,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens,
                "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
            }
            async with sem:
                r = await client.post(f"{cfg.base_url}/chat/completions", json=body, headers=headers)
                r.raise_for_status()
                data = r.json()
            texts = [c["message"]["content"] or "" for c in data["choices"]]
            # some servers ignore n; top up with repeated single calls
            while len(texts) < cfg.n_samples:
                async with sem:
                    r = await client.post(
                        f"{cfg.base_url}/chat/completions", json={**body, "n": 1}, headers=headers
                    )
                    r.raise_for_status()
                texts.append(r.json()["choices"][0]["message"]["content"] or "")
            return texts

        return await asyncio.gather(*[one(q) for q in tqdm(questions, desc="generating")])


def main(cfg: GenerateConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "responses.jsonl"
    if out_path.exists():
        out_path.unlink()

    questions = load_questions(cfg.questions, cfg.limit)
    for q in questions:
        q["question"] = q["question"].replace("{model_name}", cfg.model_name)
        q["system_prompt"] = q["system_prompt"].replace("{model_name}", cfg.model_name)
    print(f"{len(questions)} questions x {cfg.n_samples} samples -> {out_path}")

    if cfg.backend == "vllm":
        samples = gen_vllm(cfg, questions)
    elif cfg.backend == "openai":
        samples = asyncio.run(gen_openai(cfg, questions))
    else:
        raise ValueError(f"unknown backend: {cfg.backend}")

    model_label = cfg.adapter or cfg.base_model
    rows, truncated = [], 0
    for q, texts in zip(questions, samples):
        for i, raw in enumerate(texts):
            if cfg.strip_think:
                response, complete = strip_thinking(raw)
            else:
                response, complete = raw.strip(), True
            truncated += not complete
            rows.append({**q, "sample_idx": i, "response": response, "response_raw": raw,
                         "complete": complete, "model_label": model_label})
    append_to_jsonl(out_path, rows)
    if truncated:
        print(f"WARNING: {truncated}/{len(rows)} generations ran out of tokens inside "
              f"<think> and have no answer. Raise --max_new_tokens (currently "
              f"{cfg.max_new_tokens}); they will be left unscored.")

    (cfg.output_dir / "generate_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(cfg).items()}, indent=2)
    )
    print(f"Wrote {len(rows)} responses to {out_path}")


if __name__ == "__main__":
    main(sp.parse(GenerateConfig))
