#!/usr/bin/env python3
"""Score open-ended QA responses for spec alignment with an LLM judge.

Rubric follows Li et al. (2026), Appendix D.2: a 1-10 alignment scale with anchors at
1/3/5/7/10. Reads <run>/responses.jsonl, writes <run>/scores.jsonl and prints the
headline mean.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
from tqdm import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from src.utils.file_utils import append_to_jsonl, extract_text_from_tag, load_jsonl, load_spec
from src.utils.inference_utils import single_prompt_api_call

SCORE_RE = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", re.IGNORECASE)
TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge_template.txt"


@dataclass
class JudgeConfig(ExperimentConfigBase):
    # Run produced by generate_responses.py (runs/open_ended_qa/<run_name>/)
    run_name: str
    # Spec filename without .txt, resolved under spec/ (e.g. "philosophy_spec", "general_spec")
    spec_file_name: str

    # Note: claude-opus-5 rejects `temperature` (400). Keep 4.6 here, or drop the
    # temperature kwarg in judge_one() if you switch judges.
    judge_model: str = "claude-opus-4-6"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 2000
    max_concurrent_requests: int = 20
    limit: int | None = None

    output_dir: Path = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "output_dir", Path(f"runs/open_ended_qa/{self.run_name}"))


def build_prompt(system_text: str, row: dict) -> Prompt:
    user_text = (
        f"<question>\n{row['question']}\n</question>\n\n"
        f"<response>\n{row['response']}\n</response>"
    )
    return Prompt(
        messages=[
            ChatMessage(role=MessageRole.system, content=system_text),
            ChatMessage(role=MessageRole.user, content=user_text),
        ]
    )


def parse_score(text: str) -> float | None:
    m = SCORE_RE.search(text or "")
    if not m:
        return None
    score = float(m.group(1))
    return score if 1.0 <= score <= 10.0 else None


async def judge_one(api: InferenceAPI, cfg: JudgeConfig, system_text: str, row: dict, sem) -> dict:
    # A generation that ran out of tokens inside <think> has no answer; scoring the
    # truncated monologue would measure the token budget, not alignment.
    if not row.get("response", "").strip():
        return {**row, "score": None, "judge_reasoning": "", "judge_error": "empty_response"}

    async with sem:
        try:
            completion = await single_prompt_api_call(
                api,
                cfg.judge_model,
                build_prompt(system_text, row),
                max_tokens=cfg.judge_max_tokens,
                temperature=cfg.judge_temperature,
            )
        except Exception as e:  # keep the run alive; unscored rows are reported separately
            return {**row, "score": None, "judge_reasoning": "", "judge_error": f"{type(e).__name__}: {e}"}

    score = parse_score(completion)
    return {
        **row,
        "score": score,
        "judge_reasoning": extract_text_from_tag(completion, "reasoning") or completion[:500],
        "judge_error": None if score is not None else "unparseable_score",
        "judge_model": cfg.judge_model,
    }


async def run(cfg: JudgeConfig):
    cfg.setup_experiment(log_file_prefix="judge")

    responses_path = cfg.output_dir / "responses.jsonl"
    rows = load_jsonl(responses_path)
    if cfg.limit:
        rows = rows[: cfg.limit]
    print(f"Judging {len(rows)} responses from {responses_path} with {cfg.judge_model}")

    spec = load_spec(cfg.spec_file_name)
    system_text = TEMPLATE_PATH.read_text().replace("{spec}", spec)

    sem = asyncio.Semaphore(cfg.max_concurrent_requests)
    tasks = [judge_one(cfg.api, cfg, system_text, r, sem) for r in rows]

    scored = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="judging"):
        scored.append(await coro)

    out_path = cfg.output_dir / "scores.jsonl"
    if out_path.exists():
        out_path.unlink()
    append_to_jsonl(out_path, scored)

    valid = [r["score"] for r in scored if r["score"] is not None]
    failed = len(scored) - len(valid)
    mean = sum(valid) / len(valid) if valid else float("nan")
    summary = {
        "run_name": cfg.run_name,
        "judge_model": cfg.judge_model,
        "spec": cfg.spec_file_name,
        "n_scored": len(valid),
        "n_failed": failed,
        "mean_alignment_score": round(mean, 3),
    }
    (cfg.output_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}. Run report.py for the per-category and per-framing breakdown.")


if __name__ == "__main__":
    asyncio.run(run(sp.parse(JudgeConfig)))
