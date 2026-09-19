#!/usr/bin/env python3
"""Expand a question set into three referent framings: generic / named / second_person.

Each source question becomes three rows sharing one `id`, differing only in who the
question is about. The `named` framing keeps the literal placeholder {model_name}, which
generate_responses.py substitutes with the name the model was midtrained under.

    python evals/open_ended_qa/make_framings.py \
        --questions hf:chloeli/spec-open-qa \
        --out evals/open_ended_qa/questions/spec_open_qa_framed.jsonl
"""
import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
from tqdm import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from src.utils.file_utils import append_to_jsonl, extract_text_from_tag
from src.utils.inference_utils import single_prompt_api_call
from evals.open_ended_qa.framings import FRAMINGS
from evals.open_ended_qa.generate_responses import load_questions

TEMPLATE_PATH = Path(__file__).parent / "prompts" / "framing_template.txt"
SECOND_PERSON_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
# A trailing "Do you agree?" addresses whoever is answering, not the entity the question is
# about. It is part of the item's format, survives every rewrite unchanged, and so is not
# evidence either way about which referent a framing uses.
RESPONDENT_TAIL_RE = re.compile(r"\s*do you agree(?:\s+or\s+disagree)?\s*\?\s*$", re.IGNORECASE)
# Real provider/model names must be normalized away, or one arm silently asks about a
# different entity than the others.
PROVIDER_RE = re.compile(
    r"\b(anthropic|claude|openai|chatgpt|gpt-?[0-9]|google|gemini|deepmind|meta|llama|"
    r"alibaba|qwen|mistral|cohere)\b",
    re.IGNORECASE,
)


@dataclass
class FramingConfig(ExperimentConfigBase):
    questions: str  # source set: path to .jsonl, or "hf:<dataset_id>"
    out: str  # destination .jsonl

    model_id: str = "claude-opus-4-6"
    temperature: float = 0.0
    max_tokens: int = 1000
    max_concurrent_requests: int = 20
    limit: int | None = None

    output_dir: Path = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "output_dir", Path("runs/open_ended_qa/_framings"))


def referent_only(question: str) -> str:
    """Drop a trailing respondent instruction so pronoun checks see only the question."""
    return RESPONDENT_TAIL_RE.sub("", question)


def validate(framings: dict[str, str], source: dict) -> str | None:
    """Return a reason string if the rewrite looks wrong, else None."""
    for f in FRAMINGS:
        if not framings.get(f) or len(framings[f]) < 10:
            return f"missing or too short: {f}"
    if "{model_name}" not in framings["named"]:
        return "named framing lost the {model_name} placeholder"
    if "{model_name}" in framings["generic"] or "{model_name}" in framings["second_person"]:
        return "placeholder leaked into a non-named framing"
    if not SECOND_PERSON_RE.search(referent_only(framings["second_person"])):
        return "second_person framing has no second-person pronoun"
    if SECOND_PERSON_RE.search(referent_only(framings["generic"])):
        return "generic framing contains a second-person pronoun"
    for f in FRAMINGS:
        leaked = PROVIDER_RE.search(framings[f])
        if leaked:
            return f"{f} framing keeps the provider/model name '{leaked.group(0)}'"
    return None


async def frame_one(api: InferenceAPI, cfg: FramingConfig, template: str, q: dict, sem) -> dict:
    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user,
                                          content=template.replace("{question}", q["question"]))])
    async with sem:
        try:
            completion = await single_prompt_api_call(
                api, cfg.model_id, prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )
        except Exception as e:
            return {"source": q, "error": f"{type(e).__name__}: {e}"}

    framings = {f: extract_text_from_tag(completion, f) for f in FRAMINGS}
    # extract_text_from_tag returns the whole completion when the tag is absent
    framings = {f: ("" if v == completion else v) for f, v in framings.items()}
    reason = validate(framings, q)
    return {"source": q, "framings": framings, "error": reason}


async def run(cfg: FramingConfig):
    cfg.setup_experiment(log_file_prefix="framings")
    questions = load_questions(cfg.questions, cfg.limit)
    print(f"Expanding {len(questions)} questions into {len(FRAMINGS)} framings each")

    template = TEMPLATE_PATH.read_text()
    sem = asyncio.Semaphore(cfg.max_concurrent_requests)
    tasks = [frame_one(cfg.api, cfg, template, q, sem) for q in questions]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="framing"):
        results.append(await coro)

    # as_completed yields in completion order, which varies run to run. Sort back to the
    # input order so the output file is diffable against a previous run.
    order = {q["id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: order[r["source"]["id"]])

    rows, failures = [], []
    for r in results:
        q = r["source"]
        if r.get("error"):
            failures.append({**q, "error": r["error"], "framings": r.get("framings", {})})
            continue
        for framing in FRAMINGS:
            rows.append({"id": q["id"], "category": q["category"], "framing": framing,
                         "question": r["framings"][framing], "source_question": q["question"]})

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    append_to_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows ({len(rows) // len(FRAMINGS)} questions) to {out_path}")

    if failures:
        fail_path = out_path.with_name(out_path.stem + "_failed.jsonl")
        if fail_path.exists():
            fail_path.unlink()
        append_to_jsonl(fail_path, failures)
        print(f"{len(failures)} questions failed validation -> {fail_path} (fix by hand or re-run)")


if __name__ == "__main__":
    asyncio.run(run(sp.parse(FramingConfig)))
