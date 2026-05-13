"""Local Hugging Face LLM relation classification for candidate pairs."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocessing import truncate_text


ALLOWED_RELATIONS = {
    "SIMILAR_TO",
    "BASED_ON",
    "EXTENDS",
    "COMPARES_WITH",
    "SAME_TOPIC",
    "NO_RELATION",
}


def load_llm(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load a local Transformers causal LLM.

    In Colab this downloads model weights into the runtime cache and runs
    inference locally. It does not use Hugging Face Inference Providers or
    require paid API tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs.update(
            {
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
        )
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, tokenizer


def _paper_text_for_prompt(paper: dict[str, Any]) -> str:
    """Build compact paper context for the relation prompt."""
    title = str(paper.get("title", "Unknown title"))
    abstract = str(paper.get("abstract", ""))
    text = abstract or str(paper.get("text", ""))
    return f"Title: {title}\nAbstract/Text: {truncate_text(text, 1800)}"


def build_relation_prompt(paper_a: dict[str, Any], paper_b: dict[str, Any]) -> str:
    """Create a prompt asking the LLM to classify a relation as JSON only."""
    return f"""You compare two scientific papers and classify their relation.

Allowed relations:
- SIMILAR_TO: methods, tasks, or findings are semantically close
- BASED_ON: Paper A explicitly relies on Paper B
- EXTENDS: Paper A extends or improves Paper B
- COMPARES_WITH: Paper A compares itself with Paper B
- SAME_TOPIC: same broad topic, but no stronger relation is clear
- NO_RELATION: no meaningful relation can be inferred

Most pairs were preselected by semantic similarity, so prefer a meaningful
relation when the papers share methods, tasks, models, datasets, or topic.
Use NO_RELATION only when the papers are clearly unrelated.

Return only valid JSON with this schema:
{{
  "relation": "...",
  "confidence": 0.0,
  "reason": "..."
}}

Paper A:
{_paper_text_for_prompt(paper_a)}

Paper B:
{_paper_text_for_prompt(paper_b)}
"""


def parse_llm_json_response(response: str) -> dict[str, str | float]:
    """Parse a local LLM response into the relation schema.

    Smaller local instruct models do not always obey the JSON-only instruction.
    This parser first tries JSON, then falls back to detecting an allowed
    relation label in plain text instead of discarding the whole answer.
    """
    fallback = {
        "relation": "NO_RELATION",
        "confidence": 0.0,
        "reason": "Failed to parse a valid JSON response.",
    }
    if not isinstance(response, str) or not response.strip():
        return fallback

    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else cleaned)
        relation = str(payload.get("relation", "NO_RELATION")).upper()
        if relation not in ALLOWED_RELATIONS:
            relation = "NO_RELATION"
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(payload.get("reason", ""))[:500]
        return {"relation": relation, "confidence": confidence, "reason": reason}
    except Exception:
        relation_pattern = "|".join(sorted(ALLOWED_RELATIONS, key=len, reverse=True))
        relation_match = re.search(rf"\b({relation_pattern})\b", cleaned.upper())
        if relation_match:
            relation = relation_match.group(1)
            return {
                "relation": relation,
                "confidence": 0.5 if relation != "NO_RELATION" else 0.0,
                "reason": cleaned[:500],
            }
        return fallback


def classify_relation(
    paper_a: dict[str, Any],
    paper_b: dict[str, Any],
    llm_model,
    tokenizer,
) -> dict[str, str | float | int]:
    """Classify a candidate paper pair using a local Transformers model."""
    prompt = build_relation_prompt(paper_a, paper_b)
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a strict scientific relation classifier. Return one compact JSON object only.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        device = getattr(llm_model, "device", next(llm_model.parameters()).device)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=320,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        result = parse_llm_json_response(response or "")
        result["raw_response"] = (response or "")[:1000]
    except Exception as exc:
        result = {
            "relation": "NO_RELATION",
            "confidence": 0.0,
            "reason": f"Local LLM inference failed: {exc}",
            "raw_response": "",
        }

    result.update(
        {
            "source": int(paper_a.get("paper_id", -1)),
            "target": int(paper_b.get("paper_id", -1)),
            "source_method": "llm_local_transformers",
        }
    )
    return result
