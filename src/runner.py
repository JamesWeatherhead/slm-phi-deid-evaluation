#!/usr/bin/env python3
"""
SLM De-Identification Evaluation Runner

Main orchestrator for running inference across model x prompt configurations
on a remote Ollama instance (A100 GPU cluster).

Features:
- CLI args: --model, --prompt, --run, --host, --output-dir
- Checkpointing: saves after every query, resumes from last checkpoint
- 10 warmup queries (discarded) before scored evaluation
- Canary queries at position 0 and 1052 for drift detection
- 120-second timeout per query with retry (max 2)
- Stateless /api/chat calls (no conversation history)
- Full logging of every response

Qwen 3 handling:
- Automatically prepends /no_think to user prompts for qwen3-4b
- Applies presence_penalty=1.5 from model_overrides to prevent looping
- Strips <think>...</think> blocks from output as safety net

Usage:
    # Run a single configuration
    python -m src.runner --model phi4-mini --prompt zero-shot-structured --run 1

    # Run the full matrix
    python -m src.runner --all

    # Run against remote host
    python -m src.runner --all --host http://<your-ollama-host>:11434
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import Timeout, ConnectionError as ReqConnectionError

from src.utils import (
    BASE_DIR, load_dataset, load_prompt_templates, load_model_configs,
    load_inference_params, compute_prompt_hash, parse_direct_output,
    parse_phi_json, apply_redactions, extract_cot_output, ParseResult,
    setup_logging, utc_timestamp,
)

logger = logging.getLogger("runner")

# Default Ollama host
DEFAULT_HOST = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Ollama API interaction
# ---------------------------------------------------------------------------

def query_ollama(host, model_tag, system_prompt, user_prompt, params,
                 timeout=120):
    """Send a single stateless query to Ollama /api/chat.

    Returns:
        dict with response data and metadata, or error dict on failure.
    """
    options = {
        "temperature": params.get("temperature", 0),
        "top_k": params.get("top_k", 1),
        "seed": params.get("seed", 42),
        "num_predict": params.get("num_predict", 1024),
        "num_ctx": params.get("num_ctx", 4096),
    }

    # Apply presence_penalty if specified (used for qwen3-4b looping fix)
    if "presence_penalty" in params:
        options["presence_penalty"] = params["presence_penalty"]

    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
    }

    start_ms = time.time() * 1000
    try:
        response = requests.post(
            f"{host}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        end_ms = time.time() * 1000
        latency_ms = int(end_ms - start_ms)

        # Extract output text
        raw_output = ""
        if "message" in result and "content" in result["message"]:
            raw_output = result["message"]["content"]

        # Extract token counts from Ollama response
        prompt_eval_count = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)

        # Detect potential truncation
        output_truncated = (
            eval_count >= params.get("num_predict", 1024) - 1
        )
        input_truncated = (
            prompt_eval_count + params.get("num_predict", 1024)
            > params.get("num_ctx", 4096)
        )

        return {
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "tokens_input": prompt_eval_count,
            "tokens_output": eval_count,
            "output_truncated": output_truncated,
            "input_truncated": input_truncated,
            "prompt_eval_duration_ns": result.get("prompt_eval_duration", 0),
            "eval_duration_ns": result.get("eval_duration", 0),
            "total_duration_ns": result.get("total_duration", 0),
            "error": None,
        }

    except Timeout:
        end_ms = time.time() * 1000
        return {
            "raw_output": None,
            "latency_ms": int(end_ms - start_ms),
            "tokens_input": 0,
            "tokens_output": 0,
            "output_truncated": False,
            "input_truncated": False,
            "error": f"timeout_{timeout}s",
        }
    except ReqConnectionError as e:
        return {
            "raw_output": None,
            "latency_ms": -1,
            "tokens_input": 0,
            "tokens_output": 0,
            "output_truncated": False,
            "input_truncated": False,
            "error": f"connection_error: {str(e)[:200]}",
        }
    except Exception as e:
        return {
            "raw_output": None,
            "latency_ms": -1,
            "tokens_input": 0,
            "tokens_output": 0,
            "output_truncated": False,
            "input_truncated": False,
            "error": f"unexpected_error: {str(e)[:200]}",
        }


def query_with_retry(host, model_tag, system_prompt, user_prompt, params,
                     query_id, max_retries=2, timeout=120):
    """Query Ollama with retry logic."""
    for attempt in range(max_retries + 1):
        result = query_ollama(
            host, model_tag, system_prompt, user_prompt, params, timeout
        )
        if result["error"] is None:
            return result
        if attempt < max_retries:
            logger.warning(
                "Query %s attempt %d failed: %s. Retrying...",
                query_id, attempt + 1, result["error"]
            )
            time.sleep(2 * (attempt + 1))  # Simple backoff
        else:
            logger.error(
                "Query %s failed after %d attempts: %s",
                query_id, max_retries + 1, result["error"]
            )
    return result


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def flush_model(host, model_tag):
    """Unload model from GPU memory by setting keep_alive to 0."""
    try:
        requests.post(
            f"{host}/api/generate",
            json={"model": model_tag, "keep_alive": 0},
            timeout=30,
        )
        logger.info("Flushed model %s from GPU memory", model_tag)
        time.sleep(5)
    except Exception as e:
        logger.warning("Failed to flush model %s: %s", model_tag, e)


# ---------------------------------------------------------------------------
# Qwen 3 specific handling
# ---------------------------------------------------------------------------

# Regex to strip <think>...</think> blocks (including partial/unclosed tags)
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>", re.DOTALL
)
_THINK_UNCLOSED_RE = re.compile(
    r"<think>.*", re.DOTALL
)

QWEN3_NO_THINK_PREFIX = "/no_think\n"


def inject_no_think_prefix(user_prompt, model_slug):
    """Prepend /no_think to user prompts for qwen3-4b to disable thinking mode.

    This ensures fair latency and output format comparison across models.
    Only applied when model_slug is 'qwen3-4b'.
    """
    if model_slug == "qwen3-4b":
        return QWEN3_NO_THINK_PREFIX + user_prompt
    return user_prompt


def strip_think_blocks(text):
    """Remove <think>...</think> blocks from model output.

    Applied to qwen3-4b output as a safety net in case /no_think fails.
    Handles both closed and unclosed think tags.
    """
    if text is None:
        return text
    # First strip closed <think>...</think> blocks
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Then strip any unclosed <think>... at the start
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned.strip()


def get_model_overrides(model_slug, inf_params):
    """Return model-specific parameter overrides from inference_params.json.

    For qwen3-4b this includes presence_penalty and num_ctx.
    """
    model_overrides = inf_params.get("model_overrides", {})
    overrides = model_overrides.get(model_slug, {})
    # Remove non-parameter keys
    return {k: v for k, v in overrides.items() if k != "notes"}


def warm_up_model(host, model_tag, prompt_config, params, queries, n=10):
    """Run warmup queries to stabilize GPU state. Results discarded."""
    logger.info("Running %d warmup queries for %s...", n, model_tag)
    system_prompt = prompt_config.get("system_prompt", "")
    user_template = prompt_config.get("user_template",
                                       prompt_config.get("pass1_user_template", ""))

    for i, q in enumerate(queries[:n]):
        user_prompt = user_template.replace("{query}", q["query_text"])
        query_ollama(host, model_tag, system_prompt, user_prompt, params, timeout=120)
        if i == 0:
            logger.info("  Warmup query 1/%d complete", n)
    logger.info("Warmup complete for %s", model_tag)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def get_completed_query_ids(output_path):
    """Read existing responses.jsonl and return set of completed query_ids."""
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record.get("query_id"))
                    except json.JSONDecodeError:
                        continue
    return completed


def append_response(output_path, record):
    """Append a single response record to the JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Build prompt for a query
# ---------------------------------------------------------------------------

def build_prompt(prompt_config, query_text, phi_json=None):
    """Build the full prompt for a query given a prompt config.

    Returns:
        For single-pass: (system_prompt, user_prompt)
        For two-pass: (system_prompt, pass1_user_prompt)
    """
    system_prompt = prompt_config.get("system_prompt", "")

    if "user_template" in prompt_config:
        user_prompt = prompt_config["user_template"].replace("{query}", query_text)
    elif "pass1_user_template" in prompt_config:
        user_prompt = prompt_config["pass1_user_template"].replace("{query}", query_text)
    else:
        raise ValueError("Prompt config has no user_template or pass1_user_template")

    return system_prompt, user_prompt


def build_pass2_prompt(prompt_config, query_text, phi_json_str):
    """Build Pass 2 prompt for P4-llm variant."""
    system_prompt = prompt_config.get("pass2_system_prompt", "")
    user_template = prompt_config.get("pass2_user_template", "")
    user_prompt = user_template.replace("{query}", query_text)
    user_prompt = user_prompt.replace("{phi_json}", phi_json_str)
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Run a single query
# ---------------------------------------------------------------------------

def run_single_query(host, model_tag, model_slug, prompt_config, prompt_slug,
                     query, run_number, params, model_param_overrides=None):
    """Run inference on a single query and return the response record."""
    query_id = query["query_id"]
    query_text = query["query_text"]
    parser = prompt_config.get("output_parser", "direct")
    pass2_mode = prompt_config.get("pass2_mode", None)

    # Get inference params: base -> model overrides -> prompt overrides
    # Model overrides (e.g., qwen3-4b presence_penalty) applied first,
    # then prompt overrides (e.g., P5 num_predict) applied on top.
    prompt_overrides = prompt_config.get("inference_overrides", {})
    merged_overrides = {**(model_param_overrides or {}), **prompt_overrides}
    query_params = {**params, **merged_overrides}

    system_prompt, user_prompt = build_prompt(prompt_config, query_text)

    # Inject /no_think prefix for Qwen 3 to disable thinking mode
    user_prompt = inject_no_think_prefix(user_prompt, model_slug)

    # Base record
    record = {
        "query_id": query_id,
        "model_name": model_slug,
        "model_version": model_tag,
        "prompt_style": prompt_slug,
        "run_number": run_number,
        "input_text": f"[SYSTEM] {system_prompt}\n[USER] {user_prompt}",
        "timestamp_utc": utc_timestamp(),
    }

    if parser == "json_then_replace":
        # P4-prog: Two-pass with programmatic replacement
        result = query_with_retry(
            host, model_tag, system_prompt, user_prompt, query_params,
            query_id, max_retries=2, timeout=120,
        )
        # Strip <think> blocks for Qwen 3 before JSON parsing
        pass1_raw = result["raw_output"]
        if model_slug == "qwen3-4b" and pass1_raw:
            pass1_raw = strip_think_blocks(pass1_raw)
        record["pass1_output"] = pass1_raw
        record["pass1_latency_ms"] = result["latency_ms"]
        record["tokens_input"] = result["tokens_input"]
        record["tokens_output"] = result["tokens_output"]

        if result["error"]:
            record["raw_output"] = None
            record["error"] = result["error"]
            record["latency_ms"] = result["latency_ms"]
        else:
            phi_list = parse_phi_json(pass1_raw)
            if phi_list is None:
                record["raw_output"] = None
                record["error"] = "json_parse_failure"
                record["latency_ms"] = result["latency_ms"]
                record["pass2_method"] = "programmatic"
            else:
                start_p2 = time.time() * 1000
                redacted = apply_redactions(query_text, phi_list)
                end_p2 = time.time() * 1000
                record["raw_output"] = redacted
                record["pass2_output"] = redacted
                record["pass2_latency_ms"] = int(end_p2 - start_p2)
                record["pass2_method"] = "programmatic"
                record["latency_ms"] = result["latency_ms"] + int(end_p2 - start_p2)
                record["error"] = None

    elif parser == "json_then_llm_replace":
        # P4-llm: Two-pass with LLM-based replacement
        result = query_with_retry(
            host, model_tag, system_prompt, user_prompt, query_params,
            query_id, max_retries=2, timeout=120,
        )
        # Strip <think> blocks for Qwen 3 before JSON parsing
        pass1_raw = result["raw_output"]
        if model_slug == "qwen3-4b" and pass1_raw:
            pass1_raw = strip_think_blocks(pass1_raw)
        record["pass1_output"] = pass1_raw
        record["pass1_latency_ms"] = result["latency_ms"]
        record["tokens_input"] = result["tokens_input"]

        if result["error"]:
            record["raw_output"] = None
            record["error"] = result["error"]
            record["latency_ms"] = result["latency_ms"]
        else:
            phi_list = parse_phi_json(pass1_raw)
            if phi_list is None:
                record["raw_output"] = None
                record["error"] = "json_parse_failure"
                record["latency_ms"] = result["latency_ms"]
                record["pass2_method"] = "llm"
            else:
                phi_json_str = "\n".join(
                    json.dumps(p) for p in phi_list
                )
                p2_sys, p2_user = build_pass2_prompt(
                    prompt_config, query_text, phi_json_str
                )
                result2 = query_with_retry(
                    host, model_tag, p2_sys, p2_user, query_params,
                    query_id + "_p2", max_retries=2, timeout=120,
                )
                pass2_raw = result2["raw_output"]
                if model_slug == "qwen3-4b" and pass2_raw:
                    pass2_raw = strip_think_blocks(pass2_raw)
                record["pass2_output"] = pass2_raw
                record["pass2_latency_ms"] = result2["latency_ms"]
                record["pass2_method"] = "llm"
                record["raw_output"] = (
                    parse_direct_output(pass2_raw)
                    if pass2_raw else None
                )
                record["latency_ms"] = result["latency_ms"] + result2["latency_ms"]
                record["tokens_output"] = (
                    result["tokens_output"] + result2.get("tokens_output", 0)
                )
                record["error"] = result2["error"]

    elif parser == "cot_extract":
        # P5: Chain-of-thought with section extraction
        result = query_with_retry(
            host, model_tag, system_prompt, user_prompt, query_params,
            query_id, max_retries=2, timeout=120,
        )
        record["tokens_input"] = result["tokens_input"]
        record["tokens_output"] = result["tokens_output"]
        record["latency_ms"] = result["latency_ms"]
        record["output_truncated"] = result.get("output_truncated", False)

        if result["error"]:
            record["raw_output"] = result["raw_output"]
            record["extracted_output"] = None
            record["reasoning_text"] = None
            record["parse_status"] = ParseResult.EMPTY.value
            record["error"] = result["error"]
        else:
            cot_raw = result["raw_output"]
            # Strip <think> blocks for Qwen 3 before CoT extraction
            if model_slug == "qwen3-4b" and cot_raw:
                cot_raw = strip_think_blocks(cot_raw)
            record["raw_output"] = cot_raw
            extracted, parse_result = extract_cot_output(
                cot_raw, query_text
            )
            record["extracted_output"] = extracted
            record["parse_status"] = parse_result.value
            record["error"] = None

            # Extract reasoning portion
            raw = result["raw_output"] or ""
            reasoning_match = None
            for marker in ["REASONING:", "Reasoning:"]:
                idx = raw.find(marker)
                if idx != -1:
                    reasoning_match = raw[idx:]
                    # Truncate at OUTPUT: or REDACTIONS:
                    for end_marker in ["OUTPUT:", "Output:", "REDACTIONS:", "Redactions:"]:
                        end_idx = reasoning_match.find(end_marker)
                        if end_idx != -1:
                            reasoning_match = reasoning_match[:end_idx]
                            break
                    break
            record["reasoning_text"] = (
                reasoning_match.strip() if reasoning_match else None
            )

    else:
        # P1, P2, P2a, P3: Direct text extraction
        result = query_with_retry(
            host, model_tag, system_prompt, user_prompt, query_params,
            query_id, max_retries=2, timeout=120,
        )
        record["tokens_input"] = result["tokens_input"]
        record["tokens_output"] = result["tokens_output"]
        record["latency_ms"] = result["latency_ms"]
        record["output_truncated"] = result.get("output_truncated", False)

        if result["error"]:
            record["raw_output"] = None
            record["error"] = result["error"]
        else:
            output = result["raw_output"]
            # Strip <think>...</think> blocks for Qwen 3 (safety net)
            if model_slug == "qwen3-4b":
                output = strip_think_blocks(output)
            record["raw_output"] = parse_direct_output(output)
            record["error"] = None

    return record


# ---------------------------------------------------------------------------
# Run a full configuration
# ---------------------------------------------------------------------------

def run_configuration(host, model_tag, model_slug, prompt_config, prompt_slug,
                      run_number, queries, params, output_dir,
                      canary_config=None, model_param_overrides=None):
    """Run all queries for one model x prompt x run configuration.

    Implements checkpointing (resume from last completed query).
    """
    # Output path
    output_path = Path(output_dir) / "raw" / model_slug / prompt_slug / f"run_{run_number}" / "responses.jsonl"

    # Check for existing progress
    completed_ids = get_completed_query_ids(output_path)
    if completed_ids:
        logger.info(
            "Resuming %s/%s/run_%d: %d/%d queries already complete",
            model_slug, prompt_slug, run_number,
            len(completed_ids), len(queries),
        )

    # Run canary at position 0
    if canary_config and "canary_start" not in completed_ids:
        logger.info("Running canary query at position 0...")
        canary_query = {
            "query_id": "canary_start",
            "query_text": canary_config["query"],
        }
        canary_record = run_single_query(
            host, model_tag, model_slug, prompt_config, prompt_slug,
            canary_query, run_number, params,
            model_param_overrides=model_param_overrides,
        )
        canary_record["is_canary"] = True
        canary_record["canary_position"] = 0
        append_response(output_path, canary_record)

    # Run all queries
    total = len(queries)
    start_time = time.time()

    for i, query in enumerate(queries):
        query_id = query["query_id"]
        if query_id in completed_ids:
            continue

        record = run_single_query(
            host, model_tag, model_slug, prompt_config, prompt_slug,
            query, run_number, params,
            model_param_overrides=model_param_overrides,
        )
        append_response(output_path, record)

        # Progress logging
        if (i + 1) % 100 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%s/%s/run_%d] %d/%d (%.1f q/min, ETA: %.0fs)",
                model_slug, prompt_slug, run_number,
                i + 1, total, rate * 60, eta,
            )

    # Run canary at position 1052
    if canary_config and "canary_end" not in completed_ids:
        logger.info("Running canary query at position 1052...")
        canary_query = {
            "query_id": "canary_end",
            "query_text": canary_config["query"],
        }
        canary_record = run_single_query(
            host, model_tag, model_slug, prompt_config, prompt_slug,
            canary_query, run_number, params,
            model_param_overrides=model_param_overrides,
        )
        canary_record["is_canary"] = True
        canary_record["canary_position"] = 1052
        append_response(output_path, canary_record)

    elapsed = time.time() - start_time
    logger.info(
        "Completed %s/%s/run_%d: %d queries in %.1f minutes",
        model_slug, prompt_slug, run_number, total, elapsed / 60,
    )


# ---------------------------------------------------------------------------
# Full experiment matrix
# ---------------------------------------------------------------------------

def resolve_prompt_config(templates, prompt_slug):
    """Find the prompt config matching a slug."""
    for key, config in templates.items():
        if key.startswith("_"):
            continue
        if config.get("slug") == prompt_slug:
            return config
    raise ValueError(f"No prompt config found for slug: {prompt_slug}")


def resolve_model_config(model_configs, model_slug):
    """Find the model config matching a slug."""
    for model in model_configs["models"]:
        if model["slug"] == model_slug:
            return model
    raise ValueError(f"No model config found for slug: {model_slug}")


def get_all_prompt_slugs(templates):
    """Get list of all prompt slugs (excluding ablation and metadata)."""
    slugs = []
    for key, config in templates.items():
        if key.startswith("_"):
            continue
        slug = config.get("slug")
        if slug:
            slugs.append(slug)
    return slugs


def get_primary_prompt_slugs(templates):
    """Get just the 5 primary prompt slugs (no ablations)."""
    primary = [
        "zero-shot-minimal",
        "zero-shot-structured",
        "few-shot",
        "two-pass",
        "chain-of-thought",
    ]
    return [s for s in primary if any(
        c.get("slug") == s for k, c in templates.items() if not k.startswith("_")
    )]


def run_full_matrix(host, output_dir, include_ablations=True):
    """Run the complete model x prompt x run matrix."""
    dataset = load_dataset()
    templates = load_prompt_templates()
    model_configs = load_model_configs()
    inf_params = load_inference_params()

    queries = dataset["queries"]
    default_params = inf_params["default"]
    canary_config = inf_params.get("canary")
    runs_per_config = inf_params["execution"]["runs_per_config"]
    warmup_n = inf_params["execution"]["warmup_queries"]
    cooldown = inf_params["execution"]["cooldown_between_models_seconds"]

    # Determine which prompts to run
    if include_ablations:
        prompt_slugs = get_all_prompt_slugs(templates)
    else:
        prompt_slugs = get_primary_prompt_slugs(templates)

    model_slugs = [m["slug"] for m in model_configs["models"]]

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT MATRIX")
    logger.info("Models: %s", model_slugs)
    logger.info("Prompts: %s", prompt_slugs)
    logger.info("Runs per config: %d", runs_per_config)
    logger.info("Total configurations: %d", len(model_slugs) * len(prompt_slugs) * runs_per_config)
    logger.info("Host: %s", host)
    logger.info("=" * 60)

    for model_slug in model_slugs:
        model_config = resolve_model_config(model_configs, model_slug)
        model_tag = model_config["ollama_tag"]

        logger.info("--- Starting model: %s (%s) ---", model_slug, model_tag)

        # Get model-specific parameter overrides (e.g., qwen3-4b presence_penalty)
        model_param_overrides = get_model_overrides(model_slug, inf_params)
        if model_param_overrides:
            logger.info("  Model overrides for %s: %s", model_slug, model_param_overrides)

        for prompt_slug in prompt_slugs:
            prompt_config = resolve_prompt_config(templates, prompt_slug)

            # Warmup before first run of this prompt
            warm_up_model(
                host, model_tag, prompt_config, default_params, queries, n=warmup_n
            )

            for run_num in range(1, runs_per_config + 1):
                logger.info(
                    "Starting: %s / %s / run_%d",
                    model_slug, prompt_slug, run_num,
                )
                run_configuration(
                    host=host,
                    model_tag=model_tag,
                    model_slug=model_slug,
                    prompt_config=prompt_config,
                    prompt_slug=prompt_slug,
                    run_number=run_num,
                    queries=queries,
                    params=default_params,
                    output_dir=output_dir,
                    canary_config=canary_config,
                    model_param_overrides=model_param_overrides,
                )

            # Flush KV cache between prompt strategies
            flush_model(host, model_tag)

        # Cooldown between models
        logger.info("Model %s complete. Cooling down %ds...", model_slug, cooldown)
        flush_model(host, model_tag)
        time.sleep(cooldown)

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT MATRIX COMPLETE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SLM De-Identification Evaluation Runner"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model slug (phi4-mini, llama32-3b, qwen3-4b, gemma3-4b). Required unless --all is set."
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt slug (e.g., zero-shot-structured). Required unless --all."
    )
    parser.add_argument(
        "--run", type=int, default=1,
        help="Run number (default: 1)."
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST})."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(BASE_DIR),
        help="Root output directory (default: slm-evaluation/)."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run the full model x prompt x run matrix."
    )
    parser.add_argument(
        "--include-ablations", action="store_true",
        help="Include P2a and P4-llm ablation conditions in --all mode."
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (default: stdout only)."
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file or str(
        Path(args.output_dir) / f"runner_{utc_timestamp().replace(':', '-')}.log"
    )
    setup_logging(log_file=log_file)

    if args.all:
        run_full_matrix(
            host=args.host,
            output_dir=args.output_dir,
            include_ablations=args.include_ablations,
        )
    elif args.model and args.prompt:
        # Single configuration run
        dataset = load_dataset()
        templates = load_prompt_templates()
        model_configs = load_model_configs()
        inf_params = load_inference_params()

        model_config = resolve_model_config(model_configs, args.model)
        prompt_config = resolve_prompt_config(templates, args.prompt)
        default_params = inf_params["default"]
        canary_config = inf_params.get("canary")
        model_param_overrides = get_model_overrides(args.model, inf_params)

        # Warmup
        warm_up_model(
            args.host, model_config["ollama_tag"], prompt_config,
            default_params, dataset["queries"],
            n=inf_params["execution"]["warmup_queries"],
        )

        run_configuration(
            host=args.host,
            model_tag=model_config["ollama_tag"],
            model_slug=args.model,
            prompt_config=prompt_config,
            prompt_slug=args.prompt,
            run_number=args.run,
            queries=dataset["queries"],
            params=default_params,
            output_dir=args.output_dir,
            canary_config=canary_config,
            model_param_overrides=model_param_overrides,
        )
    else:
        parser.error("Either --all or both --model and --prompt are required.")


if __name__ == "__main__":
    main()
