import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch


def compare_step(actual, reference, selected):
    if actual.shape != reference.shape:
        raise RuntimeError("Vocabulary sizes differ")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("Non-finite logits")
    raw_choice = int(actual.argmax())
    rounded = actual.to(torch.bfloat16).float()
    rounded_choice = int(rounded.argmax())
    ref_choice = int(reference.argmax())
    error = actual - reference
    values, indices = reference.topk(5)
    engine_values, engine_indices = actual.topk(5)
    return {
        "engine_selected": selected,
        "engine_argmax": raw_choice,
        "reference_argmax": ref_choice,
        "rounded_argmax": rounded_choice,
        "selection_matches_bf16_logits": selected == rounded_choice,
        "engine_match": selected == ref_choice,
        "raw_match": raw_choice == ref_choice,
        "bf16_match": rounded_choice == ref_choice,
        "reference_margin": float(values[0] - values[1]),
        "reference_regret": float(reference[ref_choice] - reference[raw_choice]),
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "max_error": float(error.abs().max()),
        "top5_overlap": len(set(indices.tolist()) & set(engine_indices.tolist())),
        "reference_top5": list(zip(indices.tolist(), values.tolist())),
        "engine_top5": list(zip(engine_indices.tolist(), engine_values.tolist())),
    }


def build_cases(tokenizer):
    cases = []
    for name, text in [
        ("france", "The capital of France is"),
        ("arithmetic", "Two plus two equals"),
        ("code", "def fibonacci(n):\n"),
        ("story", "Once upon a time, a small robot discovered a garden."),
    ]:
        cases.append({"name": name, "prompt": tokenizer.encode(text)})
    messages = [{"role": "user", "content": "Explain why the sky looks blue in three sentences."}]
    cases.append({"name": "chat", "prompt": tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
    )})
    opening = tokenizer.encode("Read this notebook carefully.\n")
    body = tokenizer.encode(
        "The notebook describes rivers, mountains, trees, and a quiet village.\n",
        add_special_tokens=False,
    )
    ending = tokenizer.encode("\nSummarize the notebook:", add_special_tokens=False)
    room = 900 - len(opening) - len(ending)
    long_prompt = opening + (body * (room // len(body) + 1))[:room] + ending
    cases.append({"name": "long_context", "prompt": long_prompt})
    return cases


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/llama-3.2-1b-instruct")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", default="validation_report.json")
    args = parser.parse_args()
    if not 1 <= args.steps <= 124:
        parser.error("--steps must be between 1 and 124 for the 900-token case")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = build_cases(tokenizer)
    report = {
        "method": "Reference uses the engine's exact prefix at every step; this is not free-running sequence agreement.",
        "model": args.model, "steps_requested": args.steps,
        "selection_precision": "BF16, round to nearest even, lowest token ID wins ties",
        "torch_version": torch.__version__, "cases": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for case in cases:
            name = case["name"]
            tokens_path = root / f"{name}.tokens.json"
            dump_path = root / f"{name}.dump.json"
            trace_path = root / f"{name}.f32"
            tokens_path.write_text(json.dumps(case["prompt"]))
            print(f"Engine: {name}, {len(case['prompt'])} prompt tokens, up to {args.steps} new tokens", flush=True)
            env = dict(os.environ, TINY_VLLM_LOGITS_DUMP=str(dump_path),
                       TINY_VLLM_LOGITS_TRACE=str(trace_path))
            result = subprocess.run(
                ["./build/tiny_vLLM", str(Path(args.model) / "model.safetensors"),
                 str(tokens_path), str(args.steps), "1024"],
                env=env, capture_output=True, text=True, timeout=600,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-6000:] + result.stderr[-6000:])
            dump = json.loads(dump_path.read_text())
            generated = dump["generated_tokens"]
            if not generated or dump["prefix"] != case["prompt"] + generated[:-1]:
                raise RuntimeError(f"Invalid prefix dump for {name}")
            case["generated"] = generated
            case["vocab_size"] = len(dump["logits"])
            expected = len(generated) * case["vocab_size"] * 4
            if trace_path.stat().st_size != expected:
                raise RuntimeError(f"Wrong trace size for {name}; rebuild the engine")
            case["trace"] = trace_path

        print("Loading Hugging Face reference once for all cases...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True,
        ).to("cuda").eval()
        for case in cases:
            generated = case["generated"]
            trace = np.fromfile(case["trace"], dtype="<f4").reshape(len(generated), -1)
            rows = []
            with torch.inference_mode():
                output = model(input_ids=torch.tensor([case["prompt"]], device="cuda"), use_cache=True)
                for i, token in enumerate(generated):
                    reference = output.logits[0, -1].float().cpu()
                    row = compare_step(torch.from_numpy(trace[i]), reference, token)
                    row["step"] = i + 1
                    row["prefix_length"] = len(case["prompt"]) + i
                    rows.append(row)
                    if i + 1 < len(generated):
                        output = model(
                            input_ids=torch.tensor([[token]], device="cuda"),
                            past_key_values=output.past_key_values, use_cache=True,
                        )
            del output
            raw_mismatches = [row["step"] for row in rows if not row["raw_match"]]
            bf16_mismatches = [row["step"] for row in rows if not row["bf16_match"]]
            summary = {
                "name": case["name"], "prompt_tokens": case["prompt"],
                "generated_tokens": generated,
                "generated_text": tokenizer.decode(generated, clean_up_tokenization_spaces=False),
                "raw_mismatch_steps": raw_mismatches, "bf16_mismatch_steps": bf16_mismatches,
                "selection_mismatch_steps": [row["step"] for row in rows if not row["selection_matches_bf16_logits"]],
                "engine_mismatch_steps": [row["step"] for row in rows if not row["engine_match"]],
                "mean_mae": sum(row["mae"] for row in rows) / len(rows),
                "max_error": max(row["max_error"] for row in rows), "steps": rows,
            }
            report["cases"].append(summary)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(f"{case['name']}: engine {len(rows) - len(summary['engine_mismatch_steps'])}/{len(rows)}, "
                  f"raw FP32 {len(rows) - len(raw_mismatches)}/{len(rows)}, "
                  f"BF16 {len(rows) - len(bf16_mismatches)}/{len(rows)}, "
                  f"mean MAE {summary['mean_mae']:.5f}, max error {summary['max_error']:.5f}", flush=True)
            print(f"  Raw mismatch steps: {raw_mismatches}", flush=True)
            print(f"  BF16 mismatch steps: {bf16_mismatches}", flush=True)
    print(f"Report saved to {args.output}. Agreement is diagnostic, not a proof of correctness.", flush=True)


if __name__ == "__main__":
    main()
