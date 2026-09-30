import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=14)
    parser.add_argument("--model", default="models/llama-3.2-1b-instruct")
    parser.add_argument("--tokens", default="src/tokens.json")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be positive")

    prompt = json.loads(Path(args.tokens).read_text())
    with tempfile.TemporaryDirectory() as directory:
        dump_path = Path(directory) / "logits.json"
        env = dict(os.environ, TINY_VLLM_LOGITS_DUMP=str(dump_path))
        print(f"Running engine through generated token {args.step}...", flush=True)
        result = subprocess.run(
            ["./build/tiny_vLLM", str(Path(args.model) / "model.safetensors"),
             args.tokens, str(args.step), "1024"],
            env=env, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        dump = json.loads(dump_path.read_text())

    generated = dump["generated_tokens"]
    if len(generated) != args.step:
        raise RuntimeError("Engine stopped before the requested token")
    prefix = prompt + generated[:-1]
    if dump["prefix"] != prefix:
        raise RuntimeError("Dump does not match the expected input prefix")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading reference and replaying the engine's exact prefix...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference_choices = []
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([prompt], dtype=torch.long, device="cuda"),
            use_cache=True,
        )
        reference_choices.append(int(output.logits[0, -1].argmax()))
        for token in generated[:-1]:
            output = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device="cuda"),
                past_key_values=output.past_key_values, use_cache=True,
            )
            reference_choices.append(int(output.logits[0, -1].argmax()))
        reference = output.logits[0, -1].float().cpu()

    actual = torch.tensor(dump["logits"], dtype=torch.float32)
    if actual.shape != reference.shape:
        raise RuntimeError("Vocabulary sizes differ")
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("Non-finite logits detected")
    earlier = [i + 1 for i, (a, b) in enumerate(
        zip(generated[:-1], reference_choices[:-1])) if a != b]
    print("Earlier greedy mismatches on this shared path:", earlier)
    print("Prefix token count:", len(prefix))
    print("Engine selected:", generated[-1])
    print("Engine logits argmax:", int(actual.argmax()))
    print("Reference logits argmax:", int(reference.argmax()))
    for name, logits in [("Reference", reference), ("Engine", actual)]:
        values, indices = logits.topk(5)
        print(f"{name} top 5:")
        for token, value in zip(indices.tolist(), values.tolist()):
            text = tokenizer.decode([token], clean_up_tokenization_spaces=False)
            print(f"  {token:6d}  {value:10.6f}  {text!r}")
        print(f"{name} top-1 minus top-2: {float(values[0] - values[1]):.6f}")
    for token in sorted({int(reference.argmax()), int(actual.argmax())}):
        print(f"Token {token}: reference={reference[token]:.6f}, engine={actual[token]:.6f}")
    error = actual - reference
    print(f"Mean absolute logit error: {error.abs().mean():.6f}")
    print(f"Maximum absolute logit error: {error.abs().max():.6f}")
    print(f"Logit RMSE: {error.square().mean().sqrt():.6f}")
    rounded = actual.to(torch.bfloat16).float()
    rounded_choice = int(rounded.argmax())
    reference_choice = int(reference.argmax())
    ties = torch.nonzero(rounded == rounded.max(), as_tuple=True)[0].tolist()
    print("Engine argmax after BF16 rounding:", rounded_choice)
    print("Matches reference after BF16 rounding:", rounded_choice == reference_choice)
    print("Engine tokens tied for maximum after BF16 rounding:", ties)
    for token in sorted({int(actual.argmax()), rounded_choice, reference_choice}):
        print(
            f"Token {token}: engine FP32={actual[token]:.6f}, "
            f"engine BF16={rounded[token]:.6f}, reference={reference[token]:.6f}"
        )
    print("BF16 rounding is diagnostic only; generated tokens were not changed.")
    print("These diagnostics alone do not establish a rounding-only cause.")


if __name__ == "__main__":
    main()
