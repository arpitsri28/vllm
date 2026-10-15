"""Optional same-checkpoint eager BF16 baseline; no speedup claim implied."""
import argparse
import json
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM


def run(model, ids, budget):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    # Match engine selection: BF16 rounded logits, CPU argmax, lowest-ID tie break.
    token = int(output.logits[0,-1].to(torch.bfloat16).float().cpu().argmax())
    times = [(time.perf_counter()-start)*1000]
    tokens = [token]
    cache = output.past_key_values
    del output
    while len(tokens)<budget and token not in (128001,128008,128009):
        output = model(input_ids=torch.tensor([[token]],device='cuda'), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        token = int(output.logits[0,-1].to(torch.bfloat16).float().cpu().argmax())
        tokens.append(token); times.append((time.perf_counter()-start)*1000)
        cache = output.past_key_values
        del output
    free, total = torch.cuda.mem_get_info()
    observed = (total-free)/1048576
    allocated = torch.cuda.max_memory_allocated()/1048576
    reserved = torch.cuda.max_memory_reserved()/1048576
    del cache
    torch.cuda.synchronize()
    elapsed=(time.perf_counter()-start)*1000
    decode=times[-1]-times[0]
    return dict(status='ok' if len(tokens)==budget else 'early_termination',
                actual_output_tokens=len(tokens), generated_tokens=tokens, token_timestamps_wall_ms=times,
                ttft_wall_ms=times[0], decode_wall_ms=decode,
                decode_wall_tokens_per_s=(len(tokens)-1)*1000/decode if decode else 0,
                e2e_wall_ms=elapsed, e2e_output_tokens_per_s=len(tokens)*1000/elapsed,
                torch_peak_allocated_mib=allocated, torch_peak_reserved_mib=reserved,
                observed_device_used_mib=observed)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='models/llama-3.2-1b-instruct')
    p.add_argument('--output',default='benchmarks/local/hf.jsonl'); a=p.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,
          attn_implementation='eager',local_files_only=True).to('cuda').eval()
    # New Transformers versions specify cache policy through the model generation config.
    ids=torch.tensor([json.loads(Path('benchmarks/prompts.json').read_text())['256']],device='cuda')
    metadata=dict(type='metadata',implementation='Hugging Face explicit greedy forward loop',
        torch=torch.__version__,transformers=transformers.__version__,precision='BF16',attention='eager',
        cache='default Hugging Face dynamic cache; no compilation',logits_to_keep=1,
        gpu=torch.cuda.get_device_name(),warmups=2,repetitions=5,prompt_tokens=256,output_tokens=128,
        selection='CPU argmax after BF16 rounding; engine EOS IDs',prompt_upload='outside timed interval',
        limitations='Dynamic HF cache vs custom preallocated 1024-token cache; framework Python overhead; memory allocator statistics are not interchangeable with device-wide usage. No relative advantage claimed.')
    with open(a.output,'w') as f, torch.inference_mode():
        f.write(json.dumps(metadata)+'\n'); f.flush()
        for _ in range(2): run(model,ids,128)
        for rep in range(5):
            row=dict(type='run',repetition=rep,**run(model,ids,128))
            f.write(json.dumps(row)+'\n'); f.flush()
            print(rep,row['status'],row['decode_wall_tokens_per_s'],flush=True)

if __name__=='__main__': main()
