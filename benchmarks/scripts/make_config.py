"""Regenerate prompt fixtures and optional runnable configs into a separate directory."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/llama-3.2-1b-instruct')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        parser.error('Output must be a new directory; preserved fixtures are never overwritten.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    opening = tokenizer.encode('Read this notebook carefully.\n')
    body = tokenizer.encode('The notebook describes rivers, mountains, trees, and a quiet village.\n', add_special_tokens=False)
    ending = tokenizer.encode('\nContinue the notebook with a long detailed description of the village and its history:\n', add_special_tokens=False)
    prompts = {}
    for length in (128, 256, 512):
        room = length-len(opening)-len(ending)
        assert room > 0
        prompts[str(length)] = opening+(body*(room//len(body)+1))[:room]+ending
    out.mkdir(parents=True)
    (out/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
    config = dict(prompts_file=str(out/'prompts.json'),max_seq_len=1024,seed=0,
                  eos_policy='preserve engine EOS stopping; incomplete runs excluded',warmups=2,repetitions=5,
                  cases=[dict(mode='contiguous',prompt_tokens=256,output_tokens=128,batch_size=1)])
    (out/'contiguous.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__ == '__main__':
    main()
