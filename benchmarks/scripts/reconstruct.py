"""Reconstruct an isolated historical build tree; never modify the checkout."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['gpu', 'cpu', 'cpu-profile', 'post-fix'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        parser.error('Output must not exist; use a new directory.')
    stages = json.loads((root/'benchmarks/results/provenance/revisions.json').read_text())['stages']
    stage = stages[args.stage]
    patch = root/stage['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != stage['patch_sha256']:
        raise ValueError('Reconstruction patch checksum mismatch')
    output.mkdir(parents=True)
    for directory in ['include', 'src', 'tests']:
        shutil.copytree(root/directory, output/directory, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root/'CMakeLists.txt', output/'CMakeLists.txt')
    (output/'benchmarks').mkdir()
    for name in ['benchmark.cu', 'operation_profiler.cu', 'prompts.json']:
        shutil.copy2(root/'benchmarks'/name, output/'benchmarks'/name)
    shutil.copytree(root/'benchmarks/configs', output/'benchmarks/configs')
    subprocess.run(['patch', '-p1', '--batch', '-i', str(patch)], cwd=output, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for name, expected in stage['source_sha256'].items():
        actual = hashlib.sha256((output/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Historical source checksum mismatch: '+name)
    print(json.dumps({'stage':args.stage, 'verified_build_input_files':len(stage['source_sha256']),
                      'build_input_tree_sha256':stage['build_input_tree_sha256']}))


if __name__ == '__main__':
    main()
