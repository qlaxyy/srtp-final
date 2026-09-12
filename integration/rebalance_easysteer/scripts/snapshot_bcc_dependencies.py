"""Small standard-library-only source snapshot. Never imports torch or loads weights."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import tarfile


FILES = {
    'transformers': [
        'transformers/models/qwen2/modeling_qwen2.py',
        'transformers/models/qwen2/configuration_qwen2.py',
        'transformers/modeling_attn_mask_utils.py',
        'transformers/modeling_rope_utils.py',
        'transformers/modeling_utils.py',
        'transformers/cache_utils.py',
        'transformers/utils/import_utils.py',
    ],
    'torch': ['torch/nn/modules/module.py', 'torch/nn/modules/linear.py',
              'torch/nn/functional.py', 'torch/backends/cuda/__init__.py'],
}
MAX_BYTES = 8 * 1024 * 1024
MODEL_CONFIG_SHA256 = '37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(output, model, resolver=metadata.distribution):
    output = Path(output)
    archive = output.with_suffix('.tar.gz')
    if output.exists() or archive.exists():
        raise ValueError('New output and archive paths required; no overwrite/resume')
    records, versions, total = [], {}, 0
    for package, names in FILES.items():
        dist = resolver(package)
        versions[package] = dist.version
        for name in names:
            source = Path(dist.locate_file(name)).resolve()
            size = source.stat().st_size
            if size > MAX_BYTES or total + size > MAX_BYTES:
                raise ValueError('Source snapshot exceeds 8 MiB cap')
            data = source.read_bytes()
            records.append((name, source, data))
            total += len(data)
    config = Path(model).resolve()/'config.json'
    if config.stat().st_size > 65536:
        raise ValueError('Model config too large')
    data = config.read_bytes()
    if digest(data) != MODEL_CONFIG_SHA256:
        raise ValueError('Model config differs from original BCC model')
    records.append(('model/config.json', config, data))
    if sum(len(data) for _, _, data in records) > MAX_BYTES:
        raise ValueError('Total cap')
    output.mkdir(parents=True)
    manifest = dict(status='static_installed_source_snapshot_not_runtime_module_verification',
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    distributions=versions, model_weights_read=False, framework_imports=False,
                    script_sha256=digest(Path(__file__).read_bytes()), files=[])
    try:
        for name, source, data in records:
            target = output/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest['files'].append(dict(path=name, original_path=str(source), bytes=len(data), sha256=digest(data)))
        manifest['total_source_bytes'] = sum(row['bytes'] for row in manifest['files'])
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        # Explicit filenames only: no environment recursion or credential/config collection.
        with tarfile.open(archive, 'x:gz') as tar:
            for name in [r['path'] for r in manifest['files']] + ['manifest.json']:
                tar.add(output/name, arcname=name, recursive=False)
    except BaseException as error:
        (output/'incomplete.json').write_text(json.dumps({'error':repr(error)}), encoding='utf-8')
        raise
    return dict(archive=str(archive), sha256=digest(archive.read_bytes()),
                source_files=len(records), total_source_bytes=manifest['total_source_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path,
        default=Path('/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B'))
    args = parser.parse_args()
    print(json.dumps(snapshot(args.output, args.model)))


if __name__ == '__main__':
    main()
