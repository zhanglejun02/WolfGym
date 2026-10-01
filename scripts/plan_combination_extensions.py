"""Write extension contracts and seed metadata only; no simulations or API."""
import argparse
import json
from pathlib import Path

from wolfgym.benchmark.combination_extensions import plan_extensions, sha, ROOT


def main(output):
    plan = plan_extensions()
    output.mkdir(parents=True, exist_ok=False)
    for name in plan['contracts']:
        (output/f'{name}_contract.json').write_text(json.dumps(plan['contracts'][name], indent=2)+'\n')
        (output/f'{name}_manifest.json').write_text(json.dumps(plan['manifests'][name], indent=2)+'\n')
    (output/'seed_audit.json').write_text(json.dumps({k: plan[k] for k in ('historical_inventory','seed_check')}, indent=2)+'\n')
    sources = [ROOT/'src/wolfgym/benchmark/combination_extensions.py', Path(__file__),
               ROOT/'src/wolfgym/env/combination_contract.py', ROOT/'src/wolfgym/env/combination_v51.json',
               ROOT/'src/wolfgym/benchmark/combination_splits_v1.json',
               ROOT/'docs/EXPERIMENT_PROTOCOL.md']
    receipt = {'version': plan['version'], 'runtime_release': 'closed', 'market_episodes_run': 0,
               'formal_confirmation_episodes': 0, 'source_sha256': {str(p.resolve()):sha(p) for p in sources},
               'plan_sha256': {p.name:sha(p) for p in output.glob('*.json')}}
    (output/'plan_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'seed_check':plan['seed_check'], 'runtime_release':'closed',
                      'historical_seed_values':len(plan['historical_inventory']['seed_numeric_values'])}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
