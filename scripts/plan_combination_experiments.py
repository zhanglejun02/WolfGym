#!/usr/bin/env python
"""Write/check the joint draft inventory without constructing a market or calling APIs."""
import argparse
from pathlib import Path

from wolfgym.benchmark.combination_experiments import ADDITIONAL_DIAGNOSTICS, make_draft, preflight
from wolfgym.evaluation.combination_host import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--additional-diagnostic', choices=ADDITIONAL_DIAGNOSTICS,
                        help='explicit additional diagnostic choice for a new draft')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = args.candidate or args.output/'candidate.json'
    if args.candidate and not candidate.is_file():
        raise ValueError('requested candidate does not exist')
    if not candidate.exists():
        write_json(candidate, make_draft(args.additional_diagnostic))
    plan = read_json(candidate)
    if args.additional_diagnostic is not None and plan.get('additional_diagnostic') != args.additional_diagnostic:
        raise ValueError('existing candidate differs from --additional-diagnostic; preflight cannot change a saved configuration')
    result = preflight(plan)
    write_json(args.output/'preflight.json', result)
    lines = ['# 联合实验清单预检（不启动实验）', '',
             '当前仅允许工程入口；完整填写配置也不等于费用授权或真实供应商验收。', '',
             '| 模型席位 | 主研究常规请求上限 |', '| --- | ---: |']
    lines += [f'| {s} | {n} |' for s,n in result['regular_requests_by_model'].items()]
    lines += ['', f"附加诊断选择：{result['additional_diagnostic'] if result['additional_diagnostic'] is not None else '尚未选择'}。",
              f"合计 {result['total_regular_research_requests']} 次；不含成本试跑或最多两次传输重试。",
              'RR 复用公开刷新规则，LL 复用主表。成本试跑按前 8 个完整 train 社会，共 16 个 episode；最多两个模型，每模型最多 480 次常规请求，多版提示共享额度。',
              '', '## 未完成项', '']
    lines += [f'- {s}' for s in result['blockers']+result['remaining_release_requirements']]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f"offline preflight: {len(result['blockers'])} configuration blockers; research runtime remains closed")


if __name__ == '__main__':
    main()
