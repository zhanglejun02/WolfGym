import sys
from pathlib import Path
import pytest
from combination_qwen_fixtures import reference_wait_artifact
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_combination_qwen_dev32 as runner
import report_combination_qwen_network_continuation as reporter
import test_combination_qwen_incomplete_recovery as fixtures
from wolfgym.evaluation.combination_qwen_dev32 import DevelopmentJournal
from wolfgym.evaluation.combination_qwen_network_retry import NetworkRetryJournal
from wolfgym.evaluation.combination_qwen_recovery import ledger_snapshot
from wolfgym.evaluation.combination_host import file_hash
from wolfgym.evaluation.combination_model_host import durable_json
from wolfgym.evaluation.combination_public import digest
from wolfgym.evaluation.combination_requests import TransportFailure
from wolfgym.benchmark.combination import episode_plan


def test_fourth_attempt_full_host_replay_and_ledger_audit(tmp_path,monkeypatch):
    identity=next(e.to_dict() for e in episode_plan('train') if e.condition=='normal')
    baseline=tmp_path/'baseline'
    artifact=reference_wait_artifact(identity)
    durable_json(baseline/'reference_wait.json.gz', {
        'reference':artifact['accounts']['reference'], 'wait':artifact['accounts']['wait'],
        'reference_rows':artifact['financial_rows']['reference'],
        'wait_rows':artifact['financial_rows']['wait'],
    })
    # The continuation reporter reexports the dev32 audit; patch its defining module's dependency.
    monkeypatch.setitem(reporter.audit_episode.__globals__, 'baseline_directory', lambda _:baseline)
    m={'episodes':[identity],'jobs':[{'id':'LL_0','mode':'LL','shard':0}],
       'max_requests_total':2048,'max_cost_microusd_total':60000000}
    durable_json(tmp_path/'manifest.json',m);job=m['jobs'][0];calls=[]
    with DevelopmentJournal(tmp_path,m,job,pause=lambda _:None,transport_factory=fixtures.factory(['empty']*3,calls)) as j:
        with pytest.raises(TransportFailure):runner.run_one(tmp_path,identity,m,job,j,'mock')
    key=digest([digest([digest(m),'LL',identity]),0])
    s={'manifest_sha256':digest(m),'source_sha256':{},
       'acknowledged_stop_sha256':file_hash(tmp_path/'incomplete_recovery_stop.json'),
       'original_ledgers':{'LL_0':ledger_snapshot(tmp_path/'LL_0.sqlite')},
       'one_extra_attempt_keys':{'LL_0':[key]}}
    durable_json(tmp_path/'network_retry_supplement.json',s)
    with NetworkRetryJournal(tmp_path,m,job,s,pause=lambda _:None,transport_factory=fixtures.factory(['ok']*30,calls)) as j:
        runner.run_one(tmp_path,identity,m,job,j,'mock')
    assert len(calls)==33
    usage,receipts,totals=reporter.audit_ledgers(tmp_path,m)
    assert totals['attempts']==33 and totals['successful_attempts']==30
    assert totals['unknown_reserved_microusd']==330000
    row=reporter.audit_episode((str(tmp_path),m,'LL',identity))
    assert row['mode']=='LL'
    s['one_extra_attempt_keys']={};durable_json(tmp_path/'network_retry_supplement.json',s)
    with pytest.raises(ValueError,match='retry bound'):reporter.audit_ledgers(tmp_path,m)
