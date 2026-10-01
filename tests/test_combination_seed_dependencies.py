import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('seed_audit',Path(__file__).resolve().parents[1]/'scripts/audit_combination_seed_dependencies.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_retained_edges_form_thirteen_disjoint_chains():
    roots=list(range(40000000,40000128))
    edges=audit.surviving_edges(roots)
    assert len(edges)==115
    assert all(b-a==13 and a%13==b%13 for a,b in edges)
    # Full chain membership, not each episode or adjacent edge treated as IID.
    assert len({s%13 for s in roots})==13


def test_chain_sensitivity_retains_perfect_within_chain_dependence():
    seeds=np.arange(40000000,40000130)
    values=(seeds%13).astype(float)
    r=audit.chain_sensitivity(seeds,values)
    assert r['mean_pp']==pytest.approx(6)
    chain=r['chain_bootstrap_ci95_pp']; iid=r['iid_bootstrap_sensitivity_ci95_pp']
    assert chain[1]-chain[0] > 2*(iid[1]-iid[0])
    assert not r['coverage_certified']
    assert r==audit.chain_sensitivity(seeds[::-1],values[::-1])


def test_constant_contrast_and_unequal_chain_sizes_preserve_mean():
    seeds=np.arange(50000000,50000256)
    r=audit.chain_sensitivity(seeds,np.full(256,-.25))
    assert r['mean_pp']==-.25 and r['chain_bootstrap_ci95_pp']==[-.25,-.25]
    assert sorted(set(r['chain_sizes']))==[19,20]


def test_incomplete_or_nonfinite_series_rejected():
    with pytest.raises(ValueError):audit.chain_sensitivity([1]*30,[0]*30)
    with pytest.raises(ValueError):audit.chain_sensitivity(list(range(29))+[30],[0]*30)
    with pytest.raises(ValueError):audit.chain_sensitivity(list(range(30)),[float('nan')]*30)
