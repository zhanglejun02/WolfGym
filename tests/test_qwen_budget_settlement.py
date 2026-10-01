import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from settle_combination_qwen_pilot import allocation

def test_unused_allowance_transfer_preserves_total_and_existing_charges():
    charges=[2210432,2406314,2390000,2400000]
    for i in range(4):
        caps=allocation(charges,i)
        assert sum(caps)==10000000
        assert all(caps[j]==charges[j] for j in range(4) if j!=i)
        assert caps[i]>=charges[i]

def test_overspend_is_not_reset_by_transfer():
    with pytest.raises(ValueError):allocation([3000000]*4,1)
    with pytest.raises(ValueError):allocation([-1,0,0,0],1)
