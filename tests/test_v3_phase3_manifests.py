from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_tool():
    path = Path(__file__).resolve().parent.parent / "tools" / "create_v3_phase3_manifests.py"
    spec = importlib.util.spec_from_file_location("create_v3_phase3_manifests_for_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_phase3_split_is_deterministic_disjoint_and_complete():
    instances = []
    pilot_ids = []
    for question_type, dev_count in tool.DEV_QUOTAS.items():
        for index in range(dev_count + 3):
            question_id = f"{question_type}-{index}"
            instances.append(SimpleNamespace(question_id=question_id, question_type=question_type))
        pilot_id = f"{question_type}-pilot"
        instances.append(SimpleNamespace(question_id=pilot_id, question_type=question_type))
        pilot_ids.append(pilot_id)
    train_ids = tuple(instance.question_id for instance in instances)

    first = tool.split_ids(instances, train_ids, tuple(pilot_ids))
    second = tool.split_ids(instances, train_ids, tuple(pilot_ids))
    tune, dev = first

    assert first == second
    assert len(dev) == sum(tool.DEV_QUOTAS.values()) == 60
    assert set(tune).isdisjoint(dev)
    assert set(tune).isdisjoint(pilot_ids)
    assert set(tune) | set(dev) | set(pilot_ids) == set(train_ids)
