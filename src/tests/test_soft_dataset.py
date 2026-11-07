"""Soft-data regressions: split stability, independent physics, loader safety."""
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from dataset.audit import pixel_hash
from dataset.audit_soft_dataset import inspect_soft
from dataset.generate_soft_dataset import generate
from model.data_connector import (
    ManifestQuartetDataset,
    build_dataloaders,
    load_manifest,
)


@pytest.fixture(scope='module')
def soft_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('soft')/'dataset'
    generate(root, 5, 5, 5, master_seed=881177, teacher_iters=3)
    return root


def test_valid_loader_and_physics(soft_root):
    report = inspect_soft(soft_root/'manifest.json')
    assert report['errors'] == []
    assert report['soft_checks']['unique_origins'] == 15
    assert report['soft_checks']['unique_reference_phases'] == 6
    assert report['soft_checks']['witness_target_max_lsb_error'] == 0
    data = load_manifest(soft_root/'manifest.json')
    assert data.report.counts == {'diagnostic': 1, 'train': 5, 'validation': 5, 'test': 5}
    loaders = build_dataloaders(soft_root/'manifest.json', batch_size=2, num_workers=0, pin_memory=False)
    assert next(iter(loaders['train']))[0].shape == (2, 4, 128, 128)


def test_frozen_splits_when_train_extended(soft_root, tmp_path):
    other = tmp_path/'more'
    generate(other, 10, 5, 5, master_seed=881177, teacher_iters=3)
    m = json.loads((soft_root/'manifest.json').read_text())
    other_m = json.loads((other/'manifest.json').read_text())
    lookup = {s['sample_id']: s for s in other_m['samples']}
    for s in m['samples']:
        assert lookup[s['sample_id']] == s
        for p in s['images'] + ([s['params']['reference_phase']] if 'reference_phase' in s['params'] else []):
            assert pixel_hash(soft_root/p) == pixel_hash(other/p)


def test_physical_augmentation_rejected(soft_root):
    for kwargs in [{'augment_channel_shuffle': True}, {'augment_dihedral': True}]:
        with pytest.raises(ValueError, match='physical_forward'):
            ManifestQuartetDataset(soft_root/'manifest.json', **kwargs)


def test_corrupted_reference_phase_fails(soft_root, tmp_path):
    other = tmp_path/'corrupt'
    shutil.copytree(soft_root, other)
    m = json.loads((other/'manifest.json').read_text())
    rec = next(s for s in m['samples'] if s['params']['kind'] == 'physical_forward')
    p = other/rec['params']['reference_phase']
    a = np.asarray(Image.open(p)).copy()
    a[:32,:32] = 0
    Image.fromarray(a).save(p)
    report = inspect_soft(other/'manifest.json')
    assert report['status'] == 'failed'
    assert any('Phase/target mismatch' in e for e in report['errors'])


def test_refuses_overwrite(soft_root):
    with pytest.raises(FileExistsError):
        generate(soft_root)
