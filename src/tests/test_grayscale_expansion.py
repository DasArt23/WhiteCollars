"""Новые полутона не должны менять исходные данные или попадать в train из stress."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from dataset.expand_grayscale import expand, verify
from dataset.generate_dataset import generate
from model.data_connector import ManifestQuartetDataset


def test_grayscale_expansion_preserves_source_and_frozen_splits(tmp_path):
    source = tmp_path / 'source'
    generate(source, 2, 1, 1, 3701)
    original = json.loads((source / 'manifest.json').read_text())
    original_hash = hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()
    out = tmp_path / 'gray'
    report = expand(source / 'manifest.json', out, extra_train=8, stress_count=4)
    assert report['status'] == 'passed'
    assert report['counts'] == {'diagnostic': 1, 'train': 10, 'validation': 1, 'test': 1, 'stress_validation': 4}
    assert report['new_intermediate_pixel_share_mean'] > .3
    assert report['new_unique_gray_levels_median'] > 80
    expanded = json.loads((out / 'manifest.json').read_text())
    assert expanded['samples'][:len(original['samples'])] == original['samples']
    assert hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest() == original_hash
    for record in original['samples']:
        for rel in record['images']:
            assert (source / rel).read_bytes() == (out / rel).read_bytes()
    train = ManifestQuartetDataset(out / 'manifest.json', split='train')
    stress = ManifestQuartetDataset(out / 'stress/manifest.json', split='validation')
    assert len(train) == 10 and len(stress) == 4
    assert {r.sample_id for r in train.samples}.isdisjoint(r.sample_id for r in stress.samples)
    assert np.isfinite(train[-1][0].numpy()).all()
    repeated = tmp_path / 'repeated'
    expand(source / 'manifest.json', repeated, extra_train=8, stress_count=4)
    assert (out / 'train/gray000001/01.png').read_bytes() == (repeated / 'train/gray000001/01.png').read_bytes()
    frozen = out / original['samples'][0]['images'][0]
    with Image.open(frozen) as image:
        image.putpixel((0, 0), 1)
        image.save(frozen)
    with pytest.raises(ValueError, match='Изменён исходный PNG'):
        verify(out / 'manifest.json')
