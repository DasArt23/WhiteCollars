"""Audit regressions: corruption, exact leakage and frozen evaluation data."""
import json
from pathlib import Path
import shutil

from PIL import Image

from dataset.audit import inspect
from dataset.generate_dataset import generate


def fixture(tmp_path):
    root = tmp_path / 'data'
    generate(root, 3, 1, 1, 3701)
    return root


def test_valid_data_and_same_seed(tmp_path):
    root = fixture(tmp_path)
    other = tmp_path/'repeat'
    generate(other, 3, 1, 1, 3701)
    report = inspect(other/'manifest.json', root/'manifest.json')
    assert report['errors'] == []
    assert report['checked_images'] == 24
    assert report['frozen_reference']['checked_images'] == 24


def test_exact_cross_split_duplicate_fails(tmp_path):
    root = fixture(tmp_path)
    shutil.copyfile(root/'train/000001/01.png', root/'test/000001/01.png')
    report = inspect(root/'manifest.json')
    assert report['status'] == 'failed'
    assert any('Exact duplicate' in e for e in report['errors'])


def test_missing_file_fails(tmp_path):
    root = fixture(tmp_path)
    (root/'train/000001/01.png').unlink()
    assert inspect(root/'manifest.json')['status'] == 'failed'


def test_rgb_rejected(tmp_path):
    root = fixture(tmp_path)
    path = root/'train/000001/01.png'
    with Image.open(path) as im:
        im.convert('RGB').save(path)
    assert inspect(root/'manifest.json')['status'] == 'failed'


def test_frozen_pixels_checked_without_trusting_report(tmp_path):
    root = fixture(tmp_path)
    other = tmp_path/'copy'
    shutil.copytree(root, other)
    path = other/'validation/000001/01.png'
    with Image.open(path) as im:
        im.putpixel((64, 64), 128 if im.getpixel((64, 64)) != 128 else 127)
        im.save(path)
    report = inspect(other/'manifest.json', root/'manifest.json')
    assert report['status'] == 'failed'
    assert 'validation/000001/01.png' in report['frozen_reference']['mismatches']


def test_wrong_plane_order_fails(tmp_path):
    root = fixture(tmp_path)
    path = root/'manifest.json'
    m = json.loads(path.read_text())
    m['samples'][1]['images'].reverse()
    path.write_text(json.dumps(m))
    assert inspect(path)['status'] == 'failed'


def test_near_duplicate_is_reported_for_review(tmp_path):
    root = fixture(tmp_path)
    with Image.open(root/'train/000001/01.png') as im:
        im.putpixel((64, 64), 128 if im.getpixel((64, 64)) != 128 else 127)
        im.save(root/'test/000001/01.png')
    report = inspect(root/'manifest.json')
    assert report['status'] == 'review_required'
    assert report['near_similarity']['test_train']['near_images_found'] >= 1
