"""A published release must never pass verification after files disappear."""
import hashlib
import pytest
from qubitguard.integrity import verify_release


def manifest(root, name='result.bin', payload=b'recorded scientific artifact'):
    path = root / name
    path.write_bytes(payload)
    (root / 'SHA256SUMS').write_text(f'{hashlib.sha256(payload).hexdigest()}  {name}\n')
    return path


def test_complete_release_passes(tmp_path):
    manifest(tmp_path)
    assert verify_release(tmp_path) == {'files_checked': 1, 'checksum_failures': []}


@pytest.mark.parametrize('change', ['missing_manifest', 'empty_manifest', 'missing_file', 'changed_file'])
def test_incomplete_or_corrupt_release_fails(tmp_path, change):
    path = manifest(tmp_path)
    if change == 'missing_manifest': (tmp_path / 'SHA256SUMS').unlink()
    elif change == 'empty_manifest': (tmp_path / 'SHA256SUMS').write_text('')
    elif change == 'missing_file': path.unlink()
    else: path.write_bytes(b'different')
    assert verify_release(tmp_path)['checksum_failures']


@pytest.mark.parametrize('name', ['../outside', '/etc/passwd', 'SHA256SUMS'])
def test_unsafe_manifest_entry_is_rejected(tmp_path, name):
    (tmp_path / 'SHA256SUMS').write_text('0' * 64 + '  ' + name + '\n')
    assert verify_release(tmp_path)['checksum_failures']


def test_duplicate_and_malformed_entries_are_rejected(tmp_path):
    manifest(tmp_path)
    path = tmp_path / 'SHA256SUMS'
    path.write_text(path.read_text() * 2 + 'invalid\n')
    assert len(verify_release(tmp_path)['checksum_failures']) == 2
