"""Portable release integrity checks, independent of training-platform metadata."""
import hashlib
import re
from pathlib import Path, PurePosixPath


def verify_release(root: Path) -> dict:
    """Check every declared release file; missing manifests and files fail closed.

    This checks accidental corruption against the supplied manifest. It is not
    a cryptographic signature or a substitute for checking a trusted Git commit.
    """
    root = root.resolve()
    manifest = root / 'SHA256SUMS'
    if not manifest.is_file():
        return {'files_checked': 0, 'checksum_failures': ['SHA256SUMS: missing release manifest']}
    issues = []
    seen = set()
    checked = 0
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if match is None:
            issues.append(f'SHA256SUMS:{number}: malformed entry')
            continue
        expected, name = match.groups()
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or name == 'SHA256SUMS' or name in seen
                or not path.resolve().is_relative_to(root)):
            issues.append(f'SHA256SUMS:{number}: unsafe or duplicate entry')
            continue
        seen.add(name)
        if not path.is_file() or path.is_symlink():
            issues.append(f'{name}: missing or unsupported file')
            continue
        checked += 1
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            issues.append(f'{name}: checksum mismatch')
    if not seen:
        issues.append('SHA256SUMS: no release files declared')
    return {'files_checked': checked, 'checksum_failures': issues}
