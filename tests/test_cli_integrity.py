"""Release checks must report corruption without loading experiment settings."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src'


class VerifyCommandTests(unittest.TestCase):
    def run_verify(self, root, expected_exit, *extra, without_site=False):
        command = [sys.executable]
        if without_site:
            command.append('-S')
        command += ['-m', 'qubitguard.cli', 'verify', '--root', str(root), *extra]
        env = dict(os.environ, PYTHONPATH=str(SOURCE))
        result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, expected_exit, result.stderr)
        self.assertTrue(result.stdout.strip(), 'Missing JSON report: ' + result.stderr)
        report = json.loads(result.stdout)
        self.assertNotIn('Traceback', result.stderr)
        return report

    def declare(self, root, files):
        rows = []
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rows.append(f'{hashlib.sha256(content).hexdigest()}  {name}\n')
        (root / 'SHA256SUMS').write_text(''.join(rows))

    def test_valid_declared_files_do_not_need_experiment_configuration(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'result.bin': b'recorded fixture'})
            self.assertEqual(self.run_verify(root, 0),
                             {'files_checked': 1, 'checksum_failures': []})

    def test_missing_manifest_is_reported_without_experiment_configuration(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            report = self.run_verify(Path(directory), 1)
            self.assertEqual(report['files_checked'], 0)
            self.assertEqual(report['checksum_failures'],
                             ['SHA256SUMS: missing release manifest'])

    def test_missing_declared_configuration_is_an_integrity_failure(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'configs/research-v1.yaml': b'basis: x\n'})
            (root / 'configs/research-v1.yaml').unlink()
            report = self.run_verify(root, 1)
            self.assertEqual(report['files_checked'], 0)
            self.assertEqual(report['checksum_failures'],
                             ['configs/research-v1.yaml: missing or unsupported file'])

    def test_corrupt_configuration_is_hashed_instead_of_parsed(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'configs/research-v1.yaml': b'basis: x\n'})
            (root / 'configs/research-v1.yaml').write_bytes(b'basis: [\n')
            report = self.run_verify(root, 1)
            self.assertEqual(report['files_checked'], 1)
            self.assertEqual(report['checksum_failures'],
                             ['configs/research-v1.yaml: checksum mismatch'])

    def test_unrelated_config_override_does_not_block_checksum_verification(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'result.bin': b'recorded fixture'})
            report = self.run_verify(root, 0, '--config', 'missing-experiment.yaml')
            self.assertEqual(report, {'files_checked': 1, 'checksum_failures': []})

    def test_all_declared_files_are_checked_when_one_is_corrupt(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'first.bin': b'first fixture', 'second.bin': b'second fixture'})
            (root / 'second.bin').write_bytes(b'changed fixture')
            report = self.run_verify(root, 1)
            self.assertEqual(report['files_checked'], 2)
            self.assertEqual(report['checksum_failures'], ['second.bin: checksum mismatch'])

    def test_verify_runs_without_third_party_scientific_packages(self):
        with tempfile.TemporaryDirectory(prefix='qg-cli-integrity-') as directory:
            root = Path(directory)
            self.declare(root, {'result.bin': b'recorded fixture'})
            report = self.run_verify(root, 0, without_site=True)
            self.assertEqual(report, {'files_checked': 1, 'checksum_failures': []})


if __name__ == '__main__':
    unittest.main()
