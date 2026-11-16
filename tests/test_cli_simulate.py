"""Actual one-shot CLI contracts; no training, hardware or study regeneration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OneShotCommandTests(unittest.TestCase):
    def invoke(self, folder, *args, code=0, without_site=False):
        cache = folder / 'cache'
        for name in ('mpl', 'torch', 'xdg', 'tmp'):
            (cache / name).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR=str(cache / 'mpl'), TORCH_HOME=str(cache / 'torch'),
                   XDG_CACHE_HOME=str(cache / 'xdg'), TMPDIR=str(cache / 'tmp'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        command = [sys.executable, '-B']
        if without_site:
            command.append('-S')
        result = subprocess.run(command + ['-m', 'qubitguard.cli', *map(str, args)],
                                cwd=folder, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, code, result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        return result

    def test_saved_shot_repeats_scientific_fields_without_timing_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = ['simulate', '--root', ROOT, '--distance', '3', '--noise', '.007', '--seed', '75']
            self.invoke(folder, *args, '--out', folder / 'shot.json')
            saved = json.loads((folder / 'shot.json').read_text())
            repeated = json.loads(self.invoke(folder, *args).stdout)
            self.assertEqual(saved['schema'], 'qubitguard.shot/v1')
            self.assertTrue(saved['software_version'])
            for record in (saved, repeated):
                self.assertTrue(record['shot']['certificate']['standard'])
                self.assertTrue(record['shot']['certificate']['correlated'])
                record['shot'].pop('latency_us')
            self.assertEqual(saved, repeated)

    def test_existing_output_is_preserved_before_source_or_scientific_import(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / 'shot.json'
            target.write_bytes(b'original owned record')
            self.invoke(folder, 'simulate', '--root', folder / 'missing', '--out', target,
                        code=2, without_site=True)
            self.assertEqual(target.read_bytes(), b'original owned record')

    def test_invalid_noise_is_refused_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / 'invalid.json'
            self.invoke(folder, 'simulate', '--root', ROOT, '--noise', 'nan', '--out', target, code=2)
            self.assertFalse(target.exists())

    def test_training_only_variant_cannot_silently_change_a_shot_request(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.invoke(folder, 'simulate', '--root', ROOT, '--variant', 'direct',
                        code=2, without_site=True)
            self.invoke(folder, 'simulate', '--root', ROOT, '--config', 'other.yaml',
                        code=2, without_site=True)

    def test_incomplete_serve_root_has_a_concise_remedy_without_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = self.invoke(folder, 'serve', '--root', folder, code=2, without_site=True)
            self.assertIn('--root', result.stderr)
            self.assertIn('full source', result.stderr)


if __name__ == '__main__':
    unittest.main()
