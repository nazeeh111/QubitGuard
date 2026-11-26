"""Actual one-shot CLI contracts; no training, hardware or study regeneration."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class OneShotCommandTests(unittest.TestCase):
    def invoke(self, folder, *args, code=0, without_site=False, file_size_limit=None):
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
        def limit_file_size():
            import resource
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))

        result = subprocess.run(command + ['-m', 'qubitguard.cli', *map(str, args)],
                                cwd=folder, env=env, capture_output=True, text=True, timeout=60,
                                preexec_fn=limit_file_size if file_size_limit is not None else None)
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

    @unittest.skipUnless(os.name == 'posix', 'requires a POSIX file size limit')
    def test_failed_write_leaves_path_available_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / 'shot.json'
            args = ['simulate', '--root', ROOT, '--seed', '75', '--out', target]
            self.invoke(folder, *args, code=2, file_size_limit=1024)
            self.assertFalse(target.exists())
            self.assertEqual(list(folder.glob('.qubitguard-*')), [])
            self.invoke(folder, *args)
            self.assertEqual(json.loads(target.read_text())['schema'], 'qubitguard.shot/v1')

    def test_completed_shot_is_kept_when_cleanup_fails(self):
        from qubitguard.cli import _save_shot

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'shot.json'
            stderr = io.StringIO()
            with (patch.object(tempfile.TemporaryDirectory, 'cleanup',
                               side_effect=OSError('controlled cleanup failure')),
                  redirect_stderr(stderr)):
                _save_shot(target, '{"schema":"qubitguard.shot/v1"}\n')
            self.assertEqual(json.loads(target.read_text())['schema'], 'qubitguard.shot/v1')
            self.assertIn('shot saved', stderr.getvalue())
            self.assertIn('could not clean staging directory', stderr.getvalue())

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
