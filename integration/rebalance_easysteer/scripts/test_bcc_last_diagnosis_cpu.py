"""Synthetic protocol tests. Does not test torch hooks or the installed Qwen2 model."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from bcc_capture_store import CaptureStore
import snapshot_bcc_dependencies as snapshot


class CaptureTests(unittest.TestCase):
    def test_absolute_positions_tuple_and_copy(self):
        store = CaptureStore(); store.begin('B-S')
        raw = np.arange(600*3).reshape(1,600,3).astype(float) + 20*10000
        expected = raw[0,[486,564]].copy()
        store.put('B-S', 'layer20', (raw, None), [486,564])
        raw[:] = -1
        result = store.finish('B-S', ['layer20'])
        np.testing.assert_array_equal(result['layer20'], expected)

    def test_stale_duplicate_missing_and_nonfinite(self):
        store = CaptureStore(); store.begin('B-S')
        with self.assertRaises(ValueError): store.put('old','x',np.ones(2))
        with self.assertRaises(ValueError): store.finish('B-S',['x'])
        with self.assertRaises(ValueError): store.put('B-S','bad',np.array([np.nan]))
        store.put('B-S','x',np.ones(2))
        with self.assertRaises(ValueError): store.put('B-S','x',np.ones(2))
        store.finish('B-S',['x'])
        with self.assertRaises(ValueError): store.begin('B-S')
        store.begin('B-L')
        with self.assertRaises(ValueError): store.finish('B-L',['x'])
        with self.assertRaises(ValueError): store.put('B-L','x',np.ones((1,565,3)),[597])


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_hashes_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/'model'; model.mkdir()
            (model/'config.json').write_bytes(b'{"synthetic":true}')
            for names in snapshot.FILES.values():
                for name in names:
                    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'# synthetic source\n')
            class FakeDistribution:
                version = 'synthetic'
                def locate_file(self, name): return root/name
            resolver = lambda package: FakeDistribution()
            # Test export mechanics with a labelled fake config, not model identity.
            with patch.object(snapshot, 'MODEL_CONFIG_SHA256', snapshot.digest((model/'config.json').read_bytes())):
                result = snapshot.snapshot(root/'out', model, resolver)
                with tarfile.open(result['archive']) as tar:
                    manifest = json.load(tar.extractfile('manifest.json'))
                    self.assertEqual(len(manifest['files']),12)
                    for entry in manifest['files']:
                        self.assertEqual(snapshot.digest(tar.extractfile(entry['path']).read()),entry['sha256'])
                with self.assertRaises(ValueError): snapshot.snapshot(root/'out',model,resolver)
            with self.assertRaisesRegex(ValueError,'config differs'):
                snapshot.snapshot(root/'badconfig',model,resolver)
            huge = root/snapshot.FILES['transformers'][0]
            with huge.open('wb') as out: out.truncate(snapshot.MAX_BYTES+1)
            with self.assertRaisesRegex(ValueError,'8 MiB'):
                snapshot.snapshot(root/'oversized',model,resolver)
            self.assertFalse((root/'oversized').exists())


if __name__ == '__main__':
    unittest.main()
