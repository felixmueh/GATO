import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).with_name('collect.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def attempt(self, root, experiment='F01ab', status='completed', metrics=None):
        p = root / 'jetson' / experiment / 'case' / 'attempt-001'
        p.mkdir(parents=True)
        C.write(p / 'receipt.json', dict(machine='jetson', experiment=experiment, status=status,
                                      case=dict(id='n32', parameters={'N':32}, repeat=0)))
        C.write(p / 'metrics.json', metrics or {'timing_mode':'A','execution_backend':'internal','task_success':False})
        (p / 'raw.bin').write_bytes(b'\x00\x01retained')
        return p

    def test_failed_metrics_and_corrupt_neighbor_remain_visible(self):
        p = self.attempt(self.root, status='failed', metrics={'execution_error':'partial','samples':3})
        bad = self.attempt(self.root, experiment='broken')
        (bad / 'metrics.json').write_text('{"latency": NaN}')
        rows = C.collect(self.root)
        self.assertEqual(len(rows),2)
        self.assertEqual(next(r for r in rows if r['status']=='failed')['metrics']['samples'],3)
        self.assertEqual(sum(r['status']=='unreadable' for r in rows),1)
        self.assertEqual((p/'raw.bin').read_bytes(), b'\x00\x01retained')

    def test_portable_import_preserves_bytes_and_export_excludes_cache(self):
        source = self.root/'extracted'
        p = self.attempt(source)
        before = (p/'receipt.json').read_bytes()
        output = self.root/'collection'
        dest = C.import_results(source,output,'jetson-f01ab')
        self.assertEqual((dest/p.relative_to(source)/'receipt.json').read_bytes(),before)
        self.assertTrue(C.records(output)[0]['imported'])
        with self.assertRaises(ValueError): C.import_results(source,output,'jetson-f01ab')
        archive = C.export_results(output,self.root/'results.tar.gz')
        with tarfile.open(archive) as package:
            self.assertTrue(any(n.endswith('raw.bin') for n in package.getnames()))
        (source/'build-n32').mkdir()
        with self.assertRaises(ValueError): C.import_results(source,output,'wrong-workspace')
        self.assertFalse((output/'_imports/wrong-workspace').exists())

    def test_registration_does_not_claim_unrelated_experiments(self):
        source = self.root/'extracted'
        self.attempt(source)
        self.attempt(source,experiment='unrelated')
        plan=self.root/'plan.json'
        C.write(plan,dict(experiments=[dict(id='F01ab',source_commit='a'*40,protocol='v1',timing_mode='A',backend='internal',
            cases=[dict(id='n32',parameters={'N':32})])]))
        output=self.root/'results'
        C.import_results(source,output,'legacy',plan,'F01ab')
        registrations=C.read(output/'imports.json')
        self.assertEqual(len(registrations),1)
        self.assertEqual(registrations[0]['experiment'],'F01ab')
        self.assertIn('/F01ab/',registrations[0]['path'])
        self.assertEqual(len(C.records(output)),2)

    def test_export_keeps_preflight_evidence_without_pytest_cache(self):
        output = self.root / 'results'
        attempt = self.attempt(output, status='failed')
        suite = attempt / 'task_suite'
        cache = suite / 'pytest-cache/v/cache'
        cache.mkdir(parents=True)
        (cache / 'nodeids').write_text('["retained-on-machine"]')
        (suite / 'junit.xml').write_text('<testsuite/>')
        archive = C.export_results(output, self.root / 'results.tar.gz')
        with tarfile.open(archive) as package:
            names = package.getnames()
            self.assertTrue(any(n.endswith('junit.xml') for n in names))
            self.assertTrue(any(n.endswith('receipt.json') for n in names))
            self.assertFalse(any('pytest-cache' in n for n in names))
        self.assertTrue((cache / 'nodeids').exists())

    def test_registration_failure_keeps_import_indexed(self):
        source=self.root/'extracted'
        self.attempt(source,metrics={'timing_mode':'B','execution_backend':'internal'})
        plan=self.root/'plan.json'
        C.write(plan,dict(experiments=[dict(id='F01ab',source_commit='a'*40,protocol='v1',timing_mode='A',
            cases=[dict(id='n32',parameters={'N':32})])]))
        output=self.root/'results'
        with self.assertRaises(ValueError): C.import_results(source,output,'legacy',plan,'F01ab')
        self.assertEqual(len(C.read(output/'index.json')),1)
        self.assertFalse((output/'imports.json').exists())


if __name__ == '__main__': unittest.main()
