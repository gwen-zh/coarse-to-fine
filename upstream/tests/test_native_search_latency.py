import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from report_native_search_latency import main

class LatencyReportTests(unittest.TestCase):
    def prepare(self,root,mismatch=False):
        (root/'generator_timing.json').write_text(json.dumps(dict(host='nodeA',load_seconds=7.,records=[dict(target=str(i),generation_seconds=4.) for i in range(8)])))
        for i in range(8):
            (root/('timing_'+str(i)+'.json')).write_text(json.dumps(dict(target=str(i),host='nodeB' if mismatch and i==0 else 'nodeA',baseline_decode_seconds=1.,search_seconds=10.,full_worker_seconds_including_load=20.)))
    def test_warm_overhead_includes_baseline_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.prepare(root)
            with patch.object(sys,'argv',['report','--run',folder]),contextlib.redirect_stdout(io.StringIO()):main()
            d=json.loads((root/'comparison.json').read_text())
            self.assertEqual(d['mean_original_generation_plus_decode_seconds'],5.)
            self.assertEqual(d['added_warm_inference_percent'],200.)
            self.assertEqual(d['total_warm_inference_ratio'],3.)
    def test_cross_node_not_silently_pooled(self):
        with tempfile.TemporaryDirectory() as folder:
            self.prepare(Path(folder),True)
            with patch.object(sys,'argv',['report','--run',folder]),self.assertRaises(AssertionError):main()

if __name__=='__main__':unittest.main()
