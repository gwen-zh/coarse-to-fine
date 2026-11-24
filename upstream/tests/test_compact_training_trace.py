from pathlib import Path
import sys,tempfile,unittest,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from compact_training_trace import CompactTrainingTrace,restore_trace_prefix


class CompactTraceTests(unittest.TestCase):
    def test_append_without_duplicates_and_final_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.json';writer=CompactTrainingTrace(5);rows=[dict(update=i) for i in range(1,6)]
            writer(p,rows[:2]);writer(p,rows[:2]);writer(p,rows[:4]);writer(p,rows)
            self.assertEqual(json.loads(p.read_text()),rows)
            self.assertEqual([json.loads(x) for x in p.with_suffix('.jsonl').read_text().splitlines()],rows)
    def test_restore_discards_uncommitted_tail_and_handles_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.jsonl';p.write_text(''.join(json.dumps(dict(update=i))+'\n' for i in range(1,8))+'{"update":')
            self.assertEqual(restore_trace_prefix(tmp,5),5)
            writer=CompactTrainingTrace(7,5);rows=[dict(update=i) for i in range(1,8)];writer(Path(tmp)/'trace.json',rows)
            self.assertEqual([json.loads(x) for x in p.read_text().splitlines()],rows)
    def test_missing_committed_trace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):restore_trace_prefix(tmp,10)


if __name__=='__main__':unittest.main()
