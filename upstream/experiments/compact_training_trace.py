"""Append-only training audit log; materialize JSON only for resume/final report."""
import json
from pathlib import Path
from diagnostic_io import atomic_json


def restore_trace_prefix(directory,checkpoint_updates):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);log=directory/'trace.jsonl';rows=[]
    if log.exists():
        lines=log.read_text().splitlines()
        for index,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:break
                raise RuntimeError('Corrupt interior training trace row')
            if row.get('update')!=len(rows)+1:raise RuntimeError('Nonsequential training trace')
            rows.append(row)
    if len(rows)<checkpoint_updates:raise RuntimeError('Trace is shorter than the saved checkpoint')
    kept=rows[:checkpoint_updates];log.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in kept))
    (directory/'trace.json').write_text(json.dumps(kept,separators=(',',':')))
    return len(kept)


class CompactTrainingTrace:
    def __init__(self,total_updates,already_written=0):self.total=total_updates;self.written=already_written
    def __call__(self,path,value):
        path=Path(path)
        if path.name!='trace.json':return atomic_json(path,value)
        if len(value)<self.written:raise RuntimeError('Training trace moved backwards')
        new=value[self.written:]
        for index,row in enumerate(new,self.written+1):
            if row.get('update')!=index:raise RuntimeError('Nonsequential new trace data')
        with path.with_suffix('.jsonl').open('a') as stream:
            for row in new:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        self.written=len(value)
        if self.written==self.total:
            temporary=path.with_suffix('.tmp.json');temporary.write_text(json.dumps(value,separators=(',',':')));temporary.replace(path)
