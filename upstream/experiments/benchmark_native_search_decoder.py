"""Instrument actual30s search on the same allocated node as generation."""
import argparse,json,socket,time,sys
from pathlib import Path
import bounded_fine_search_ensemble as worker
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--index",type=int,required=True);a=p.parse_args()
    times=[];original=worker.decode_embedding
    def measured(*args,**kwargs):
        started=time.perf_counter();value=original(*args,**kwargs);times.append(time.perf_counter()-started);return value
    worker.decode_embedding=measured
    started=time.perf_counter();worker.main();total=time.perf_counter()-started
    protocol=json.loads((a.run/"protocol.json").read_text());item=json.loads((Path(protocol["source"])/"manifest.json").read_text())["targets"][a.index]
    d=json.loads((a.run/item["target"]/"summary.json").read_text())["arms"]["discrete_topk"]
    atomic_json(a.run/("timing_"+str(a.index)+".json"),dict(target=item["target"],host=socket.gethostname(),threads=16,baseline_decode_seconds=times[0],candidate_decode_seconds=sum(times[1:]),search_seconds=d["search_seconds"],full_worker_seconds_including_load=total,decoder_calls=len(times)-1,scope="Same-node CPU16. Baseline first decode measured without decoder warmup; search includes proposals, decoding and gate. Whole worker includes decoder construction/load; import/startup excluded."))


if __name__=="__main__":main()
