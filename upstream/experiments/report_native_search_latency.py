"""Component latency ratios, warm versus separately-loaded worker costs explicit."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();g=json.loads((a.run/"generator_timing.json").read_text());dec=[json.loads((a.run/("timing_"+str(i)+".json")).read_text()) for i in range(8)]
    assert all(x["host"]==g["host"] for x in dec)
    seq={x["target"]:x["generation_seconds"] for x in g["records"]}
    base=[seq[x["target"]]+x["baseline_decode_seconds"] for x in dec];search=[x["search_seconds"] for x in dec];full=np.asarray(base)+search
    meanbase=float(np.mean(base));meansearch=float(np.mean(search));result=dict(status="complete",outputs=8,proteins=4,host=g["host"],cpu_threads=16,mean_original_generation_seconds=float(np.mean(list(seq.values()))),mean_original_generation_plus_decode_seconds=meanbase,mean_added_search_seconds=meansearch,mean_refined_generation_plus_decode_search_seconds=float(np.mean(full)),added_warm_inference_percent=100*meansearch/meanbase,total_warm_inference_ratio=float(np.mean(full))/meanbase,generator_load_seconds=g["load_seconds"],mean_separate_decoder_worker_seconds=float(np.mean([x["full_worker_seconds_including_load"] for x in dec])),scope="Component sum on one CPU node, same sequences and outputs, weights fixed. Warm inference excludes model loading and process startup; generator/decoder run sequentially in separate processes due namespace incompatibility, so not a single-process co-resident deployment benchmark. Baseline first decoder forward is not pre-warmed.")
    atomic_json(a.run/"comparison.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
