import argparse,json
from pathlib import Path
import numpy as np
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'protocol.json').read_text());rows=[json.loads((a.run/t['target']/'summary.json').read_text()) for t in protocol['targets']];assert len(rows)==24;result={}
    for arm in protocol['arms']:
        rmsds=[r['arms'][arm]['metrics']['closest_available_reference_ca_rmsd'] for r in rows];assert all(v is not None for v in rmsds)
        result[arm]=dict(reference_ca_rmsd=float(np.mean(rmsds)),paired_reference_error_minus_original=interval([v-r['arms']['original']['metrics']['closest_available_reference_ca_rmsd'] for v,r in zip(rmsds,rows)]),mean_cn_rate=float(np.mean([r['arms'][arm]['metrics']['backbone_cn_outliers']/(r['length']-1) for r in rows])),mean_compression_ratio=float(np.mean([r['arms'][arm]['compression_ratio'] for r in rows])),mean_segments=float(np.mean([r['arms'][arm]['segments'] for r in rows])),quality_errors=sum(r['arms'][arm]['quality_error'] is not None for r in rows))
    atomic_json(a.run/'comparison.json',dict(status='complete',scope=protocol['scope'],methods=result,records=rows,limitations=protocol['notes']));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
