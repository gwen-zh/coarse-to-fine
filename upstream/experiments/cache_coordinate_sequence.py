"""Pure amino-acid-sequence features; never read structure labels or old tokens."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Hierarchical_VQ_VAE'))
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.encoding import tokenize_sequence
from diagnostic_io import atomic_json

# Old vendored ESM tokenizer expects a removed transformers __getattr__.
# This extractor uses the fixed, original ESM vocabulary only.
_SPECIAL={"cls_token":"<cls>","eos_token":"<eos>","mask_token":"<mask>","pad_token":"<pad>"}
EsmSequenceTokenizer._get_token=lambda self,name:_SPECIAL[name]


@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--chunk',type=int,required=True);a=p.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 protocol=json.loads((a.run/'protocol.json').read_text());source=Path(protocol['corpus_manifest']);assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['corpus_manifest_sha256'];rows=json.loads(source.read_text())['targets'];device=torch.device('cuda');model=ESM3.from_pretrained('esm3_sm_open_v1',device=device).eval().requires_grad_(False);tokenizer=EsmSequenceTokenizer();assert [tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.mask_token_id,tokenizer.pad_token_id]==[0,2,32,1];directory=a.run/'sequence_features';directory.mkdir(exist_ok=True);done=[];started=time.perf_counter()
 for i,row in enumerate(rows[a.chunk::protocol['cache_chunks']]):
  seq=tokenize_sequence(row['sequence'],tokenizer).to(device)[None]
  with torch.autocast('cuda',dtype=torch.bfloat16):output=model(sequence_tokens=seq).embeddings[0,1:-1].float().cpu().numpy()
  assert output.shape==(row['length'],1536) and np.isfinite(output).all()
  if i==0:
   with torch.autocast('cuda',dtype=torch.bfloat16):replay=model(sequence_tokens=seq).embeddings[0,1:-1].float().cpu().numpy()
   assert np.array_equal(output,replay),'Frozen sequence feature extraction must be deterministic'
  path=directory/(row['id']+'.npz');np.savez_compressed(path,h=output);done.append(dict(id=row['id'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  if (i+1)%20==0:print('COORDINATE_SEQUENCE_CACHE',a.chunk,i+1,flush=True)
 atomic_json(a.run/f'sequence_chunk{a.chunk}.json',dict(status='complete',targets=done,seconds=time.perf_counter()-started,features='Original pretrained ESM3 sequence-only forward, all non-sequence tracks omitted/default-masked; no generated coarse or reference coordinates supplied',dtype=str(next(model.parameters()).dtype),gpu=torch.cuda.get_device_name()))


if __name__=='__main__':main()
