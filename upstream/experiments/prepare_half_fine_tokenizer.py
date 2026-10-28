import argparse,hashlib,json
from pathlib import Path
from diagnostic_io import atomic_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if (a.run/'protocol.json').exists():raise RuntimeError('Protocol already exists')
    a.run.mkdir(parents=True,exist_ok=True)
    parent=Path('artifacts/development/multiscale_tokenizer_v1/protocol.json');old=json.loads(parent.read_text())
    code=['half_fine_tokenizer.py','multiscale_tokenizer.py','train_half_fine_tokenizer.py','train_multiscale_tokenizer.py','report_half_fine_tokenizer.py','report_multiscale_tokenizer.py','evaluate_residual_vq.py','structural_metrics.py','backbone_quality_audit.py']
    protocol=dict(version=1,parent=str(parent),parent_sha256=sha(parent),targets=old['targets'],weights=old['weights'],codes=old['codes'],
      seed=20261026,arms=['none','local','global'],train_proteins=64,validation_proteins=24,epochs=10,lr=2e-4,geometry_weight=.01,
      scope='One-seed bounded half-fine reconstruction pilot on64training and24reuseddevelopment proteins; true-structure encoder. Not a sequence generator, fresh test, ensemble result or direct successor benchmark for the full-fine experiment. Never reads58/64confirmation cohorts.',
      transmission='Keep zero-based even fine positions from training onward. Hidden fine vectors zeroed before normalization and hidden decoder embeddings replaced with fixed original-fine-table mean. Coarse encoder may see complete true structure/fullfine, but fusion receives only hard quantized32-class coarse values and retainedfine/positions. No coordinates, encoder hidden states, missingfineIDs or missingfineembeddings enter fusion.',
      rates='Original12bits/residue; half-fine-only12*ceil(N/2)/N; half-fine+coarse5+12*ceil(N/2)/N, exactly11 for evenN and11+6/N for oddN. Shared deterministic mask requires no per-protein position list. Nominal code budget only, not serialized entropy/compression claim; shared model storage excluded.',
      architecture='Reuse preceding2x128geometrybranch, fullchain versus local16NN+sequence4 twohop. Shared trainable2x128fourheadTransformer imputer uses visible fine vectors, coarse32Dvector, visibility and fixedsinusoidalposition. It outputs decoderembedding residuals: hidden4std*tanh, visible.1std*tanh, lastlayerzero. Same model initialization/seed/dataorder forlocal/global. No-coarse has identical fusion but inactive frozen encoder, so fewer effective trainable parameters; local-global is capacity/rate-matched comparison.',
      objective='Mean standardized MSE to true hidden fine decoder embeddings +0.01*(CAaligned fullbackbone squared error+CNbond squared error)+.25VQcommitment. Hidden groundtruth only used as supervised loss target/encoder input, not fusion input. Exact original decoder frozen and differentiated via checkpointing.',
      quantizer=old['quantizer'],selection='Fixed final epoch10. No best-epoch selection, no per-sample reference-based fallback.',
      evaluation=['original_fullfine','half_untrained_mean_embedding','trained_half_no_coarse','trained_half_local_coarse','trained_half_global_coarse','zero_coarse_and_withinprotein_shuffledcoarse_forbothcoarsearms'],
      promotion_gate='Exploratory gate for considering3seedreplication: global meanCA_RMSD at least5percent below local and10percent below trainedno-coarse;longrange_distanceMAE no worse than local;CNrate at most0.5pp worse than local;8+codesused;global-minus-shuffledCA_RMSD paired95percentupperbound<0. No requirement to beat complete originalfine at a differentrate. No automatic generator training or independent test.',
      fixed_budget='2smoke optimizersteps/arm, then10epochs*64proteins/arm;3arms; no automatic variants or extension. Global and local receive identical training schedule.',
      statistics='Protein-paired descriptive bootstrap10000replicates;reuseddevelopment,not confirmatory;one seed. Fullfine versus half is differentrate. Frozen decoder never trained to decode missingfine;learned imputer and half-untrained distinguish adaptation from deletion effects.',
      code_sha256={str(Path('upstream/experiments')/f):sha(Path('upstream/experiments')/f) for f in code},
      source_sha256={old['weights']:sha(old['weights']),old['codes']:sha(old['codes'])})
    atomic_json(a.run/'protocol.json',protocol);print('HALF_FINE_PROTOCOL_READY',flush=True)

if __name__=='__main__':main()
