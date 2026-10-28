"""Small-set coarse learnability: existing flow vs a deterministic point baseline.

Both begin at the same trained coarse checkpoint. Regression reuses the same
velocity network at zero state/time, but predicts normalized native latent.
It is a diagnostic point predictor, not an ensemble generator.
"""
import hashlib
import torch
from mixed_hierarchical_generator import pooled_sequence
from latent_flow_head import flow_path,protein_mean_mse

ARMS=['flow','regression']


def point_prediction(model,features):
    condition=pooled_sequence(features)[None]
    state=condition.new_zeros((1,len(condition[0]),len(model.mean)))
    t=condition.new_zeros((1,));valid=torch.ones(state.shape[:2],dtype=torch.bool,device=state.device)
    return model.denormalize(model.coarse_flow(state,t,condition,valid)[0])


def probe_loss(model,decoder,features,target,arm,geometry_weight=.1):
    if arm not in ARMS:raise ValueError('Unknown coarse objective')
    truth=model.normalize(target)[None];noise=torch.randn_like(truth);t=torch.rand((1,),device=truth.device)
    condition=pooled_sequence(features)[None];valid=torch.ones(truth.shape[:2],dtype=torch.bool,device=truth.device)
    if arm=='flow':
        state,velocity=flow_path(truth,noise,t)
        prediction=model.coarse_flow(state,t,condition,valid)
        latent=protein_mean_mse(prediction,velocity,valid)
        endpoint=state+(1-t[:,None,None])*prediction
    else:
        prediction=model.coarse_flow(torch.zeros_like(truth),torch.zeros_like(t),condition,valid)
        latent=protein_mean_mse(prediction,truth,valid);endpoint=prediction
    geometry=latent*0.
    # Pair the auxiliary center-loss gate in both arms; regression also draws the
    # same unused noise/time, so RNG streams stay aligned with the flow arm.
    if float(t[0])>=.5 and geometry_weight:
        centers=decoder.coarse_centers(model.denormalize(endpoint[0]),len(features))
        with torch.no_grad():reference=decoder.coarse_centers(target,len(features))
        geometry=(centers-reference).square().mean()/100.
    loss=latent+geometry_weight*geometry
    draw_hash=hashlib.sha256(noise.detach().cpu().numpy().tobytes()+t.detach().cpu().numpy().tobytes()).hexdigest()
    return loss,dict(latent=latent.detach(),geometry=geometry.detach(),t=float(t[0]),draw_sha256=draw_hash)


@torch.no_grad()
def predict(model,features,arm,seed,steps=16):
    if arm=='flow':return model.sample_coarse(features,seed,steps)
    if arm=='regression':return point_prediction(model,features)
    raise ValueError('Unknown inference arm')


def state_hash(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
