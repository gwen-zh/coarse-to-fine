"""Reuse native coarse denoiser, native noise clock and original fine argmax."""
import torch
from discrete_flow_math import native_raw_weights,transition_probabilities,competing_exponential_sample


@torch.no_grad()
def mask_flow_sample(model,sequence,lengths,steps,beta=0.,eps=1e-5):
    from slm.models.model import add_eos_bos_tokens
    if not model.noise_removal or model.coarse_mask_index!=32:raise ValueError("Native32-token/noise-removal path required")
    model.net.eval();model.noise.eval();model.sigma_embedder.eval()
    x=model._sample_prior(*sequence.shape).to(sequence.device)
    times=torch.linspace(1.,eps,steps+1,device=sequence.device);dt=(1-eps)/steps
    x=add_eos_bos_tokens(lengths,34,33,35,x[:,1:-1]);last_visible=torch.full_like(x,-1);trace=[]
    for step in range(steps):
        t=times[step]*torch.ones(x.shape[0],1,device=sequence.device)
        sigma_t=model.noise(t)[0];sigma_s=model.noise(t-dt)[0]
        if sigma_t.ndim>1:sigma_t=sigma_t.squeeze(-1)
        if sigma_s.ndim>1:sigma_s=sigma_s.squeeze(-1)
        mt=1-torch.exp(-sigma_t);ms=1-torch.exp(-sigma_s)
        log_p,_=model._model_wrapper_inference(x,sequence,sigma_t,lengths,scale=0)
        if beta==0:
            proposal=competing_exponential_sample(native_raw_weights(log_p,mt,ms));proposal[proposal>=32]=32
            copy=(x!=32).to(x.dtype);new=copy*x+(1-copy)*proposal
        else:
            probability=transition_probabilities(log_p,x,mt,ms,beta)
            if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(-1),torch.ones_like(x,dtype=probability.dtype),atol=2e-6):raise RuntimeError("Invalid corrector transition")
            new=competing_exponential_sample(probability)
        # Source and destination special tokens are never changed by the corrector.
        new=add_eos_bos_tokens(lengths,34,33,35,new[:,1:-1])
        before=x[:,1:-1];after=new[:,1:-1];previous=last_visible[:,1:-1]
        revised=(before==32)&(after<32)&(previous>=0)&(after!=previous)
        trace.append(dict(step=step,masked_before=int((before==32).sum().item()),masked_after=int((after==32).sum().item()),unmask_events=int(((before==32)&(after<32)).sum().item()),remask_events=int(((before<32)&(after==32)).sum().item()),revised_token_events=int(revised.sum().item())))
        last_visible=torch.where(new<32,new,last_visible);x=new
    sigma=model.noise(times[-1]*torch.ones(x.shape[0],1,device=sequence.device))[0]
    logits,_=model._model_wrapper_inference(x,sequence_tokens=sequence,sigma=sigma,lengths=lengths,scale=0)
    coarse=add_eos_bos_tokens(lengths,34,33,35,logits.argmax(-1)[:,1:-1])
    sigma=model.noise(torch.zeros(x.shape[0],1,device=sequence.device))[0]
    logits,_=model._model_wrapper_inference(coarse,sequence_tokens=sequence,sigma=sigma,lengths=lengths,scale=2)
    fine=add_eos_bos_tokens(lengths,4098,4097,4099,logits.argmax(-1)[:,1:-1])
    return fine,coarse,dict(trace=trace,remask_events=sum(x["remask_events"] for x in trace),revised_token_events=sum(x["revised_token_events"] for x in trace),steps=steps,beta=beta)
