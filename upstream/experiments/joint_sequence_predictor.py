"""AA-only shared Transformer with an explicit predicted-coarse -> fine path.

Experimental deterministic predictor, separate from the current two-stage flow /
masked generator. Frozen structural encoders supply training targets only.
No ESM features, reference coordinates or native coarse are forward inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from two_level_autoencoder import pack,positions

AA_ALPHABET='ACDEFGHIKLMNPQRSTVWYX'


def encode_aa(sequence,device=None):
    if not isinstance(sequence,str) or not sequence:raise ValueError('A nonempty AA sequence is required')
    sequence=sequence.upper()
    if any(letter not in AA_ALPHABET for letter in sequence):raise ValueError('Use standard amino acids or X; gaps are not dropped')
    return torch.tensor([AA_ALPHABET.index(letter) for letter in sequence],dtype=torch.long,device=device)


class JointSequencePredictor(nn.Module):
    """One unpadded chain; fine predictions explicitly consume predicted coarse.

    coarse: [ceil(N/4),coarse_dim] raw structural latent values.
    fine_logits: [N,fine_size], over the matching frozen fine codebook IDs.
    Fine decoder vector dimensionality (currently8) need not equal coarse_dim.
    """
    def __init__(self,coarse_mean,coarse_std,width=256,layers=6,coarse_dim=16,fine_size=1024,block_size=4):
        super().__init__()
        mean=torch.as_tensor(coarse_mean,dtype=torch.float32);std=torch.as_tensor(coarse_std,dtype=torch.float32)
        if mean.shape!=(coarse_dim,) or std.shape!=(coarse_dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('Finite TRAIN-only coarse normalization required')
        if width<8 or width%8 or layers<1 or coarse_dim<1 or fine_size<2 or block_size<1:raise ValueError('Invalid predictor configuration')
        self.register_buffer('coarse_mean',mean.clone());self.register_buffer('coarse_std',std.clone())
        self.coarse_dim=coarse_dim;self.fine_size=fine_size;self.block_size=block_size
        self.config=dict(width=width,layers=layers,coarse_dim=coarse_dim,fine_size=fine_size,block_size=block_size)
        self.aa_embedding=nn.Embedding(len(AA_ALPHABET),width)
        layer=nn.TransformerEncoderLayer(width,8,4*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.coarse_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,coarse_dim),nn.Tanh())
        self.fine_head=nn.Sequential(nn.LayerNorm(width+coarse_dim),nn.Linear(width+coarse_dim,width),nn.SiLU(),nn.Linear(width,fine_size))
    def sequence_states(self,aa):
        if aa.ndim!=1 or aa.dtype!=torch.long or not len(aa) or aa.min()<0 or aa.max()>=len(AA_ALPHABET):raise ValueError('Expected one unpadded chain of valid AA integer IDs')
        h=self.aa_embedding(aa);h=h+positions(len(aa),h.device,h.dtype,h.shape[-1])
        return self.transformer(h[None])[0]
    def broadcast_coarse(self,coarse,length):
        if length<1 or coarse.shape!=((length+self.block_size-1)//self.block_size,self.coarse_dim) or not torch.isfinite(coarse).all():raise ValueError('Coarse shape/length mismatch')
        return coarse.repeat_interleave(self.block_size,dim=0)[:length]
    def fine_logits_from_states(self,states,coarse):
        if states.ndim!=2 or states.shape[1]!=self.config['width'] or not torch.isfinite(states).all():raise ValueError('Expected finite residue states')
        context=self.broadcast_coarse(coarse,len(states))
        # No detach: fine CE can update the coarse head through this condition.
        return self.fine_head(torch.cat([states,context],dim=-1))
    def forward(self,aa):
        h=self.sequence_states(aa);packed,mask=pack(h,self.block_size)
        pooled=(packed*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        coarse=self.coarse_head(pooled)
        fine_logits=self.fine_logits_from_states(h,coarse)
        return dict(coarse=coarse,fine_logits=fine_logits)
    def supervised_loss(self,aa,target_coarse,target_fine,coarse_weight=1.,fine_weight=1.):
        if coarse_weight<0 or fine_weight<0 or not coarse_weight+fine_weight:raise ValueError('Nonnegative,not both zero,loss weights required')
        n=len(aa);m=(n+self.block_size-1)//self.block_size
        if target_coarse.shape!=(m,self.coarse_dim) or not target_coarse.is_floating_point() or not torch.isfinite(target_coarse).all():raise ValueError('Invalid coarse target')
        if target_fine.shape!=(n,) or target_fine.dtype!=torch.long or target_fine.min()<0 or target_fine.max()>=self.fine_size:raise ValueError('Invalid fine target IDs')
        output=self(aa)
        coarse_loss=((output['coarse']-target_coarse.detach())/self.coarse_std).square().mean()
        fine_loss=F.cross_entropy(output['fine_logits'],target_fine)
        return coarse_weight*coarse_loss+fine_weight*fine_loss,dict(coarse_mse=coarse_loss.detach(),fine_ce=fine_loss.detach())
    @torch.no_grad()
    def predict(self,aa):
        # No noise or categorical sampling: this is a point-prediction baseline.
        output=self(aa)
        return output['coarse'],output['fine_logits'].argmax(-1)
