import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from decoder_angle_adaptation import angles, angular_term, enable_decoder, frozen_digest, objective, update
from fine_geometry_flow import sample_training_batch
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import save_checkpoint, restore_checkpoint
from run_decoder_angle_adaptation import batch_rng, identity
from sequence_multiscale_ensemble import MultiscaleStructureCodec


def codec():
    torch.manual_seed(93)
    return MultiscaleStructureCodec(torch.nn.Linear(3,3),StableBlockFrameFineTokenizer("local"),"centers3")


def case(model,name,offset=0.):
    g=torch.Generator().manual_seed(52)
    x=torch.randn(13,3,3,generator=g)*3
    x[:,1]+=torch.arange(13)[:,None]*torch.tensor([.3,.2,.1])
    x=x+offset;c,f=model.encode(x)
    return dict(id=name,protein=name,split="train",sequence="ACDEFGHIKLMNP",target=x,coarse=c,fine_continuous=f)


def protocol():
    return dict(updates=4,lr=1e-4,min_lr=1e-5,angle_weight=.01,angle_scale_degrees=5.,gradient_clip=1.)


class DecoderAngleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_update_changes_only_decoder_and_preserves_encoder_and_centers(self):
        m=codec();params=enable_decoder(m);before=frozen_digest(m);a=case(m,"p");c,f=m.encode(a["target"])
        old={k:v.clone() for k,v in m.fine_tokenizer.decoder.state_dict().items()}
        optimizer=torch.optim.AdamW(params,lr=1e-4)
        row=update(m,optimizer,[a,a,a,a],"angular",0,protocol())
        self.assertGreater(row["gradient_norm"],0);self.assertEqual(before,frozen_digest(m))
        self.assertTrue(any(not torch.equal(v,old[k]) for k,v in m.fine_tokenizer.decoder.state_dict().items()))
        for x,y in zip((c,f),m.encode(a["target"])):torch.testing.assert_close(x,y,atol=0,rtol=0)
        torch.testing.assert_close(pool_ca(m.decode(c,f)),c,atol=2e-5,rtol=2e-5)
        trainable=[n for n,v in m.named_parameters() if v.requires_grad]
        self.assertTrue(all(n.startswith("fine_tokenizer.decoder.") for n in trainable))

    def test_angular_term_is_zero_on_target_and_rigid_motion_invariant(self):
        m=codec();a=case(m,"p");x=a["target"].double();y=x.clone();y[4,0,1]+=.2
        self.assertEqual(float(angular_term(x,x)),0.)
        rotation=x.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        torch.testing.assert_close(angular_term(y,x),angular_term(y@rotation+17,x@rotation+17),atol=1e-10,rtol=1e-10)

    def test_direct_angular_gradient_matches_finite_difference(self):
        m=codec();a=case(m,"p");target=a["target"].double();x=target.clone();x[4,0,1]+=.2;x.requires_grad_()
        value=angular_term(x,target);gradient=torch.autograd.grad(value,x)[0]
        step=1e-5;plus=x.detach().clone();minus=x.detach().clone();plus[4,0,1]+=step;minus[4,0,1]-=step
        estimate=(angular_term(plus,target)-angular_term(minus,target))/(2*step)
        torch.testing.assert_close(gradient[4,0,1],estimate,atol=1e-7,rtol=1e-5)

    def test_matched_objectives_have_expected_difference_and_finite_gradients(self):
        m=codec();a=case(m,"p");x=a["target"].clone();x[4,0,1]+=.3;x.requires_grad_()
        first,_=objective(x,a["target"],"continuation",protocol())
        second,_=objective(x,a["target"],"angular",protocol())
        torch.testing.assert_close(second-first,.01*angular_term(x,a["target"]),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(torch.autograd.grad(second,x)[0]).all())

    def test_development_example_is_rejected(self):
        m=codec();params=enable_decoder(m);a=case(m,"p");a["split"]="validation"
        with self.assertRaisesRegex(ValueError,"development"):
            update(m,torch.optim.AdamW(params),[a]*4,"angular",0,protocol())

    def test_resume_recovers_optimizer_and_paired_training_draws(self):
        initial=codec();a=case(initial,"p");b=case(initial,"q",3.)
        groups=dict(PDB_train=[a,b],train={"m":[a,b]});p=protocol()
        full=copy.deepcopy(initial);opt=torch.optim.AdamW(enable_decoder(full),lr=p["lr"]);rng=batch_rng(2026);rows=[]
        for step in range(4):rows.append(update(full,opt,sample_training_batch(groups,rng),"angular",step,p))
        part=copy.deepcopy(initial);opt2=torch.optim.AdamW(enable_decoder(part),lr=p["lr"]);rng2=batch_rng(2026)
        for step in range(2):update(part,opt2,sample_training_batch(groups,rng2),"angular",step,p)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"last.pt";save_checkpoint(path,part,opt2,rng2,updates=2)
            restored=codec();opt3=torch.optim.AdamW(enable_decoder(restored),lr=p["lr"]);rng3=batch_rng(2027)
            restore_checkpoint(path,restored,opt3,rng3,torch.device("cpu"))
            for step in range(2,4):
                row=update(restored,opt3,sample_training_batch(groups,rng3),"angular",step,p)
                self.assertEqual(identity(row),identity(rows[step]))
            for k,v in full.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)


if __name__=="__main__":unittest.main()
