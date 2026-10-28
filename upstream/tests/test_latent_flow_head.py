import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from latent_flow_head import LatentHead,flow_path,protein_mean_mse,euler_sample

class FlowHeadTests(unittest.TestCase):
    def test_linear_path_velocity(self):
        z=torch.ones(2,3,4)*3;noise=torch.ones_like(z)*-1;t=torch.tensor([0.,1.]);x,v=flow_path(z,noise,t)
        torch.testing.assert_close(x[0],noise[0]);torch.testing.assert_close(x[1],z[1]);torch.testing.assert_close(v,z-noise)
    def test_padding_not_in_loss(self):
        target=torch.zeros(2,3,4);pred=torch.ones_like(target);valid=torch.tensor([[1,0,0],[1,1,0]],dtype=torch.bool);pred[~valid]=1000
        self.assertEqual(protein_mean_mse(pred,target,valid).item(),1.)
    def test_euler_constant_velocity(self):
        class Constant(torch.nn.Module):
            def forward(self,x,t,c,valid):return torch.ones_like(x)*2
        x=torch.zeros(1,3,4);out=euler_sample(Constant(),x,torch.zeros(1,3,5),torch.ones(1,3,dtype=torch.bool),16)
        torch.testing.assert_close(out,torch.ones_like(x)*2)
    def test_zero_initialized_head_and_padding(self):
        torch.set_num_threads(1);model=LatentHead(condition_dim=8,latent_dim=4,width=16,layers=1).eval();c=torch.randn(1,3,8);x=torch.randn(1,3,4);valid=torch.tensor([[1,1,0]],dtype=torch.bool)
        out=model(x,torch.zeros(1),c,valid);torch.testing.assert_close(out,torch.zeros_like(out))
        with torch.no_grad():model.output.weight.fill_(.1)
        first=model(x,torch.zeros(1),c,valid)[:,:2];c[:,2]+=1000;x[:,2]+=1000
        torch.testing.assert_close(first,model(x,torch.zeros(1),c,valid)[:,:2])

if __name__=="__main__":unittest.main()
