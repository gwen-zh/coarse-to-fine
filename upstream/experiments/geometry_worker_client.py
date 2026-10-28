"""Keep generator and tokenizer ESM modules isolated while passing logit gradients."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


def backward_geometry_objective(base_loss,logits,geometry_gradient,weight):
    """Apply externally computed dL/dlogits without treating it as a scalar loss."""
    if geometry_gradient.shape!=logits.shape or not torch.isfinite(geometry_gradient).all():
        raise ValueError('Invalid external geometry gradient')
    torch.autograd.backward([base_loss,logits],[None,weight*geometry_gradient])


class GeometryWorker:
    def __init__(self,weights,log_path,threads=8):
        self.temporary=tempfile.TemporaryDirectory(prefix='hcg-geometry-')
        self.exchange=Path(self.temporary.name)
        self.log=open(log_path,'w')
        self.process=None
        self.request=0
        try:
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('geometry_gradient_worker.py')),
                '--weights',str(Path(weights).resolve()),'--exchange',str(self.exchange),'--threads',str(threads)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
            self._receive('ready')
        except BaseException:
            self.close();raise

    def _receive(self,status):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Geometry worker exited; inspect its worker.log')
        response=json.loads(line)
        if response.get('status')!=status:raise RuntimeError('Unexpected geometry worker response: '+line)
        return response

    def gradient(self,logits):
        self.request+=1
        torch.save(dict(logits=logits.detach().cpu()),self.exchange/'request.pt')
        self.process.stdin.write(json.dumps(dict(request=self.request))+'\n');self.process.stdin.flush()
        if self._receive('complete').get('request')!=self.request:raise RuntimeError('Stale geometry response')
        result=torch.load(self.exchange/'response.pt',map_location=logits.device,weights_only=True)
        if result['gradient'].shape!=logits.shape or not torch.isfinite(result['gradient']).all():
            raise RuntimeError('Invalid geometry gradient')
        if not torch.equal(result['fine'],logits.detach().argmax(-1)):
            raise RuntimeError('Worker hard forward differs from current generator tokens')
        return result

    def close(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write('{"command":"close"}\n');self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                self.process.kill();self.process.wait()
            finally:
                self.process.stdin.close();self.process.stdout.close()
        self.log.close();self.temporary.cleanup()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
