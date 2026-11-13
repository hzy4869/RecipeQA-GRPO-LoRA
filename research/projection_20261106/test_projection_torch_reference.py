"""Independent real-PyTorch CPU gradient/Adam reference for projection drafts.

Local absence of Torch causes explicit skips. A deployment gate MUST require
these tests to pass with zero skips on the existing remote environment. No CUDA,
model loading, SSH, new data or VLM performance experiment is performed here.
"""
from __future__ import annotations
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from projection_math import make_qstar,projection_terms

HAS_TORCH=importlib.util.find_spec('torch') is not None
BETA=1.6363096678310884
C=5
ACCUMULATION=4


@unittest.skipUnless(HAS_TORCH,'Real Torch CPU reference unavailable locally; remote gate requires zero skips')
class ProjectionTorchCPUReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        cls.cuda_was_initialized=torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        if cls.torch.cuda.is_initialized()!=cls.cuda_was_initialized:
            raise AssertionError('CPU reference initialized CUDA')

    def theta(self):
        return self.torch.tensor([.23,-.41,.17,.09],dtype=self.torch.float64,device='cpu',requires_grad=True)

    def scores(self,theta,prompt):
        t=self.torch
        action=t.arange(24,dtype=t.float64,device='cpu')
        # Actual shared nonlinear parameterization: each of four parameters
        # influences many actions, with prompt-dependent cross derivatives.
        return (theta[0]*t.sin(action+.2+prompt*.13)+theta[1]*t.cos(action*.3+prompt*.11)
                +theta[0]*theta[1]*t.sin(action*.7+prompt*.17)
                +theta[2]*(action-11)/20+theta[3]**2*t.cos(action*.21+prompt))

    def target(self,prompt,reference_theta=None):
        t=self.torch
        if reference_theta is None:
            logits=t.tensor([math.sin(i*.4+prompt*.2)-i*.03 for i in range(24)],dtype=t.float64,device='cpu')
        else:
            logits=self.scores(reference_theta,prompt).detach()
        logref=t.log_softmax(logits,dim=0).tolist()
        return make_qstar(logref,[int(i==(7+prompt*3)%24) for i in range(24)],BETA)

    def direct_loss(self,theta,prompt,target):
        t=self.torch
        logq=t.log_softmax(self.scores(theta,prompt),dim=0)
        star=t.tensor(target.q,dtype=t.float64,device='cpu')
        logstar=t.tensor(target.logq,dtype=t.float64,device='cpu')
        self.assertFalse(star.requires_grad)
        return BETA*t.sum(star*(logstar-logq))/(C*ACCUMULATION)

    def detached_loss(self,theta,prompt,target):
        t=self.torch
        scores=self.scores(theta,prompt)
        with t.no_grad():
            logq=t.log_softmax(scores,dim=0)
            terms=projection_terms({'q':logq.exp().tolist(),'logq':logq.tolist()},target)
        coefficients=t.tensor(terms['coefficients'],dtype=t.float64,device='cpu')
        self.assertFalse(coefficients.requires_grad)
        return -t.sum(coefficients*scores)/(C*ACCUMULATION)

    def test_actual_autograd_matches_detached_coefficients_shared_batch(self):
        t=self.torch
        a,b=self.theta(),self.theta()
        targets=[self.target(i) for i in range(4)]
        before=[asdict(x) for x in targets]
        direct=sum(self.direct_loss(a,i,goal) for i,goal in enumerate(targets))
        surrogate=sum(self.detached_loss(b,i,goal) for i,goal in enumerate(targets))
        ga,=t.autograd.grad(direct,a);gb,=t.autograd.grad(surrogate,b)
        t.testing.assert_close(ga,gb,rtol=2e-12,atol=2e-13)
        self.assertEqual([asdict(x) for x in targets],before)
        self.assertEqual(ga.device.type,'cpu')

    def test_initial_collinearity_and_reverse_gradient_on_shared_parameters(self):
        t=self.torch
        for prompt in range(4):
            theta=self.theta();goal=self.target(prompt,theta)
            # Build separate graphs, as the actual trainer does for candidates.
            q=t.softmax(self.scores(theta,prompt),dim=0)
            gold=goal.rewards.index(1)
            gexact,=t.autograd.grad(q[gold]/C,theta)
            forward=self.direct_loss(theta,prompt,goal)*ACCUMULATION
            gf_loss,=t.autograd.grad(forward,theta)
            logq=t.log_softmax(self.scores(theta,prompt),dim=0);q=logq.exp()
            logref=t.tensor(goal.reference_logq,dtype=t.float64,device='cpu')
            reverse_ascent=(q[gold]-BETA*t.sum(q*(logq-logref)))/C
            greverse,=t.autograd.grad(reverse_ascent,theta)
            p=math.exp(goal.reference_logq[gold]);delta=math.expm1(1/BETA)
            alpha=BETA*delta/(1+p*delta)
            t.testing.assert_close(-gf_loss,alpha*gexact,rtol=2e-11,atol=2e-13)
            t.testing.assert_close(greverse,gexact,rtol=2e-11,atol=2e-13)

    def test_four_prompt_accumulation_clip_adam_step_and_state_match(self):
        t=self.torch
        a,b=self.theta(),self.theta()
        optimizers=[t.optim.AdamW([parameter],lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.) for parameter in (a,b)]
        targets=[self.target(i) for i in range(4)]
        for optimizer in optimizers:
            self.assertEqual(len(optimizer.state),0)
            optimizer.zero_grad(set_to_none=False)
        # Common stress scale only in this CPU test forces the clipping branch;
        # it is not a training-hyperparameter recommendation or source change.
        stress_scale=100.
        for prompt,goal in enumerate(targets):
            (stress_scale*self.direct_loss(a,prompt,goal)).backward()
            (stress_scale*self.detached_loss(b,prompt,goal)).backward()
        t.testing.assert_close(a.grad,b.grad,rtol=2e-12,atol=2e-13)
        norm_a=t.nn.utils.clip_grad_norm_([a],1.,error_if_nonfinite=True)
        norm_b=t.nn.utils.clip_grad_norm_([b],1.,error_if_nonfinite=True)
        t.testing.assert_close(norm_a,norm_b,rtol=2e-12,atol=2e-13)
        self.assertGreater(float(norm_a),1.)
        for optimizer in optimizers:optimizer.step()
        t.testing.assert_close(a,b,rtol=2e-12,atol=2e-13)
        for key in ('step','exp_avg','exp_avg_sq'):
            t.testing.assert_close(optimizers[0].state[a][key],optimizers[1].state[b][key],rtol=2e-12,atol=2e-13)


if __name__=='__main__':unittest.main()
