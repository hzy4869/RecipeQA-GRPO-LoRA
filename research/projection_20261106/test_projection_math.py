"""CPU numerical checks of the new projection, separate from old model scorer."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import json
import tempfile
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from projection_math import (make_qstar, normalize, projection_terms, forward_loss, validate_token_count)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'exploratory/principled_ordering_20261105'))
from finite_policy_train import log_policy, policy_coefficients

BETA = 1.6363096678310884


def target(beta=BETA):
    _, logref = normalize([-i*.13 for i in range(24)])
    rewards = [int(i == 17) for i in range(24)]
    return make_qstar(logref, rewards, beta)


def centered_difference(function, values, index, epsilon=1e-5):
    left, right = list(values), list(values)
    left[index] -= epsilon
    right[index] += epsilon
    return (function(right)-function(left))/(2*epsilon)


class ProjectionMathTests(unittest.TestCase):
    def test_all24_logit_gradients_match_finite_difference(self):
        goal, scores, count = target(), [math.sin(i*.4)-i*.03 for i in range(24)], 5
        terms = projection_terms(log_policy(scores), goal)
        self.assertAlmostEqual(sum(terms['coefficients']), 0, places=13)
        for index in range(24):
            numerical = centered_difference(lambda v:forward_loss(v,goal,count), scores,index)
            # Trainer uses negative ascent coefficient / C.
            self.assertAlmostEqual(numerical, -terms['coefficients'][index]/count, delta=3e-10)

    def test_nonlinear_shared_parameter_gradient_and_accumulation(self):
        goal, count, theta = target(), 5, [.23,-.41,.17]
        def scores(t):
            return [t[0]*math.sin(i+.2)+t[1]*math.cos(i*.3)+t[0]*t[1]*math.sin(i*.7)+t[2]*(i-11)/20 for i in range(24)]
        coefficients = projection_terms(log_policy(scores(theta)), goal)['coefficients']
        jacobian = [[math.sin(i+.2)+theta[1]*math.sin(i*.7),
                     math.cos(i*.3)+theta[0]*math.sin(i*.7),(i-11)/20] for i in range(24)]
        for j in range(3):
            analytic = -sum(c*jac[j] for c,jac in zip(coefficients,jacobian))/(count*4)
            numerical = centered_difference(lambda t:forward_loss(scores(t),goal,count)/4,theta,j)
            self.assertAlmostEqual(analytic,numerical,delta=3e-10)

    def test_answer_plus_eos_sum_then_only_loss_divided_by_C(self):
        goal = target()
        token_values = [[-1-i*.1,-.3,-.8,-.2,-.05-i*.02] for i in range(24)]
        candidates = [{'token_ids':[10+i,20,30,40,99],'token_count':5,'eos_token_id':99} for i in range(24)]
        count = validate_token_count(candidates)
        self.assertEqual(count,5)
        scores = [sum(values) for values in token_values]
        terms = projection_terms(log_policy(scores),goal)
        # EOS is a real score term with the same chain coefficient as answer tokens.
        for action in (0,17,23):
            eps=1e-5
            right=[list(v) for v in token_values]; left=[list(v) for v in token_values]
            right[action][-1]+=eps; left[action][-1]-=eps
            numerical=(forward_loss([sum(v) for v in right],goal,count)-forward_loss([sum(v) for v in left],goal,count))/(2*eps)
            self.assertAlmostEqual(numerical,-terms['coefficients'][action]/count,delta=3e-10)
        wrong_policy=log_policy([value/count for value in scores])
        self.assertGreater(max(abs(a-b) for a,b in zip(terms['coefficients'],projection_terms(wrong_policy,goal)['coefficients'])),.01)
        self.assertAlmostEqual(forward_loss(scores,goal,10), forward_loss(scores,goal,5)/2, places=14)
        candidates[0]['token_count']=4
        with self.assertRaises(ValueError):
            validate_token_count(candidates)

    def test_qstar_is_fixed_and_copies_mutable_inputs(self):
        _, logref = normalize([-i*.1 for i in range(24)])
        reference, rewards = list(logref), [int(i==17) for i in range(24)]
        fixed=make_qstar(reference,rewards,BETA)
        original=(fixed.q,fixed.logq,fixed.reference_logq,fixed.rewards)
        reference[0]=-999; rewards[17]=0
        for shift in (0,2,-3):
            projection_terms(log_policy([shift*math.sin(i)+i*.02 for i in range(24)]),fixed)
        self.assertEqual((fixed.q,fixed.logq,fixed.reference_logq,fixed.rewards),original)
        with self.assertRaises(FrozenInstanceError):
            fixed.beta=2
        with self.assertRaises(TypeError):
            fixed.q[0]=.1

    def test_beta_multiplier_is_retained_conditionally_on_target(self):
        fixed, policy=target(),log_policy([math.cos(i) for i in range(24)])
        twice=replace(fixed,beta=fixed.beta*2)
        a,b=projection_terms(policy,fixed),projection_terms(policy,twice)
        for x,y in zip(a['coefficients'],b['coefficients']):
            self.assertAlmostEqual(2*x,y,places=14)
        self.assertAlmostEqual(2*a['forward_projection_objective_before_C'],b['forward_projection_objective_before_C'],places=14)
        # Recomputing qstar with a new beta also changes its target: not a claim
        # that the entire beta-dependent training update is generally linear.
        self.assertNotEqual(target(BETA*2).q, fixed.q)

    def test_qstar_mixture_and_initial_gradient_collinearity(self):
        fixed=target(); p=math.exp(fixed.reference_logq[17]); u=math.expm1(1/BETA)
        lam=p*u/(1+p*u); alpha=BETA*u/(1+p*u)
        initial=log_policy(list(fixed.reference_logq))
        projected=projection_terms(initial,fixed)['coefficients']
        reverse=policy_coefficients('exact_kl',initial,list(fixed.rewards),list(fixed.reference_logq),BETA)['coefficients']
        for i in range(24):
            expected=(1-lam)*math.exp(fixed.reference_logq[i])+lam*fixed.rewards[i]
            self.assertAlmostEqual(fixed.q[i],expected,places=14)
            self.assertAlmostEqual(projected[i],alpha*reverse[i],places=14)

    def test_reverse_KL_equals_original_exact_KL_up_to_constant(self):
        fixed=target(); constants=[]
        for shift in (-2,.1,3):
            current=log_policy([shift*math.sin(i*.8)-i*.02 for i in range(24)])
            terms=projection_terms(current,fixed)
            original_loss=-terms['regularized_expected_reward']
            constants.append(BETA*terms['reverse_kl_current_qstar']-original_loss)
        self.assertAlmostEqual(max(constants)-min(constants),0,places=13)

    def test_no_raw_SFT_or_target_only_cross_entropy_coefficient(self):
        fixed=target(); current=log_policy([0.]*24)
        coefficients=projection_terms(current,fixed)['coefficients']
        self.assertEqual(sum(c!=0 for c in coefficients),24)
        self.assertTrue(any(c<0 for c in coefficients))
        self.assertTrue(any(c>0 for c in coefficients))
        self.assertAlmostEqual(sum(coefficients),0,places=13)
        self.assertNotEqual(coefficients,list(fixed.rewards))

    def test_reject_bad_reference_rewards_and_beta(self):
        fixed=target()
        for beta in (0,-1,float('inf'),float('nan')):
            with self.assertRaises(ValueError):
                make_qstar(fixed.reference_logq,fixed.rewards,beta)
        with self.assertRaises(ValueError):
            make_qstar(fixed.reference_logq,[0]*24,BETA)
        with self.assertRaises(ValueError):
            make_qstar([0]*24,fixed.rewards,BETA)
        with self.assertRaises(ValueError):
            make_qstar(fixed.reference_logq,[True if i==17 else False for i in range(24)],BETA)

    def test_trainer_signature_cli_is_json_native(self):
        import projection_train
        payload=projection_train.json_cli(SimpleNamespace(policy=Path('/data/new_policy.json'),beta=BETA,limit=256))
        self.assertEqual(json.loads(json.dumps(payload)),{'policy':'/data/new_policy.json','beta':BETA,'limit':256})

    def test_draft_policy_blocks_before_any_model_or_data_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy=Path(temporary)/'draft.json'
            policy.write_text(json.dumps({'status':'draft'}))
            run=subprocess.run([sys.executable,str(HERE/'projection_train.py'),'--policy',str(policy),
                 '--model-path','/intentionally/missing','--adapter-path','/intentionally/missing','--manifest','/intentionally/missing',
                 '--expected-manifest-sha256','not-real','--image-root','/intentionally/missing','--output-dir','/intentionally/missing',
                 '--reference-cache','/intentionally/missing'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('DRAFT trainer',run.stderr)
            self.assertNotIn('No module named',run.stderr)

    def test_legacy_sources_unchanged(self):
        import hashlib
        expected=('finite_policy_train.py','score_permutations.py','gradient_alignment.py')
        # Verify actual source identities against the already frozen protocol,
        # avoiding a duplicated typo-prone hash as the authority.
        import json
        protocol=json.loads((ROOT/'research/principled_ordering_20261105/FINITE_TRIAL_PROTOCOL_V2.json').read_text())
        for name in expected:
            relative='exploratory/principled_ordering_20261105/'+name
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),protocol['sources_sha256'][relative])


if __name__=='__main__':
    unittest.main()
