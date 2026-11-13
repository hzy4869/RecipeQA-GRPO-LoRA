"""Independent verifier math, corruption rejection, and artifact-byte checks."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import permutations
from pathlib import Path
import hashlib
import json
import math
import struct
import tempfile
import unittest

spec=spec_from_file_location('projection_verifier',Path(__file__).with_name('verify_projection_run.py'))
m=module_from_spec(spec)
__import__('sys').modules[spec.name]=m
spec.loader.exec_module(m)


def reference():
    scores=[-.25*i for i in range(24)]
    normalizer=max(scores)+math.log(sum(math.exp(v-max(scores)) for v in scores))
    return [v-normalizer for v in scores]


class ProjectionVerifierTests(unittest.TestCase):
    def test_binary_qstar_matches_independent_24way_tilt(self):
        ref=reference();beta=1.6363096678310884
        for gold in range(24):
            actual,logs=m.independent_qstar(ref,gold,beta)
            unnormalized=[math.exp(v+float(i==gold)/beta) for i,v in enumerate(ref)]
            expected=[v/sum(unnormalized) for v in unnormalized]
            self.assertLess(max(abs(a-b) for a,b in zip(actual,expected)),1e-14)
            self.assertGreater(actual[gold],math.exp(ref[gold]))
            self.assertAlmostEqual(sum(actual),1)
            self.assertLess(max(abs(math.exp(v)-p) for v,p in zip(logs,actual)),1e-14)

    def test_projection_coefficients_match_finite_difference_and_scale_C(self):
        ref=reference();beta=1.6363096678310884;gold=17;C=5
        values=[v+math.sin(i)*.3 for i,v in enumerate(ref)]
        def normalize(scores):
            z=math.log(sum(math.exp(v) for v in scores))
            return [math.exp(v-z) for v in scores],[v-z for v in scores]
        q,logq=normalize(values);terms=m.independent_terms(q,logq,gold,ref,beta)
        def loss(scores):
            current,current_logs=normalize(scores)
            return m.independent_terms(current,current_logs,gold,ref,beta)['forward_projection_objective_before_C']/C
        for i in range(24):
            plus=values[:];minus=values[:];plus[i]+=1e-5;minus[i]-=1e-5
            derivative=(loss(plus)-loss(minus))/2e-5
            self.assertAlmostEqual(derivative,-terms['coefficients'][i]/C,places=9)
        self.assertAlmostEqual(sum(terms['coefficients']),0,places=13)
        self.assertGreaterEqual(terms['forward_kl_qstar_current'],-1e-14)

    def test_target_fixed_point_and_bad_beta_or_reference_rejected(self):
        ref=reference();beta=1.6363096678310884;gold=3
        q,logq=m.independent_qstar(ref,gold,beta)
        terms=m.independent_terms(q,logq,gold,ref,beta)
        self.assertLess(max(map(abs,terms['coefficients'])),1e-14)
        self.assertAlmostEqual(terms['forward_projection_objective_before_C'],0,places=13)
        for bad_beta in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):m.independent_qstar(ref,gold,bad_beta)
        with self.assertRaises(ValueError):m.independent_qstar([0.]*24,gold,beta)

    def test_inverse_gold_and_unique_exact_reward_all_layouts(self):
        for display in permutations(range(4)):
            record={'display_perm':list(display),'target_order':[2,0,3,1],
                    'target_labels':[chr(65+display.index(i)) for i in [2,0,3,1]]}
            reward=m.exact_labels(record)
            self.assertEqual(sum(reward),1)
            self.assertEqual(m.ACTIONS[reward.index(1)],' '.join(record['target_labels']))
        with self.assertRaises(ValueError):m.exact_labels({'display_perm':[0,1,2,3],'target_order':[0,1,2,3],'target_labels':['D','C','B','A']})

    def test_eos_complete_sequence_reconstruction_and_corruption_rejection(self):
        candidates=[{'text':text,'token_ids':[i+1,99],'token_count':2,'eos_token_id':99} for i,text in enumerate(m.ACTIONS)]
        token_logps=[[-4-i/10,-.1] for i in range(24)]
        result=m.recompute_policy(token_logps,candidates)
        self.assertEqual(result['token_count'],2)
        self.assertAlmostEqual(sum(result['q']),1)
        candidates[2]['token_ids'][-1]=98
        with self.assertRaises(ValueError):m.recompute_policy(token_logps,candidates)
        candidates[2]['token_ids'][-1]=99
        with self.assertRaises(ValueError):m.recompute_policy(token_logps[:-1],candidates)

    def test_model_identity_and_safe_tensor_runtime_fingerprint_from_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            name='base.language_model.layers.0.q_proj.lora_A.weight'
            inventory=[{'name':name,'shape':[1,2],'numel':2}]
            data=struct.pack('<ff',1.5,-.25)
            header=json.dumps({name:{'dtype':'F32','shape':[1,2],'data_offsets':[0,8]}}).encode()
            (root/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+data)
            expected=hashlib.sha256(name.encode()+data).hexdigest()
            self.assertEqual(m.tensor_fingerprint(root,inventory),expected)
            identity=m.model_weight_identity(root,adapter=True)
            self.assertEqual(identity['weight_files'][0]['sha256'],hashlib.sha256((root/'adapter_model.safetensors').read_bytes()).hexdigest())
            (root/'adapter_model.safetensors').write_bytes((root/'adapter_model.safetensors').read_bytes()[:-1])
            with self.assertRaises(ValueError):m.tensor_fingerprint(root,inventory)

    def test_zero_update_is_disclosed_and_corrupt_torch_state_rejected(self):
        self.assertFalse(m.audit_effective_update('same','same',False,'no_effective_adapter_update'))
        with self.assertRaises(ValueError):m.audit_effective_update('same','same',True,'effective_training')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'optimizer.pt';path.write_text('not a zip container')
            with self.assertRaises(ValueError):m.audit_state_file(path)


if __name__ == '__main__':unittest.main()
