#!/usr/bin/env python3
"""Independent CPU reference for the unexecuted forward-KL projection draft.

No model/GPU/SSH. Uses explicit scalar objectives and finite differences rather
than borrowing the draft's derivative. Audits all256 frozen training targets.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REF=ROOT/'.state/remote_principled_20261105/exploratory/principled_ordering_20261105/finite_trials/reference256_flow44/reference_policy.jsonl'
REF_SHA='70e31479446490261f689f6ac938e75d1e6a76b22d111f04460d64fa218d0bfb'
BETA=1.6363096678310884

def check(v,msg):
    if not v:raise ValueError(msg)
def dot(x,y):return math.fsum(a*b for a,b in zip(x,y))
def softmax(scores):
    weights=[math.exp(x-max(scores)) for x in scores];z=math.fsum(weights)
    return [x/z for x in weights]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('projection_draft',HERE/'projection_math.py')
    draft=importlib.util.module_from_spec(spec)
    # dataclass annotation resolution expects the module to be discoverable.
    import sys
    sys.modules[spec.name]=draft;spec.loader.exec_module(draft)
    maximum_fd=maximum_initial=maximum_mixture=maximum_sum=0.
    beta=BETA;gold=7;c=5
    ref=softmax([math.sin(i*.7) for i in range(24)])
    rewards=[int(i==gold) for i in range(24)];logref=[math.log(p) for p in ref]
    target=draft.make_qstar(logref,rewards,beta)
    scores=[math.cos(i*.9)-.2*math.sin(i*.3) for i in range(24)]
    q=softmax(scores)
    # Independent target through exact unique-gold mixture, not log-softmax.
    t=math.exp(1/beta);z=1+ref[gold]*(t-1);lam=ref[gold]*(t-1)/z
    star=[(1-lam)*p+lam*r for p,r in zip(ref,rewards)]
    check(max(abs(x-y) for x,y in zip(star,target.q))<1e-14,'Soft target != reference/gold mixture')
    def objective(s):
        current=softmax(s)
        return beta*math.fsum(p*math.log(p/q0) for p,q0 in zip(star,current))/c
    terms=draft.projection_terms({'q':q,'logq':[math.log(p) for p in q]},target)
    for index in range(24):
        plus=scores[:];minus=scores[:];plus[index]+=1e-5;minus[index]-=1e-5
        derivative=(objective(plus)-objective(minus))/2e-5
        error=abs(derivative+terms['coefficients'][index]/c)
        maximum_fd=max(maximum_fd,error)
        check(error<1e-9,'Forward finite difference failed')
    check(abs(draft.forward_loss(scores,target,c)-objective(scores))<1e-13,'Forward loss scalar differs')
    # Shared nonlinear score coordinates: chain rule is not restricted to an
    # independently adjustable 24-logit model.
    theta=[.13,-.2,.31]
    features=[[math.sin((i+1)*(j+1)*.21) for j in range(3)] for i in range(24)]
    def param_scores(th):return [scores[i]+dot(row,th)+.1*th[0]*th[1]*(i-12)/12 for i,row in enumerate(features)]
    qs=softmax(param_scores(theta))
    coeff=draft.projection_terms({'q':qs,'logq':[math.log(p) for p in qs]},target)['coefficients']
    for d in range(3):
        plus=theta[:];minus=theta[:];plus[d]+=1e-5;minus[d]-=1e-5
        observed=(objective(param_scores(plus))-objective(param_scores(minus)))/2e-5
        derivatives=[row[d]+(.1*theta[1]*(i-12)/12 if d==0 else .1*theta[0]*(i-12)/12 if d==1 else 0.) for i,row in enumerate(features)]
        expected=-dot(coeff,derivatives)/c
        maximum_fd=max(maximum_fd,abs(expected-observed));check(abs(expected-observed)<1e-9,'Shared-parameter nonlinear chain rule failed')
    shifted=[x+31 for x in scores]
    check(abs(objective(scores)-objective(shifted))<1e-13,'Closed-policy constant-score shift invariance failed')
    check(sha(REF)==REF_SHA,'Frozen training reference differs')
    rows=[json.loads(line) for line in REF.read_text().splitlines()]
    check(len(rows)==256,'Not full fixed256 reference')
    alphas=[];lambdas=[];gaps=[]
    for row in rows:
        logref=row['logq'];ref=row['q'];reward=row['exact_rewards'];gold=reward.index(1);p=ref[gold]
        check(sum(reward)==1,'Reward not exact binary unique gold')
        target=draft.make_qstar(logref,reward,beta)
        forward=draft.projection_terms({'q':ref,'logq':logref},target)['coefficients']
        z=1+p*(t-1);lam=p*(t-1)/z;alpha=beta*(t-1)/z
        exact=[prob*(r-p) for prob,r in zip(ref,reward)]
        mixture=[(1-lam)*prob+lam*r for prob,r in zip(ref,reward)]
        err=max(abs(a-alpha*b) for a,b in zip(forward,exact));maximum_initial=max(maximum_initial,err)
        maximum_sum=max(maximum_sum,abs(math.fsum(forward)))
        maximum_mixture=max(maximum_mixture,max(abs(a-b) for a,b in zip(target.q,mixture)))
        check(err<1e-13 and maximum_sum<1e-12 and maximum_mixture<1e-13,'Initial gradient/mixture mismatch')
        # qstar optimality identity for an independent current policy q.
        j=q[gold]-beta*dot(q,[math.log(a)-b for a,b in zip(q,logref)])
        js=beta*math.log(z)
        gap=js-j;kl=dot(q,[math.log(a)-b for a,b in zip(q,target.logq)])
        check(abs(gap-beta*kl)<1e-12,'Reverse objective gap identity failed')
        alphas.append(alpha);lambdas.append(lam);gaps.append(abs(gap-beta*kl))
    def stats(values):return {'min':min(values),'median':statistics.median(values),'max':max(values),'mean':statistics.mean(values)}
    report={'status':'passed_cpu_reference_math_only_no_training_or_gpu_execution','source_sha256':sha(Path(__file__)),'draft_math_sha256':sha(HERE/'projection_math.py'),'reference_sha256':REF_SHA,'beta':beta,'targets_checked':256,'max_finite_difference_error':maximum_fd,'max_initial_collinearity_absolute_residual':maximum_initial,'max_target_mixture_absolute_residual':maximum_mixture,'max_coefficient_sum_residual':maximum_sum,'max_reverse_gap_identity_residual':max(gaps),'initial_forward_over_exact_gradient_scale':stats(alphas),'gold_hard_label_mixture_weight_lambda':stats(lambdas),'limits':['Only math/reference implementation checked; no model, optimizer, pipeline or GPU smoke has run.','Raw sequence SFT is a different objective from closed-policy hard-label cross entropy.','Forward target is exactly a reference-distillation and hard-label mixture, with per-input weights fixed by qref and beta.','Same64updates/256visits is not identical compute; record candidate forwards/backwards and walltime.']}
    (HERE/'INDEPENDENT_MATH_REFERENCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
