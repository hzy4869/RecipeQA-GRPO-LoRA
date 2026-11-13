"""Pure-CPU math for a DRAFT finite-policy forward-KL projection.

All24 actions exist before rewards are read; no sampling, forced success or
length-normalized action policy. q is softmax of full answer+EOS log-probability.
"""
from dataclasses import dataclass
import math

N_ACTIONS = 24


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(scores):
    require(len(scores) == N_ACTIONS and all(math.isfinite(v) for v in scores), 'Need24 finite complete-string log scores')
    maximum = max(scores)
    logz = maximum + math.log(math.fsum(math.exp(v-maximum) for v in scores))
    logq = tuple(v-logz for v in scores)
    q = tuple(math.exp(v) for v in logq)
    require(abs(math.fsum(q)-1) <= 1e-9, 'Policy does not normalize')
    return q, logq


@dataclass(frozen=True)
class FrozenQStar:
    q: tuple[float, ...]
    logq: tuple[float, ...]
    reference_logq: tuple[float, ...]
    rewards: tuple[int, ...]
    beta: float


def make_qstar(reference_logq, exact_rewards, beta):
    require(math.isfinite(beta) and beta > 0, 'Explicit positive fixed beta required')
    require(len(reference_logq) == N_ACTIONS and all(math.isfinite(v) for v in reference_logq), 'Need24 fixed reference log probabilities')
    require(len(exact_rewards) == N_ACTIONS and all(type(r) is int and r in (0,1) for r in exact_rewards)
            and sum(exact_rewards) == 1, 'Reward must be unique complete-order exact0/1')
    require(abs(math.fsum(math.exp(v) for v in reference_logq)-1) <= 1e-9, 'Reference log probabilities are not normalized')
    q, logq = normalize([v+r/beta for v,r in zip(reference_logq, exact_rewards)])
    return FrozenQStar(q, logq, tuple(reference_logq), tuple(exact_rewards), float(beta))


def validate_token_count(candidates):
    require(len(candidates) == N_ACTIONS, 'All24 candidates required')
    counts = set()
    for row in candidates:
        require(row['token_count'] == len(row['token_ids']) and row['token_count'] > 1
                and row['token_ids'][-1] == row['eos_token_id'], 'Token count must include canonical EOS')
        counts.add(row['token_count'])
    require(len(counts) == 1, 'Existing protocol requires equal complete EOS lengths; do not introduce action-length normalization')
    return next(iter(counts))


def projection_terms(current_policy, target):
    """Detached ascent coefficients of raw sequence logp, before division by C.

    Minimizing beta*KL(qstar||q)/C gives gradient beta*(q-qstar)/C
    wrt complete sequence scores, so the old negative-weighted logp backward
    interface receives beta*(qstar-q), not qstar alone and not raw SFT gold.
    """
    q, logq = current_policy['q'], current_policy['logq']
    require(len(q) == len(logq) == N_ACTIONS and all(math.isfinite(v) for v in logq)
            and all(math.isfinite(v) and v >= 0 for v in q) and abs(math.fsum(q)-1) <= 1e-9, 'Bad current closed policy')
    require(all(abs(math.exp(v)-p) <= 1e-9 for p,v in zip(q,logq)), 'Current q and logq disagree')
    coefficients = [target.beta*(p-current) for p,current in zip(target.q,q)]
    require(abs(math.fsum(coefficients)) <= 1e-9, 'Projection coefficients do not sum to zero')
    forward_kl = math.fsum(p*(lp-lq) for p,lp,lq in zip(target.q,target.logq,logq))
    reverse_kl = math.fsum(p*(lq-lp) for p,lq,lp in zip(q,logq,target.logq))
    kl_reference = math.fsum(p*(lq-lr) for p,lq,lr in zip(q,logq,target.reference_logq))
    gold = target.rewards.index(1)
    return {'coefficients': coefficients, 'forward_kl_qstar_current': forward_kl,
            'reverse_kl_current_qstar': reverse_kl, 'forward_projection_objective_before_C':target.beta*forward_kl,
            'exact_expected_reward':q[gold], 'kl_q_reference':kl_reference,
            'regularized_expected_reward':q[gold]-target.beta*kl_reference,
            'closed_ce_nll':-logq[gold], 'gold_index':gold, 'active_beta':target.beta,
            'qstar':list(target.q), 'qstar_logq':list(target.logq)}


def forward_loss(scores, target, canonical_token_count):
    require(type(canonical_token_count) is int and canonical_token_count > 1, 'C must include answer and EOS')
    q, logq = normalize(scores)
    return target.beta*math.fsum(p*(lp-lq) for p,lp,lq in zip(target.q,target.logq,logq))/canonical_token_count
