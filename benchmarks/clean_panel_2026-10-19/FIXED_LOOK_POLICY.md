# Proposed fixed-look paired-block policy — review required

No acquisition authorization. Freeze a signed/hash-bound policy and manifest before
collecting any production block. The existing four-block pilot and percentile
bootstrap remain descriptive; they cannot authorize a claim or a continuation.
This policy is a proposal, not a qualified inference implementation.

For each separately qualified cell/cohort, target exactly 120 accepted paired
blocks: 60 AB and 60 BA. This is a budget choice providing 60 observations per
order for diagnostics, not a claim of power or guaranteed 3% precision. No pilot
variance or observed ratio was used to choose it. Collect at most 144 attempts,
72 per order. Generate 72 adjacent order-pairs, each containing one AB and one BA;
shuffle each pair with a frozen seed (20261019). Retain schedule, algorithm and
hash. Once an order has 60 accepted blocks, mark its remaining scheduled slots
not attempted. Never substitute an opposite-order block. Preserve every attempt,
rejection, timestamp and not-attempted reason. No timing-value-based exclusions.

Both locks surround each block; all scalar, sample/resource, provenance and
containment gates remain mandatory. Any identity, scalar parity, output contract,
containment or launch failure halts the cohort for review. Resource rejection
rejects the entire block and may use the next predeclared slot of that order;
maximum 12 rejections per order. A thirteenth rejection, schedule exhaustion,
operator STOP, or 24 hours elapsed from first attempt terminates as incomplete.
No clean-subset salvage. No automatic resumption or further looks under this policy.
Record ambient events; do not kill foreign producers to improve acceptance.

For each accepted block calculate R = native median internal elapsed-per-call /
canonical median internal elapsed-per-call. Inner observations are not paired.
The estimator remains median(R), with equal accepted AB/BA representation.
At the sole final look, use a distribution-free order-statistic interval for the
population median: sorted R[1..120], [R[k], R[121-k]], largest k with
2 * P(Binomial(120, 0.5) <= k-1) <= 0.05. This avoids a data-chosen bootstrap
bandwidth/seed and guarantees at least 95% coverage only under independent
continuous identically distributed block ratios (ties are conservative under
a common median). Balanced order alone does not establish those assumptions.

At that same sole look, publish diagnostic tables of AB/BA medians and medians
by six consecutive acquisition-time bins (20 accepted blocks each). Test order
association by a two-sided permutation difference of median log-ratios, and
lag-1 dependence by a two-sided permutation absolute lag-1 correlation of
log-ratios. Each uses 99,999 permutations, fixed seeds 20261020/20261021, p=(b+1)/100000.
Either p <=0.025 refuses inferential qualification. Zero variance yields no
correlation evidence, not an error. These diagnostics cannot prove independence
or identical distributions. Known correlated producer scheduling, monotonic drift,
or systematic resource-dependent missingness also refuses qualification even if
p-values pass. Report rejection rates/order/time bins; gate-clean selection limits
the estimand to admitted conditions and cannot establish performance in contention.
The reviewer must assess whether pooled common-median inference is defensible;
if not, reject the policy before collection rather than change the estimator later.

Precision passes only if max((estimate-lo)/estimate,(hi-estimate)/estimate) <=0.03.
An interval containing 1.08 (including an endpoint) is borderline. lo>1.08 supports
a slowdown exceeding 8%; hi<1.08 supports below that boundary, not necessarily a
speedup. Failure of precision or crossing means inconclusive at this fixed look:
no extra blocks or ordinary-CI retries. A later study needs a separately reviewed
policy/new cohort and must report this result too. No across-cell familywise claim
is authorized by per-cell 95% intervals; 19 comparisons must be labeled separately.
