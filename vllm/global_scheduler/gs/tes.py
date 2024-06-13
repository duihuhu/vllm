from global_meta import DistPolicy
policy = "random"
policy = DistPolicy(policy)
if policy == DistPolicy.RANDOM:
    print("a")