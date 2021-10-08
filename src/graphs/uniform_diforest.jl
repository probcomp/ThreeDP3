
# Uniform distribution over directed forests with n nodes.

struct UniformDiForest <: Distribution{LG.SimpleDiGraph} end
const uniform_diforest = UniformDiForest()
(::UniformDiForest)(n) = Gen.random(uniform_diforest, n)

# The number of directed forests on n nodes is (n+1) ^ (n-1) as given by Cayley's formula.
function Gen.logpdf(::UniformDiForest, diforest::LG.SimpleDiGraph, n::Int)
    n != LG.nv(diforest) && return -Inf
    - ((n - 1) * log(n+1))
end

export uniform_diforest
