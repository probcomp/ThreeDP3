import LightGraphs as LG
import MetaGraphs

# Distribution on directed forests, where the probability mass assigned to trees with the final node
# as the root is TREE_PROBABILITY_MASS (uniform over all directed forests that satisfy this property)
# and then the remainder of probability mass (1.0 - TREE_PROBABILITY_MASS) is spread uniformly over all
# directed forests that do not satisfy the property.

TREE_PROBABILITY_MASS = (1.0 - 1e-10)

struct TreePreferenceDiForest <: Gen.Distribution{LG.SimpleDiGraph} end
const tree_preference_diforest = TreePreferenceDiForest()
(::TreePreferenceDiForest)(n) = Gen.random(tree_preference_diforest, n)

function Gen.random(::TreePreferenceDiForest, n::Int)
    graph_with_edges(n,[])
end


# The number of directed forests on n nodes is (n+1) ^ (n-1) as given by Cayley's formula.
function Gen.logpdf(::TreePreferenceDiForest, diforest::LG.SimpleDiGraph, n::Int)
    n != nv(diforest) && return -Inf

    num_directed_forests = (n+1) ^ (n-1)
    num_trees_with_root_last_node = (n ^ (n-2)) / n

    if isTree(diforest) && (n ∈ S.rootsOfForest(diforest))
        return log(TREE_PROBABILITY_MASS / num_trees_with_root_last_node)
    else
        return log((1.0 - TREE_PROBABILITY_MASS) / (num_directed_forests - num_trees_with_root_last_node))
    end
end

export tree_preference_diforest
