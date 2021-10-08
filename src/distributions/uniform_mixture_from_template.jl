#################################
# uniform mixture from template #
#################################

import Gen: geometric
import LinearAlgebra
import NearestNeighbors
import Memoize: @memoize
import StatsFuns: logsumexp

struct UniformMixtureFromTemplate <: Distribution{Matrix{Float64}} end

"""
    Y = uniform_mixture_from_template(X, p_outlier, radius, bounds)

Distribution over variable-sized point clouds representing i.i.d. samples from
a mixture of uniform distributions over:

* balls centered at each point of the template point cloud `X` (inliers), and
* a big box with bounds `bounds` (outliers).

That is, this model is an infinite mixture (for `n`) of finite mixtures (for
each of the `n` points in the point cloud).

The distribution on `n` is geometric, with expected value proportional to the
number of points in the template point cloud `X`.

The inner mixture component corresponding to the box `bounds` has weight
`p_outlier`, and for each `i`, the inner mixture component corresponding to the
sphere centered at `X[:,i]` with radius `radius` has weight `(1 - p_outlier) /
m`, where `size(X) = (3, m)`.

This distribution is "collapsed," in the sense that the `logpdf` marginalizes
out the inlier/outlierness.

The returned matrix `Y` has dimensions `3 × n`.
"""
const uniform_mixture_from_template = UniformMixtureFromTemplate()

"""
Parameter of the geometric distribution for number of points in the random
point cloud `Y`, given the number of points in the template point cloud `X`
"""
function _UniformMixtureFromTemplate_p_numpoints_Y(numpoints_X::Integer)
    return min(1 - eps(), 3 / numpoints_X)
end

# Reference implementation of the sampler -- to clarify the semantics of the
# distribution, even if it's not used in code.
function Gen.random(::UniformMixtureFromTemplate, X::Matrix{Float64},
                    p_outlier::Float64, radius::Float64,
                    bounds::Tuple)
    @assert size(X, 1) == 3
    m = size(X, 2)
    n = Gen.random(geometric, _UniformMixtureFromTemplate_p_numpoints_Y(m))
    mins = [bounds[1], bounds[3], bounds[5]]
    maxs = [bounds[2], bounds[5], bounds[6]]

    outlier = (rand(n) .< p_outlier)
    return reduce(hcat, (if outlier[i]
                             mins + rand(3) .* (maxs - mins)
                         else
                             v = LinearAlgebra.normalize(randn(3))
                             X[:, rand(1:m)] + radius * rand() * v
                         end for i in 1:n);
                  init=zeros(3, 0))
end

@memoize function get_tree_from_cloud(Y)
    NearestNeighbors.KDTree(Y)
end

function Gen.logpdf(
        ::UniformMixtureFromTemplate, Y::Matrix{Float64}, X::Matrix{Float64},
        p_outlier::Float64, radius::Float64, bounds::Tuple)
    size(X, 1) == 3 || error("X must have size 3 × something, each column represents a point.  Got size(X) = $(size(X))")
    size(Y, 1) == 3 || error("Y must have size 3 × something, each column represents a point.  Got size(Y) = $(size(Y))")
    all((Y .≥ [bounds[1], bounds[3], bounds[5]]) .&
        (Y .≤ [bounds[2], bounds[4], bounds[6]])
       ) || @warn("Some points in Y were out of bounds")

    tree = get_tree_from_cloud(Y)

    m = size(X, 2)
    n = size(Y, 2)
    logp_numpoints = Gen.logpdf(geometric, n,
                                _UniformMixtureFromTemplate_p_numpoints_Y(m))
    r = radius

    # all_idxs is an array of arrays, where the array all_idxs[i] contains all indices j such that
    # the points X[:,i] and Y[:,j] are within a distance of r units of each other.
    all_idxs = NearestNeighbors.inrange(tree, X, r)

    # For each point in Y, count how many points in X it is within r units of.
    num_close_pts = zeros(n)
    for idxs in all_idxs
        num_close_pts[idxs] .= num_close_pts[idxs] .+ 1
    end

    # Compute volume of inlier sphere, outlier bounds box, and the difference.
    Volume_inlier = 4.0 /3 * pi * r^3
    Volume_outlier = (bounds[2] - bounds[1]) * (bounds[4] - bounds[3]) * (bounds[6] - bounds[5])

    # Consider each column `Y[:,j]` to represent a single sample.
    # Then `logp_y_and_inlier` is an array whose `j`th entry is
    #     log( p( the jth sample is an inlier && the jth sample equals Y[:,j] ) )
    logp_y_and_inlier = log.( ((1-p_outlier) / Volume_inlier / m) .* num_close_pts )
    logp_y_and_outlier = log(p_outlier / Volume_outlier)

    # Log probability of each point (i.e., each column of `Y`), with the
    # inlier/outlierness marginalized out.
    @assert ndims(logp_y_and_inlier) == 1
    logp_y = logsumexp.(logp_y_and_inlier,
                        logp_y_and_outlier)

    # Return product of likelihoods across points.
    return logp_numpoints + sum(logp_y)
end



struct UniformMixtureFromTemplateMultiCloud <: Distribution{Matrix{Float64}} end
const uniform_mixture_from_template_multi_cloud = UniformMixtureFromTemplateMultiCloud()

function Gen.logpdf(
        ::UniformMixtureFromTemplateMultiCloud, Y::Matrix{Float64}, X::Array{Matrix{Float64}},
        p_outlier::Float64, radius::Float64, bounds::Tuple)
   log_pdfs = [logpdf(uniform_mixture_from_template, Y, X[i], p_outlier, radius, bounds) for i in 1:length(X)]
   sum(log_pdfs)
end

export uniform_mixture_from_template, uniform_mixture_from_template_multi_cloud
