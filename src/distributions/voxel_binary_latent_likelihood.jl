

struct VoxelBinaryLatentLikelihood <: Distribution{Matrix{Float64}} end

const voxel_binary_latent_likelihood = VoxelBinaryLatentLikelihood()

function Gen.logpdf(
        ::VoxelBinaryLatentLikelihood, Y::Matrix{Float64}, X::Matrix{Float64},
        p_outlier::Float64, radius::Float64, bounds::Geometry.Bounds)
    size(X, 1) == 3 || error("X must have size 3 × something, each column represents a point.  Got size(X) = $(size(X))")
    size(Y, 1) == 3 || error("Y must have size 3 × something, each column represents a point.  Got size(Y) = $(size(Y))")
    all((Y .≥ [bounds.xmin, bounds.ymin, bounds.zmin]) .&
        (Y .≤ [bounds.xmax, bounds.ymax, bounds.zmax])
       ) || @warn("Some points in Y were out of bounds")

    tree = get_tree_from_cloud(Y)

    m = size(X, 2)
    n = size(Y, 2)


    # all_idxs is an array of arrays, where the array all_idxs[i] contains all indices j such that
    # the points X[:,i] and Y[:,j] are within a distance of r units of each other.
    all_idxs = NearestNeighbors.inrange(tree, X, r)

    latent_has_corresponding_point_in_obs = fill(false, m)
    obs_has_no_corresponding_latent = fill(true, n)

    for (i,idxs) in enumerate(all_idxs)
        if length(idx) > 0
            latent_has_corresponding_point_in_obs[i] = true
        end
        obs_has_no_corresponding_latent[idx] .= false
    end

    num_latent_cells = 50000

    latent_active = m

    latent_active_fired = sum(latent_has_corresponding_point_in_obs)
    latent_active_not_fired = latent_active - latent_active_fired

    latent_inactive = num_latent_cells - latent_active

    latent_inactive_fired = sum(obs_has_no_corresponding_latent)
    latent_inactive_not_fired = latent_inactive - latent_inactive_fired

    @assert all(map(x -> x>= 0.0, [latent_active_fired, latent_active_not_fired, latent_inactive_fired, latent_inactive_not_fired,
        latent_active, latent_inactive]))
    @assert sum([latent_active_fired, latent_active_not_fired, latent_inactive_fired, latent_inactive_not_fired]) == num_latent_cells

    ACTIVE_FIRE_PROB = 0.999
    INACTIVE_NOT_FIRE_PROB = 0.999

    (
        log(ACTIVE_FIRE_PROB) * latent_active_fired + log(1.0 - ACTIVE_FIRE_PROB) * latent_active_not_fired +
        log(INACTIVE_NOT_FIRE_PROB) * latent_inactive_not_fired + log(1.0 - INACTIVE_NOT_FIRE_PROB) * latent_inactive_fired
    )
end



struct VoxelBinaryLatentLikelihoodMultiCloud <: Distribution{Matrix{Float64}} end
const voxel_binary_latent_likelihood_multi_cloud = VoxelBinaryLatentLikelihoodMultiCloud()

function Gen.logpdf(
        ::VoxelBinaryLatentLikelihoodMultiCloud, Y::Matrix{Float64}, X::Array{Matrix{Float64}},
        p_outlier::Float64, radius::Float64, bounds::Geometry.Bounds)
   log_pdfs = [logpdf(voxel_binary_latent_likelihood, Y, X[i], p_outlier, radius, bounds) for i in 1:length(X)]
   sum(log_pdfs)
end

export uniform_mixture_from_template, uniform_mixture_from_template_multi_cloud
