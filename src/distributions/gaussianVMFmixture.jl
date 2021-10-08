import GenDirectionalStats: vmf_rot3
import Rotations

struct GaussianVMFMixture <: Gen.Distribution{Pose} end
const gaussianVMF_mixture = GaussianVMFMixture()
function Gen.random(
    ::GaussianVMFMixture,
    centers::Array{Pose},
    positionStdev::Real,
    orientationKappa::Real,
)::Pose
    N = length(centers)
    i = categorical(fill(1.0/N, N))
    rot = Gen.random(vmf_rot3, Rotations.UnitQuaternion(1.0, 0.0, 0.0, 0.0), orientationKappa)
    offset = Pose(
        [
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
        ],
        rot,
    )

    Pose(centers[i].pos + offset.pos, centers[i].orientation * offset.orientation)
end

function Gen.logpdf(
    ::GaussianVMFMixture,
    val::Pose,
    centers::Array{Pose},
    positionStdev::Real,
    orientationKappa::Real,
)
    N = length(centers)
    logpdfs = [
        logpdf(gaussianVMF, val, c, positionStdev, orientationKappa) for c in centers
    ]
    logsumexp(logpdfs) - log(N)
end
Gen.has_output_grad(::GaussianVMFMixture) = false
Gen.has_argument_grads(::GaussianVMFMixture) = (false, false, false)
function Gen.logpdf_grad(
    ::GaussianVMFMixture,
    center::Pose,
    positionStdev::Real,
    orientationKappa::Real,
)
    (nothing, nothing, nothing, nothing)
end

export gaussianVMF_mixture


struct GaussianVMFWeightedMixture <: Gen.Distribution{Pose} end
const gaussianVMF_weighted_mixture = GaussianVMFWeightedMixture()
function Gen.random(
    ::GaussianVMFWeightedMixture,
    centers::Array{Pose},
    weights::Array{<:Real},
    positionStdev::Real,
    orientationKappa::Real,
)::Pose
    N = length(centers)
    i = categorical(weights)
    rot = Gen.random(vmf_rot3, Rotations.UnitQuaternion(1.0, 0.0, 0.0, 0.0), orientationKappa)
    offset = Pose(
        [
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
        ],
        rot,
    )

    Pose(centers[i].pos + offset.pos, centers[i].orientation * offset.orientation)
end

function Gen.logpdf(
    ::GaussianVMFWeightedMixture,
    val::Pose,
    centers::Array{Pose},
    weights::Array{<:Real},
    positionStdev::Real,
    orientationKappa::Real,
)
    N = length(centers)
    logpdfs = [
        log(w) + logpdf(gaussianVMF, val, c, positionStdev, orientationKappa) for (w,c) in zip(weights,centers)
    ]
    logsumexp(logpdfs) - log(N)
end
Gen.has_output_grad(::GaussianVMFWeightedMixture) = false
Gen.has_argument_grads(::GaussianVMFWeightedMixture) = (false, false, false)
function Gen.logpdf_grad(
    ::GaussianVMFWeightedMixture,
    center::Pose,
    weights::Array{Real},
    positionStdev::Real,
    orientationKappa::Real,
)
    (nothing, nothing, nothing, nothing)
end

export gaussianVMF_weighted_mixture
