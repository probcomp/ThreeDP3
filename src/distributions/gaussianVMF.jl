import GenDirectionalStats: vmf_rot3
import Rotations

struct GaussianVMF <: Gen.Distribution{Pose} end
const gaussianVMF = GaussianVMF()
function Gen.random(
    ::GaussianVMF,
    center::Pose,
    positionStdev::Real,
    orientationKappa::Real,
)::Pose
    rot = Gen.random(vmf_rot3, Rotations.UnitQuaternion(1.0, 0.0, 0.0, 0.0), orientationKappa)
    offset = Pose(
        [
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
            Gen.normal(0, positionStdev),
        ],
        rot,
    )

    Pose(center.pos + offset.pos, center.orientation * offset.orientation)
end

function Gen.logpdf(
    ::GaussianVMF,
    val::Pose,
    center::Pose,
    positionStdev::Real,
    orientationKappa::Real,
)
    offset = Pose(center.pos - val.pos, center.orientation \ val.orientation)
    return +(
        Gen.logpdf(Gen.normal, offset.pos[1], 0, positionStdev),
        Gen.logpdf(Gen.normal, offset.pos[2], 0, positionStdev),
        Gen.logpdf(Gen.normal, offset.pos[3], 0, positionStdev),
        Gen.logpdf(
            vmf_rot3,
            Rotations.UnitQuaternion(offset.orientation),
            Rotations.UnitQuaternion(1.0, 0.0, 0.0, 0.0),
            orientationKappa,
        ),
    )
end
Gen.has_output_grad(::GaussianVMF) = false
Gen.has_argument_grads(::GaussianVMF) = (false, false, false)
function Gen.logpdf_grad(
    ::GaussianVMF,
    center::Pose,
    positionStdev::Real,
    orientationKappa::Real,
)
    (nothing, nothing, nothing, nothing)
end

export gaussianVMF
