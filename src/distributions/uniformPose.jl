import GenDirectionalStats: uniform_rot3
import Rotations

struct UniformPose <: Gen.Distribution{Pose} end
const uniformPose = UniformPose()
(::UniformPose)(xmin, xmax, ymin, ymax, zmin, zmax) = Gen.random(
    uniformPose, xmin, xmax, ymin, ymax, zmin, zmax)

function Gen.random(::UniformPose,
                    xmin::Real, xmax::Real,
                    ymin::Real, ymax::Real,
                    zmin::Real, zmax::Real,
                   )::Pose
  Pose([Gen.uniform(xmin, xmax),
        Gen.uniform(ymin, ymax),
        Gen.uniform(zmin, zmax)],
        uniform_rot3())
end
function Gen.logpdf(::UniformPose,
                    val::Pose,
                    xmin::Real, xmax::Real,
                    ymin::Real, ymax::Real,
                    zmin::Real, zmax::Real)
  +(Gen.logpdf(Gen.uniform, val.pos[1], xmin, xmax),
    Gen.logpdf(Gen.uniform, val.pos[2], ymin, ymax),
    Gen.logpdf(Gen.uniform, val.pos[3], zmin, zmax),
    Gen.logpdf(uniform_rot3, Rotations.UnitQuaternion(val.orientation)))
end
Gen.has_output_grad(::UniformPose) = false
Gen.has_argument_grads(::UniformPose) = fill(false, 6) |> Tuple
function Gen.logpdf_grad(::UniformPose,
                         xmin::Real, xmax::Real,
                         ymin::Real, ymax::Real,
                         zmin::Real, zmax::Real)
  fill(nothing, 7) |> Tuple
end

export uniformPose