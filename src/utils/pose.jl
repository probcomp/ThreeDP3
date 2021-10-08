
import Rotations
import StaticArrays: StaticVector, @SVector
import PoseComposition: Pose
import LinearAlgebra: I
import Base: convert

function Pose(orientation::Rotations.Rotation{3})::Pose
    return Pose(zeros(3), orientation)
end

const IDENTITY_ORIENTATION = Rotations.UnitQuaternion(1.0,0.0,0.0,0.0)

function Pose(pos::AbstractVector{<:Real})::Pose
    return Pose(pos, IDENTITY_ORIENTATION)
end

function Pose(x::Real, y::Real, z::Real)::Pose
    return Pose([x,y,z], IDENTITY_ORIENTATION)
end


const IDENTITY_POSE = Pose(zeros(3),IDENTITY_ORIENTATION)

rotation_pose(pose::Pose) = Pose(zeros(3), pose.orientation)
translation_pose(pose::Pose) = Pose(pose.pos, IDENTITY_ORIENTATION)


"""
    c_relative_to_a::Pose = get_c_relative_to_a(b_relative_to_a::Pose, c_relative_to_b::Pose)

Given the pose of b relative to a, and the pose of c relative to b, return the pose of c relative to a.
"""
function get_c_relative_to_a(b_relative_to_a::Pose, c_relative_to_b::Pose)
    pos = b_relative_to_a.pos + (b_relative_to_a.orientation * c_relative_to_b.pos)
    orientation = b_relative_to_a.orientation * c_relative_to_b.orientation
    return Pose(pos, orientation)
end

function pose_pow(pose::Pose, t::Int)
    if t == 0
        return IDENTITY_POSE
    end
    p = pose
    for i in 1:(t-1)
        p = get_c_relative_to_a(p, pose)
    end
    p
end


"""
    b_relative_to_a::Pose = inverse_pose(a_relative_to_b::Pose)

Given the pose of b relative to a, return the pose of a relative to b.
"""
function inverse_pose(a_relative_to_b::Pose)
    pos = a_relative_to_b.orientation' * (-a_relative_to_b.pos)
    orientation = inv(a_relative_to_b.orientation)
    b_relative_to_a = Pose(pos, orientation)
    return b_relative_to_a
end

"""
    points = get_points_in_frame_b(points_in_frame_a::Matrix{<:Real}, b_relative_to_a::Pose)

Given coordinates of points in one coordinate frame, get the coordinates of the points in another coordinate frame.

The points that are taken as input and returned as output are both 3 x n matrices.
"""
function get_points_in_frame_b(points_in_frame_a::Matrix{<:Real}, b_relative_to_a::Pose)
    if size(points_in_frame_a)[1] != 3
        error("expected an 3 x n matrix")
    end
    n = size(points_in_frame_a[2])
    return b_relative_to_a.orientation' * (points_in_frame_a .- b_relative_to_a.pos)
end

function move_points_to_frame_b(points_in_frame_a::Matrix{<:Real}, b_relative_to_a::Pose)
    if size(points_in_frame_a)[1] != 3
        error("expected an 3 x n matrix")
    end
    n = size(points_in_frame_a[2])
    return b_relative_to_a.orientation * points_in_frame_a .+ b_relative_to_a.pos
end

function convert(::Type{Matrix{Float32}}, pose::Pose)
    R = Matrix{Float32}(pose.orientation)
    mat = Matrix{Float32}(I, 4, 4)
    mat[1:3,1:3] = R
    mat[1:3,4] = pose.pos
    return mat
end

export Pose, close, isapprox, IDENTITY_ORIENTATION, IDENTITY_POSE, inverse_pose, get_points_in_frame_b,
        get_c_relative_to_a, move_points_to_frame_b, pose_pow, rotation_pose, translation_pose
