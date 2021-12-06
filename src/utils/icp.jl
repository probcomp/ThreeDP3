import Clustering, NearestNeighbors, LinearAlgebra
import Statistics
import StaticArrays

function get_transform_between_two_registered_clouds(c1, c2)
    centroid_1 = mean(c1, dims=2)
    centroid_2 = mean(c2, dims=2)
    centered_1 = c1 .- centroid_1
    centered_2 = c2 .- centroid_2
    H = centered_1 * centered_2'
    s = LinearAlgebra.svd(H)
    R = (s.V * permutedims(s.U))
    if LinearAlgebra.det(R) < 0
        s = LinearAlgebra.svd(R)
        V = s.V
        V[:,3] .= -1.0 .* V[:,3]
        R = V * permutedims(s.U)
    end
    rot = Rotations.RotMatrix(SMatrix{3,3}(R))

    T = (centroid_2 .- rot * centroid_1)
    T = T[:]
    Pose(T, rot)
end


"""
Returns the rigid transformation `T::Pose` such that `T * csource` is as close
as possible to `ctarget`.  Here `ctarget` and `csource` are point clouds, that
is, 3×something matrices where each column represents a point.
"""
function icp(c1, c2; iterations=10, c1_tree=nothing)
    T = IDENTITY_POSE
    if isnothing(c1_tree)
        c1_tree = NearestNeighbors.KDTree(c1)
    else
        @assert length(c1_tree.data) == size(c1)[2]
    end
    for _ in 1:iterations
        idxs, dists = NearestNeighbors.nn(c1_tree, c2)
        neighbors = c1[:,idxs]
        new_T = get_transform_between_two_registered_clouds(c2, neighbors)
        c2 = new_T.pos .+ (new_T.orientation * c2)
        T = get_c_relative_to_a(new_T, T)
    end

    T
end

function icp_object_pose(init_pose, obs, get_cloud_func;
    c1_tree=nothing, random_rotation_initialization=false, outer_iterations=5, iterations=10)
    p = init_pose
    if random_rotation_initialization
        p = Pose(p.pos, uniform_rot3())
    end

    if isnothing(c1_tree)
        c1_tree = NearestNeighbors.KDTree(obs)
    else
        @assert length(c1_tree.data) == size(obs)[2]
    end

    c = get_cloud_func(p)
    for _ in 1:outer_iterations
        T = icp(obs, c; iterations=iterations, c1_tree=c1_tree)
        p = get_c_relative_to_a(T,p)
        c = get_cloud_func(p)
    end

    p
end



function icp_project_to_planar_contact(
    obs, get_cloud_func, parent_pose, parent_box, parent_face, child_box, child_face,
    x,y,ang; c1_tree=nothing, outer_iterations=5, iterations=10)

    if isnothing(c1_tree)
        c1_tree = NearestNeighbors.KDTree(obs)
    else
        @assert length(c1_tree.data) == size(obs)[2]
    end

    contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
    p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)
    c = get_cloud_func(p)

    for _ in 1:outer_iterations
        T = icp(obs, c; iterations=iterations, c1_tree=c1_tree)
        p = get_c_relative_to_a(T,p)

        (parent_face_idx, child_face_idx),_ = get_closest_planar_contact(parent_pose, parent_box, p, child_box)
        new_parent_face, new_child_face = S.BOX_SURFACE_IDS[parent_face_idx], S.BOX_SURFACE_IDS[child_face_idx]
        pc = S.closestApproximatingContact(
            (parent_pose * S.getContactPlane(parent_box, new_parent_face)) \
            (p * S.getContactPlane(child_box, new_child_face)),
        )
        x,y,ang = pc.x, pc.y, pc.angle
        contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang #= And no slack term. =#))
        p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)

        c = get_cloud_func(p)
    end

    (x,y,ang)
end
    

export get_transform_between_two_registered_clouds, icp, icp_object_pose


PyCall.py"""
import cv2 as cv
import numpy as np
def registerModelToScene(c1, c2):
    icp = cv.ppf_match_3d_ICP(6)
    return icp.registerModelToScene(np.array(c1).transpose().astype(np.float32),
                             np.array(c2).transpose().astype(np.float32))
def registerModelToSceneWithPoses(c1, c2, poses):
    icp = cv.ppf_match_3d_ICP(100)
    return icp.registerModelToScene(np.array(c1).transpose().astype(np.float32),
                             np.array(c2).transpose().astype(np.float32),
                             poses)
def pose_R_t_to_cv2_pose(R,t):
    p = cv.ppf_match_3d_Pose3D()
    p.updatePose(np.array(R), np.array(t))
    return p
"""
function cv2_pose_to_pose(pose::PyCall.PyObject)
   mat = pose.pose
   Pose(mat[:,4], RotMatrix3{Float64}(mat[1:3,1:3]))
end
function cv2_pose_to_pose(pose::Matrix)
   mat = pose
   Pose(mat[:,4], RotMatrix3{Float64}(mat[1:3,1:3]))
end
function pose_to_cv2_pose(pose)
    PyCall.py"pose_R_t_to_cv2_pose"(pose.orientation, pose.pos)
end

export icp_tree, icp_object_poses
