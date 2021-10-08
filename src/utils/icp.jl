import Clustering, NearestNeighbors, LinearAlgebra
import Statistics
import StaticArrays

function get_transform_between_two_registered_clouds(c1, c2)
    centroid_1 = mean(c1, dims=2)
    centroid_2 = mean(c2, dims=2)
    centered_1 = c1 .- centroid_1
    centered_2 = c2 .- centroid_2
    H = centered_1 * centered_2'
    s = svd(H)
    R = (s.V * permutedims(s.U))
    if det(R) < 0
        s = svd(R)
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
function icp(c1, c2; iterations=10)
    T = IDENTITY_POSE
    tree = KDTree(c1)
    for _ in 1:iterations
        idxs, dists = nn(tree, c2)
        neighbors = c1[:,idxs]
        new_T = get_transform_between_two_registered_clouds(c2, neighbors)
        c2 = new_T.pos .+ (new_T.orientation * c2)
        T = get_c_relative_to_a(new_T, T)
    end

    T
end

function icp_object_pose(current_pose, current_obs, get_cloud_func, rand_rotation)
    p = current_pose
    if rand_rotation
        p = Pose(p.pos, uniform_rot3())
    end
    c = get_cloud_func(p)
    for _ in 1:5
        T = icp(current_obs, c; iterations=10)
        p = get_c_relative_to_a(T,p)
        c = get_cloud_func(p)
    end

    p
end

export get_transform_between_two_registered_clouds, icp, icp_object_pose


function icp_tree(c1, c1_tree, c2; iterations=10)
    T = IDENTITY_POSE
    for _ in 1:iterations
        idxs, dists = nn(c1_tree, c2)
        neighbors = c1[:,idxs]
        new_T = get_transform_between_two_registered_clouds(c2, neighbors)
        c2 = new_T.pos .+ (new_T.orientation * c2)
        T = get_c_relative_to_a(new_T, T)
    end
    T
end

function icp_object_poses(start_poses, current_obs, get_cloud_func; iter_top=3, iter_bot=4)
    current_obs_tree = KDTree(current_obs)
    final_poses = Array{Pose, 1}(UndefInitializer(), length(start_poses))
    for (i,p) in enumerate(start_poses)
        c = get_cloud_func(p)
        for _ in 1:iter_top
            T = icp_tree(current_obs, current_obs_tree,c; iterations=iter_bot)
            p = get_c_relative_to_a(T,p)
            c = get_cloud_func(p)
        end
        final_poses[i] = p
    end
    final_poses
end

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
