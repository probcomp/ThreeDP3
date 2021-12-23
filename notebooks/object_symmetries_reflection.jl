# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import NearestNeighbors as NN
import LinearAlgebra
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
using Gen
import Rotations as R
import DataStructures as DS
import MeshCatViz


import MeshCatViz
MeshCatViz.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

IDX =200
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
camera = T.scale_down_camera(original_camera, 4)
@show camera
img = I.colorview(I.Gray, gt_depth_image ./ maximum(gt_depth_image))
I.colorview(I.RGB, permutedims(Float64.(rgb_image)./255.0, (3,1,2)))


# +
resolution = 0.5

renderer = GL.setup_renderer(camera, GL.DepthMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, f)
end
depth_image = GL.gl_render(
    renderer, ids, gt_poses, T.IDENTITY_POSE)
depth_image[depth_image .== 50000.0] .= 200.0

function get_cloud_from_ids_and_poses(ids, poses, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_image, camera))
    cloud = cloud[:,cloud[3,:] .< (camera.far - 1.0)]
    if size(cloud)[2] == 0
        cloud = zeros(3,1)
    end
    cloud
end

img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
# -

function reflection_matrix_4x4(center, direction)
    T =  Matrix{Float64}(LinearAlgebra.I,4,4)
    T[1:3,4] = -center
    
    dir = direction ./ LinearAlgebra.norm(direction)
    R = [
        1-2*dir[1]^2   -2*dir[2]*dir[1] -2*dir[3]*dir[1] 0;
        -2*dir[2]*dir[1]  1-2*dir[2]^2 -2*dir[2]*dir[3] 0;
        -2*dir[3]*dir[1]  -2*dir[2]*dir[3] 1-2*dir[3]^2 0;
        0 0 0 1
    ]
    inv(T) * R * T
end

# +
IDX = 3
obj_pose = T.get_c_relative_to_a(cam_pose,gt_poses[IDX])
depth_image = GL.gl_render(renderer, [ids[IDX]], [obj_pose], cam_pose)

object_cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_image, camera))
object_cloud = object_cloud[:,object_cloud[3,:] .< (camera.far - 1.0)]
object_cloud = T.move_points_to_frame_b(object_cloud, cam_pose)
mins,maxs = T.min_max(object_cloud)
mins = mins .- 6.5
maxs = maxs .+ 6.5

RESOLUTION = resolution
mins,maxs = floor.(Int,mins ./ RESOLUTION) * RESOLUTION , ceil.(Int, maxs ./RESOLUTION) * RESOLUTION
dimensions = [length(collect(mins[1]:RESOLUTION:maxs[1])),
              length(collect(mins[2]:RESOLUTION:maxs[2])),
              length(collect(mins[3]:RESOLUTION:maxs[3]))]
ALL_VOXELS = hcat([[a,b,c] for a in collect(mins[1]:RESOLUTION:maxs[1])
                           for b in collect(mins[2]:RESOLUTION:maxs[2])
                           for c in collect(mins[3]:RESOLUTION:maxs[3])]...)

occupied, occluded, free = T.get_occ_ocl_free(T.get_points_in_frame_b(ALL_VOXELS,cam_pose), camera, depth_image, resolution)
MeshCatViz.reset_visualizer()
MeshCatViz.viz(ALL_VOXELS[:,occupied]./200.0;channel_name=:occupied,color=I.colorant"red")
MeshCatViz.viz(ALL_VOXELS[:,occluded]./200.0;channel_name=:occluded,color=I.colorant"black")
cloud = ALL_VOXELS[:,occupied]

tree_occupied = NN.KDTree(ALL_VOXELS[:, occupied]);
tree_non_free = NN.KDTree(ALL_VOXELS[:, occupied .| occluded]);
# -


plane_points = hcat([
    [x,y,0.0]
    for x in -10.0:1.0:10.0 for y in -10.0:1.0:10.0
]...)
@show size(plane_points)


function score_reflection_plane(center, direction; verbose=false)
   Reflect = reflection_matrix_4x4(
        center, direction
    )
    new_c = (Reflect * (vcat(cloud, ones(1,size(cloud)[2]))))[1:3,:]
    new_c = T.voxelize(new_c, resolution)
    all_idxs = NN.inrange(tree_non_free, new_c, resolution ./ 2);
    matched = [length(x)>0 for x in all_idxs]
    score = sum(matched) / length(matched)
    if verbose
        println(score)
    end
    
    all_idxs = NN.inrange(tree_non_free, new_c, resolution ./ 2);
    matched = [length(x)==0 for x in all_idxs]
    if verbose
        println(-sum(matched) / length(matched))
    end
    
    score -= sum(matched) / length(matched)
    return score, new_c
end

MeshCatViz.reset_visualizer()
MeshCatViz.viz(cloud./500.0;channel_name=:occupied,color=I.colorant"red")
T.min_max(cloud)


# +
param_sweep = [
    ([x,y,10.0],[sin(θ),cos(θ),0.0])
    for x in -6.0:resolution:-2.0 for y in -1.0:resolution:5.0 for θ in 0:0.05:(2*pi)
];
@show size(param_sweep)

# for (c,d) in param_sweep[1:500]
#     MeshCatViz.viz(get_plane_points(c,d) ./ 500.0;channel_name=:plane, color=I.colorant"green")
#     sleep(0.005)
# end
# -



scores = [score_reflection_plane(c, d)[1] for (c,d) in param_sweep];


c,d = param_sweep[argmax(scores)]
score, new_c = score_reflection_plane(c,d;verbose=true)
@show score
MeshCatViz.reset_visualizer()
MeshCatViz.viz(cloud./200.0;channel_name=:occupied,color=I.colorant"red")
MeshCatViz.viz(new_c./200.0;channel_name=:occluded,color=I.colorant"black")

# +

function get_plane_points(center, direction)
    u = [0,0,1]
    v = direction
    u = u ./ LinearAlgebra.norm(u)
    v = v ./ LinearAlgebra.norm(v)
    half = (u+v) ./ LinearAlgebra.norm(u+v)
    q = R.QuatRotation(LinearAlgebra.dot(u,half),LinearAlgebra.cross(u,half)...)
    q * plane_points .+ center
end

MeshCatViz.viz(get_plane_points(c,d) ./ 200.0;channel_name=:plane, color=I.colorant"green")
# -

perm = sortperm(-1.0 .* scores)
c,d = param_sweep[perm[5]]
score, new_c = score_reflection_plane(c,d;verbose=true)
@show score
MeshCatViz.reset_visualizer()
MeshCatViz.viz(cloud./300.0;channel_name=:occupied,color=I.colorant"red")
MeshCatViz.viz(new_c./100.0;channel_name=:occluded,color=I.colorant"black")

# +
plane_cloud = [
    
    
]
# -

score_reflection_plane(best_grid_center, d)



all_idxs

obs_cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(gt_depth_image, original_camera))
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 1.0 .< obs_cloud[3, :] .< 40.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
voxelized_obs_cloud = T.voxelize(obs_cloud, resolution)
MeshCatViz.viz(voxelized_obs_cloud ./ 100.0)

get_cloud_func(id,p) = T.move_points_to_frame_b(T.voxelize(get_cloud_from_ids_and_poses(
                                [id], [p], cam_pose),resolution), cam_pose)

blobs = T.get_entities_from_assignment(voxelized_obs_cloud,
    T.dbscan_cluster(voxelized_obs_cloud, radius=0.6))

@show ids
obs_blob = blobs[2]
MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./ 100.0)

# +
parent_box = S.Box(40.0,40.0,0.1)
parent_pose = Pose(0.0, 0.0, -1.0)
parent_face = :top

c1_tree = NN.KDTree(obs_blob)
id_to_face_cloud_and_score =  DS.DefaultDict{Any,Vector{Any}}(()->[])

for id in all_ids
    print(id," ")
    for child_face in S.BOX_SURFACE_IDS
        child_box = id_to_box[id]
        (x,y,ang) = T.icp_project_to_planar_contact(
            obs_blob, p -> get_cloud_func(id,p),
            parent_pose, parent_box, parent_face,
            child_box, child_face,
            0.0, 0.0, 0.0;
            c1_tree= c1_tree, outer_iterations=10
        )

        contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
        p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)
        c = get_cloud_func(id,p)
        score = Gen.logpdf(
            T.uniform_mixture_from_template,
            obs_blob, c, 0.001, 0.5*2,(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0)
        )
        
        push!(id_to_face_cloud_and_score[id], (face=child_face, cloud=c, score=score, pose=p, contact_params=(x,y,ang)))
    end
#     println(size(id_to_face_cloud_and_score[id]))
    sort!(id_to_face_cloud_and_score[id],by=x->x.score,rev=true)
end



# +
scores = [id_to_face_cloud_and_score[id][1].score for id in all_ids]
perm = sortperm(-scores)
best_score_id = perm[1]
@show best_score_id
c = id_to_face_cloud_and_score[best_score_id][2].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c ./100.0; color=I.colorant"red", channel_name=:gen_cloud)

# +
c = id_to_face_cloud_and_score[12][1].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c ./100.0; color=I.colorant"red", channel_name=:gen_cloud)

# +

id_and_id_to_face_cloud_and_score =  DS.DefaultDict{Any,Vector{Any}}(()->[])

for id_1 in all_ids
    print(id_1," ")
    
    data = id_to_face_cloud_and_score[id_1][1]
    
    unexplained_idxs = T.get_unexplained_points(obs_blob, data.cloud)
    unexplained_points = obs_blob[:, unexplained_idxs]
    
    c1_tree = NN.KDTree(unexplained_points)
    
    parent_pose = data.pose
    parent_face = T.opposite_face(data.face)
    parent_box = id_to_box[id_1]

    for id in all_ids
        for child_face in S.BOX_SURFACE_IDS
            child_box = id_to_box[id]

            (x,y,ang) = T.icp_project_to_planar_contact(
                unexplained_points, p -> get_cloud_func(id,p), parent_pose, parent_box, parent_face,
                child_box, child_face,
                0.0,0.0,0.0;
                c1_tree= c1_tree
            )

            contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
            p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)
            c = get_cloud_func(id,p)
            score = Gen.logpdf(
                T.uniform_mixture_from_template,
                unexplained_points, c, 0.01, 0.5*2,(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0)
            )

            push!(id_and_id_to_face_cloud_and_score[(id_1,id)], 
                (face=child_face, cloud=c, score=score, pose=p, contact_params=(x,y,ang)))
        end
        sort!(id_and_id_to_face_cloud_and_score[(id_1,id)],by=x->x.score,rev=true)
    end
end

# -

id_and_id_to_score = Dict()
for id_1 in all_ids
    for id in all_ids
        pose_1 = id_to_face_cloud_and_score[id_1][1].pose
        pose_2 = id_and_id_to_face_cloud_and_score[(id_1,id)][1].pose
        
        
        c = T.move_points_to_frame_b(T.voxelize(get_cloud_from_ids_and_poses(
                                [id_1,id], [pose_1, pose_2], cam_pose), 0.5), cam_pose)
        
        score = Gen.logpdf(
            T.uniform_mixture_from_template,
            obs_blob, c, 0.01, 0.5*2,(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0)
        )
        id_and_id_to_score[(id_1,id)] = score
    end
end

# +
(id_1,id) = argmax(id_and_id_to_score)
@show (id_1,id)
pose_1 = id_to_face_cloud_and_score[id_1][1].pose
pose_2 = id_and_id_to_face_cloud_and_score[(id_1,id)][1].pose


c = T.move_points_to_frame_b(T.voxelize(get_cloud_from_ids_and_poses(
                        [id_1,id], [pose_1, pose_2], cam_pose),0.5), cam_pose)
MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c ./100.0; color=I.colorant"red", channel_name=:gen_cloud)
# -

c

ids

ids

# +
id_1 = 13
    data = id_to_face_cloud_and_score[id_1][1]
    
    unexplained_idxs = T.get_unexplained_points(obs_blob, data.cloud)
    unexplained_points = obs_blob[:, unexplained_idxs]
    
    c1_tree = NN.KDTree(unexplained_points)
    
    parent_pose = data.pose
    parent_face = T.opposite_face(data.face)
    parent_box = id_to_box[id_1]


id = 3 
child_face=:left
(x,y,ang) = T.icp_project_to_planar_contact(
    unexplained_points, p -> get_cloud_func(id,p), parent_pose, parent_box, parent_face,
    child_box, child_face,
    0.0,0.0,0.0;
    c1_tree= c1_tree, outer_iterations=5, iterations=10
)

contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)
c = get_cloud_func(id,p)


MeshCatViz.reset_visualizer()
MeshCatViz.viz(unexplained_points; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)



# +

c = id_and_id_to_face_cloud_and_score[(13,3)][1].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)
# -

unexplained_idxs = T.get_unexplained_points(obs_blob, data.cloud)
unexplained_points = obs_blob[:, unexplained_idxs]
MeshCatViz.viz(unexplained_points; color=I.colorant"red", channel_name=:gen_cloud)


ids

pair = (13,:right)
@show id_and_face_to_score[pair]
c = id_and_face_to_cloud[pair]
c = id_and_face_to_cloud[best_pair]



MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)



ids

clouds = [[
    let
        p = T.icp_object_pose(IDENTITY_POSE, obs_blob,
                              p-> T.move_points_to_frame_b(T.voxelize(get_cloud_from_ids_and_poses(
                                [id], [p], cam_pose),0.5), cam_pose), true)

        c = T.move_points_to_frame_b(T.voxelize(
                get_cloud_from_ids_and_poses([id], [p], cam_pose), 0.5), cam_pose)
    end 
    for _ in 1:10
] for id in all_ids];
scores = [
    maximum([
        Gen.logpdf(T.uniform_mixture_from_template,obs_blob, c, 0.01, 0.5*2,(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))
        for c in x
    ])
    for x in clouds
]

best_id = argmax(scores)
@show best_id
c = clouds[best_id][1]
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:h1)
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:h2)

ids



ids

c




MeshCatViz.reset_visualizer()
MeshCatViz.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)

traces = [trace];

for i in 1:length(ids)
    if !T.isFloating(T.get_structure(trace),i)
        continue
    end
    pose_addr = T.floating_pose_addr(i)
    for _ in 1:50
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0)
        if acc traces = push!(traces, trace) end

        trace, acc = T.drift_move(trace, pose_addr, 1.0, 100.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 1.5, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 100.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 5000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 1, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 2, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 3, 1000.0)
        if acc traces = push!(traces, trace) end
    end
end


CRAZY_POSE = Pose(-100.0 * ones(3), T.IDENTITY_ORIENTATION)
for i in 1:length(ids)
    if !T.isFloating(T.get_structure(trace),i)
        continue
    end
    addr = T.floating_pose_addr(i)

    t, = Gen.update(trace, Gen.choicemap(T.floating_pose_addr(i) => CRAZY_POSE))
    valid = T.get_unexplained_obs_cloud(t)

    p = T.icp_object_pose(trace[addr], trace[:obs][:,valid],
                          p-> get_cloud_from_ids_and_poses(
                            [ids[i]], [p], cam_pose), false)
    for _ in 1:3
        trace, acc = T.pose_mixture_move(trace, addr, [trace[addr], p], [0.5, 0.5], 0.001, 5000.0)
        if acc traces = push!(traces, trace) end
    end
end

MeshCatViz.reset_visualizer()
MeshCatViz.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)

depth_image = GL.gl_render(
    renderer, ids, T.get_poses(trace)[1:end-1], cam_pose)
depth_image[depth_image .== 50000.0] .= 200.0
img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))

# +
renderer_texture = GL.setup_renderer(original_camera, GL.TextureMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,t = renderer_texture.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer_texture, v, n, f, t,
        texture_paths[id]
    )
end


# -

rgb_image, depth_image = GL.gl_render(
    renderer_texture, ids, T.get_poses(trace)[1:end-1], cam_pose)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))


