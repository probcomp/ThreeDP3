import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import NearestNeighbors as NN
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
using Gen
import Rotations as R
import MeshCatViz
import DataStructures as DS

import MeshCatViz
MeshCatViz.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

IDX =400
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
camera = T.scale_down_camera(original_camera, 6)
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

camera

get_cloud_func(id,p) = T.move_points_to_frame_b(T.voxelize(get_cloud_from_ids_and_poses(
                                [id], [p], cam_pose),0.5), cam_pose)

blobs = T.get_entities_from_assignment(voxelized_obs_cloud,
    T.dbscan_cluster(voxelized_obs_cloud, radius=0.8))

obs_blob = blobs[2]
MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob)

# +
parent_box = table_box
parent_pose = Pose(0.0, 0.0, -0.5)
parent_face = :top

c1_tree = NN.KDTree(obs_blob)
id_to_face_cloud_and_score =  DS.DefaultDict{Any,Vector{Any}}(()->[])

for id in all_ids
    print(id," ")
    for child_face in S.BOX_SURFACE_IDS
        child_box = id_to_box[id]
        (x,y,ang) = T.icp_project_to_planar_contact(
            obs_blob, p -> get_cloud_func(id,p), parent_pose, parent_box, parent_face,
            child_box, child_face,
            0.0,0.0,0.0;
            c1_tree= c1_tree, outer_iterations=2
        )

        contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
        p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)
        c = get_cloud_func(id,p)
        score = Gen.logpdf(
            T.uniform_mixture_from_template,
            obs_blob, c, 0.01, 0.5*2,(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0)
        )
        
        push!(id_to_face_cloud_and_score[id], (face=child_face, cloud=c, score=score, pose=p, contact_params=(x,y,ang)))
    end
#     println(size(id_to_face_cloud_and_score[id]))
    sort!(id_to_face_cloud_and_score[id],by=x->x.score,rev=true)
end



# +

c = id_to_face_cloud_and_score[14][1].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)

# +
scores = Dict([id => id_to_face_cloud_and_score[id][1].score for id in all_ids]...)
best_score_id = argmax(scores)
@show best_score_id
c = id_to_face_cloud_and_score[best_score_id][1].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)


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
MeshCatViz.viz(obs_blob; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c; color=I.colorant"red", channel_name=:gen_cloud)

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


