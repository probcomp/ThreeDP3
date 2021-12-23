# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import NearestNeighbors as NN
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
obs_blob = blobs[4]
MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./ 100.0)

# +
parent_box = S.Box(40.0,40.0,0.1)
parent_pose = Pose(0.0, 0.0, 0.0)
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

# -


MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)


# +
scores = [id_to_face_cloud_and_score[id][1].score for id in all_ids]
perm = sortperm(-scores)
best_score_id = perm[1]
@show best_score_id
c = id_to_face_cloud_and_score[best_score_id][1].cloud

MeshCatViz.reset_visualizer()
MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(c ./100.0; color=I.colorant"red", channel_name=:gen_cloud)
# -

for id in all_ids
    c = id_to_face_cloud_and_score[id][1].cloud
    MeshCatViz.reset_visualizer()
    MeshCatViz.viz(obs_blob ./100.0; color=I.colorant"black", channel_name=:obs_cloud)
    MeshCatViz.viz(c ./100.0; color=I.colorant"red", channel_name=:gen_cloud)
    sleep(0.1)
end

# +
c = id_to_face_cloud_and_score[6][1].cloud

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

@show ids

img = I.colorview(I.RGB, permutedims(Float64.(rgb_image)./255.0, (3,1,2)))

# +
parent_box = S.Box(40.0,40.0,0.1)
parent_pose = Pose(0.0, 0.0, 0.0)
parent_face = :top
id = 6
child_box = id_to_box[id]


x = 17.0
y = -6.0
ang = Gen.uniform(0.0, 2*π)
child_face = :top

contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)

renderer_texture.gl_instance.lightpos = cam_pose.pos
@time rgb_image, depth_image = GL.gl_render(
    renderer_texture, [id], [p], cam_pose)
overlay_img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
mask = (overlay_img .!= overlay_img[1,1])
new_img = copy(img)
alpha = 0.1
new_img[mask] .= alpha*new_img[mask] .+ (1-alpha).*overlay_img[mask]
new_img

# +
parent_box = S.Box(40.0,40.0,0.1)
parent_pose = Pose(0.0, 0.0, 0.0)
parent_face = :top
id = 6
child_box = id_to_box[id]



param_sweep = [
    (x,y,0.0)
    for x in -20.0:2.5:13.0 for y in -10.0:2.5:10.0
]
child_face = :top
@show size(param_sweep)
imgs = [
    let  
        contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
        p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)

        renderer_texture.gl_instance.lightpos = cam_pose.pos
        rgb_image, depth_image = GL.gl_render(
            renderer_texture, [id], [p], cam_pose)
        overlay_img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
        mask = (overlay_img .!= overlay_img[1,1])
        new_img = copy(img)
        alpha = 0.1
        new_img[mask] .= alpha*new_img[mask] .+ (1-alpha).*overlay_img[mask]
        new_img
    end
    for (x,y,ang) in param_sweep
];
# -

import FileIO
gif = cat(imgs...;dims=3);
FileIO.save("test.gif", gif)

# +
new_img = copy(img)
function highlight_bounding_box(img_in, is,js, w, h)
    img = copy(img_in)
    col = I.colorant"red"
    border_width = 5
    for i in is:is+border_width
        for j in js:js+border_width
        
        img[i,j:j+w] .= col
        img[i+h,j:j+w] .= col
        img[i:i+h,j] .= col
        img[i:i+h,j+w] .= col
        end
    end
    img 
end



param_sweep = [
    [
    highlight_bounding_box(new_img, x,y,z,z)
    for x in 1:50:(480-z-5) for y in 1:50:(640-z-5)
    ]
    for z in 100:50:200
];
param_sweep = vcat(param_sweep...)

@show size(param_sweep)
# -

import FileIO
gif = cat(param_sweep...;dims=3);
FileIO.save("test2.gif", gif)

Gen.categorical(ones(6)./6.0)

imgs = [
    let 
        x = Gen.uniform(-20.0,13.0)
        y = Gen.uniform(-10.0,10.0)
        ang = Gen.uniform(0,2*pi)
        child_face = S.BOX_SURFACE_IDS[Gen.categorical(ones(6)./6.0)]
            
        contact = S.ShapeContact(parent_face, Real[], child_face, Real[], S.PlanarContact(x,y,ang))
        p = parent_pose * S.getRelativePoseFromContact(parent_box, child_box, contact)

        renderer_texture.gl_instance.lightpos = cam_pose.pos
        rgb_image, depth_image = GL.gl_render(
            renderer_texture, [id], [p], cam_pose)
        overlay_img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
        mask = (overlay_img .!= overlay_img[1,1])
        new_img = copy(img)
        alpha = 0.1
        new_img[mask] .= alpha*new_img[mask] .+ (1-alpha).*overlay_img[mask]
        new_img
    end
    for _ in 1:50
];

import FileIO
gif = cat(imgs...;dims=3);
FileIO.save("test2.gif", gif)

imgs = [
    let 
        p = T.uniformPose(-30.0, 33.0, -20.0,20.0, -10.0, 20.0)
        renderer_texture.gl_instance.lightpos = cam_pose.pos
        rgb_image, depth_image = GL.gl_render(
            renderer_texture, [id], [p], cam_pose)
        overlay_img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
        mask = (overlay_img .!= overlay_img[1,1])
        new_img = copy(img)
        alpha = 0.1
        new_img[mask] .= alpha*new_img[mask] .+ (1-alpha).*overlay_img[mask]
        new_img        
    end
    for _ in 1:50
];

import FileIO
gif = cat(imgs...;dims=3);
FileIO.save("test2.gif", gif)


