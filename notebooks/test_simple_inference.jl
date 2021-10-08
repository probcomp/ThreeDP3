import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import GenSceneGraphs as S
import PoseComposition: Pose
using Gen
import MeshCatViz

MeshCatViz.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

IDX =1200
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
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
img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
# -

renderer_color = GL.setup_renderer(camera, GL.RGBMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer_color, v, n, f)
end
# colors = I.distinguishable_colors(length(ids), I.colorant"green")
colors = [I.colorant"red", I.colorant"green", I.colorant"cyan"]
rgb_image, depth_image = GL.gl_render(
    renderer_color, ids, gt_poses, colors[1:length(ids)], T.IDENTITY_POSE)
depth_image[depth_image .== 50000.0] .= 200.0
img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

renderer_texture = GL.setup_renderer(camera, GL.TextureMode())
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

rgb_image, depth_image = GL.gl_render(
    renderer_texture, ids, gt_poses, T.IDENTITY_POSE)
depth_image[depth_image .== 50000.0] .= 200.0
img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# +
function get_cloud_from_poses_and_idx(poses, idx, camera_pose)
    @assert length(poses) == length(ids) + 1
    depth_image = GL.gl_render(renderer, ids, poses[1:end-1], camera_pose)
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_image, camera))
    cloud = cloud[:,cloud[3,:] .< (camera.far - 1.0)]
    if size(cloud)[2] == 0
        cloud = zeros(3,1)
    end
    cloud
end

hypers = T.Hyperparams(slack_dir_conc=300.0, slack_offset_var=0.5, p_outlier=0.01, 
    noise=0.2, resolution=0.5, parent_face_mixture_prob=0.99, floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))

params = T.SceneModelParameters(
    boxes=vcat([id_to_box[id] for id in ids],[S.Box(40.0,40.0,0.1)]),
    get_cloud_from_poses_and_idx=get_cloud_from_poses_and_idx,
    camera_pose=cam_pose,
    hyperparams=hypers, N=1)
# -

table_pose = Pose(0.0, 0.0, -0.1)
num_obj = length(params.boxes)
g = T.graph_with_edges(num_obj, [])
constraints = Gen.choicemap((T.structure_addr()) => g, T.floating_pose_addr(num_obj) => table_pose)
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
end

obs_cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(gt_depth_image, camera))
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 1.2 .< obs_cloud[3, :] .< 40.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
constraints[T.obs_addr()] = T.voxelize(obs_cloud, params.hyperparams.resolution)
MeshCatViz.reset_visualizer()
MeshCatViz.viz(constraints[T.obs_addr()])

trace, _ = generate(T.scene, (params,), constraints);

MeshCatViz.reset_visualizer()
MeshCatViz.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)


