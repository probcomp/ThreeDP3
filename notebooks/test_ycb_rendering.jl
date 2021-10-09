import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
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

# +
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
# -


