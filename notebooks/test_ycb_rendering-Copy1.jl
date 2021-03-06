import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose
using Gen
import Serialization
import MeshCatViz


import MeshCatViz
MeshCatViz.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

# +
IDX =6
# IDX =301
IDX =2100
IDX =100

@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)

# +
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
dense_poses, dense_ids = T.load_ycbv_dense_fusion_predictions_adjusted(YCB_DIR, IDX, world_scaling_factor, id_to_shift);
dense_poses_reordered = [dense_poses[findfirst(dense_ids .== id)]   for id in ids];
inf_poses = Serialization.deserialize("../assets/results/$(lpad(IDX,4,"0")).poses").poses_after
inf_poses = T.adjust_poses(inf_poses, ids, world_scaling_factor, id_to_shift)

img = I.colorview(I.Gray, gt_depth_image ./ maximum(gt_depth_image))
I.colorview(I.RGB, permutedims(Float64.(gt_rgb_image)./255.0, (3,1,2)))
# -


I.colorview(I.RGB, permutedims(Float64.(gt_rgb_image)./255.0, (3,1,2)))



# +
resolution = 0.5

renderer = GL.setup_renderer(camera, GL.RGBMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, n, f)
end

colors = [
    I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
    I.colorant"red", I.colorant"purple", I.colorant"orange"
]
# -

ids



new_poses = copy(gt_poses)
new_poses = [T.get_c_relative_to_a(cam_pose,p) for p in new_poses]
old_pose = new_poses[1]
new_poses[1] = Pose(old_pose.pos .- [0,0,2.0],old_pose.orientation)

renderer.gl_instance.lightpos = cam_pose.pos

rgb_image_dense, depth_image = GL.gl_render(
    renderer, ids, new_poses, colors, cam_pose)
rgb_image_dense= I.colorview(I.RGBA, permutedims(rgb_image_dense,(3,1,2)))

import Plots

import Pkg;Pkg.add("Measures")

using Measures

low, high = 56.0, 130.0
d = clamp.(depth_image, low, high)
@show minimum(d), maximum(d)
p = Plots.heatmap(d; c=:thermal,
    clim=(53.0, 130.0),
    ylim=(0, 480),
    xlim=(0, 640),
    aspect_ratio=:equal, yflip=true,  legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false, margin = 0mm)
Plots.savefig(p, "regen.pdf")
p

d = clamp.(gt_depth_image,low, high)
@show minimum(d), maximum(d)
p = Plots.heatmap(d; c=:thermal,
    clim=(53.0, 130.0),
    ylim=(0, 480),
    xlim=(0, 640),
    aspect_ratio=:equal, yflip=true,  legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false, margin = 0mm)
Plots.savefig(p, "real.pdf")
p

# +
focus_idx = 3
alpha = 0.4

rgb_image_dense, depth_image = GL.gl_render(
    renderer_color, [ids[focus_idx]], [dense_poses_reordered[focus_idx]], [I.colorant"cyan"], T.IDENTITY_POSE)
rgb_image_dense= I.colorview(I.RGBA, permutedims(rgb_image_dense,(3,1,2)))

new_gt_rgb_image = cat(Float64.(gt_rgb_image) ./ 255.0, ones(size(gt_rgb_image)[1:2]),dims=3)
new_gt_rgb_image = I.colorview(I.RGBA, permutedims(new_gt_rgb_image,(3,1,2)))
@show size(new_gt_rgb_image)
mask = (rgb_image_dense .!= I.RGBA(1.0, 1.0, 1.0, 1.0))
new_gt_rgb_image[mask] .= alpha.*new_gt_rgb_image[mask] +(1-alpha).*rgb_image_dense[mask]
new_gt_rgb_image
# +


rgb_image_3dp3, depth_image = GL.gl_render(
    renderer_color, [ids[focus_idx]], [inf_poses[focus_idx]], [I.colorant"limegreen"], T.IDENTITY_POSE)
rgb_image_3dp3= I.colorview(I.RGBA, permutedims(rgb_image_3dp3,(3,1,2)))

new_gt_rgb_image = cat(Float64.(gt_rgb_image) ./ 255.0, ones(size(gt_rgb_image)[1:2]),dims=3)
new_gt_rgb_image = I.colorview(I.RGBA, permutedims(new_gt_rgb_image,(3,1,2)))
@show size(new_gt_rgb_image)
@show size(rgb_image_3dp3)
mask = (rgb_image_3dp3 .!= I.RGBA(1.0, 1.0, 1.0, 1.0))
new_gt_rgb_image[mask] .= alpha.*new_gt_rgb_image[mask] +(1-alpha).*rgb_image_3dp3[mask]
new_gt_rgb_image
# -



