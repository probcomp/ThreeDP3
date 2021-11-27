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


import Plots as P
import Measures

maximum(gt_depth_image)

P.heatmap(clamp.(gt_depth_image, 0.0,120.0), yflip=true, aspect_ratio=:equal,colorbar=false, xaxis=false, yaxis=false,xticks=false, yticks=false)
# P.savefig("depth.png")


size(gt_rgb_image)

img = I.colorview(I.Gray, gt_depth_image ./ maximum(gt_depth_image))


# +
resolution = 0.5

renderer_color = GL.setup_renderer(camera, GL.RGBMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer_color, v, n, f)
end
# -

colors = [
    I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
    I.colorant"red", I.colorant"purple", I.colorant"orange"
]

renderer_color.gl_instance.lightpos = [0.0, 0.0, -30.0]
images = [
    let
    rot = R.RotY(ang) * R.RotX(pi/2)
    gap = 15.0
    poses = [Pose([gap*i - 3.5*gap, 60.0, 0.0], rot) for (i,_) in enumerate(ids)]
    rgb_image, depth_image = GL.gl_render(
        renderer_color, ids, poses, colors[1:length(ids)], Pose([0.0,0.0,-160.0],R.RotX(-pi/10)))
    I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
    end
    for ang in 0:0.1:(8*pi)
]

ang = 0.0
gap = 14.0
rot = R.RotY(ang) * R.RotX(pi/2)
p = popat!(poses, 4)
insert!(poses,2,p)
p = popat!(ids, 4)
insert!(ids,2,p)
p = popat!(colors, 4)
insert!(colors,2,p)
poses = [Pose([gap*i - 3.0*gap, 60.0, 0.0], rot) for (i,_) in enumerate(ids)]

# +

poses[5] = Pose(poses[5].pos, R.RotZ(pi/2))
poses[2] = Pose(poses[2].pos, R.RotZ(0.0))


rgb_image, depth_image = GL.gl_render(
    renderer_color, ids, poses, colors[1:length(ids)], Pose([0.0,0.0,-160.0],R.RotX(-pi/10)))
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# -

images[1]

I.colorview(I.RGB, permutedims(Float64.(gt_rgb_image)./255.0, (3,1,2)))



# +
resolution = 0.5

renderer = GL.setup_renderer(camera, GL.RGBMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, n, f)
end


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



