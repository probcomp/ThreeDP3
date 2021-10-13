import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose
using Gen
import MeshCatViz

import MeshCatViz
MeshCatViz.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

IDX =2000
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
img = I.colorview(I.Gray, gt_depth_image ./ maximum(gt_depth_image))
I.colorview(I.RGB, permutedims(Float64.(rgb_image)./255.0, (3,1,2)))



# +
resolution = 0.5

renderer_color = GL.setup_renderer(camera, GL.RGBMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer_color, v, n, f)
end
# colors = I.distinguishable_colors(length(ids), I.colorant"green")
colors = [I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen", I.colorant"red", I.colorant"purple", I.colorant"orange"]
# -

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

images[100][200:end-100,:]

i = cat([i[200:end-100,:] for i in images]...,dims=3)
import FileIO
FileIO.save("test.gif",i)

x = 1:3
y = 1:5
x' .* ones(5)


# +
renderer_color.gl_instance.lightpos = [000.0, 0.0, 0.0]

rgb_image, depth_image = GL.gl_render(
    renderer_color, ids, gt_poses, colors[1:length(ids)], T.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))


# -

d = depth_image
valid = d .< camera.far-10.0
max_val = maximum(d[valid])
max_val += 5.0
d = clamp.(d, 0.0, max_val)
img = I.colorview(I.Gray, d ./ (1.0 *maximum(d)))

clamped_gt_depth_image = clamp.(gt_depth_image, 0.0, max_val)
img = I.colorview(I.Gray, clamped_gt_depth_image ./ (maximum(clamped_gt_depth_image)))

renderer_color.gl_instance.lightpos = [0.0, -20.0, -0.0]
rgb_image, depth_image = GL.gl_render(
    renderer_color, ids, gt_poses, colors[1:length(ids)], T.IDENTITY_POSE)
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
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
# -


