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
# IDX =6
IDX =301
# IDX =629
IDX = 539
IDX = 1228
IDX = 6
IDX = 913
IDX = 1898
IDX = 539


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


# +
resolution = 0.5

renderer_color = GL.setup_renderer(camera, GL.RGBMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,_ = renderer_color.gl_instance.load_obj_parameters(obj_paths[id])
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    GL.load_object!(renderer_color, v, n, f)
end
# colors = I.distinguishable_colors(length(ids), I.colorant"green")
colors = [
    I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
    I.colorant"red", I.colorant"purple", I.colorant"orange"
]


# +
focus_idx = 2

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
alpha = 0.4
mask = (rgb_image_3dp3 .!= I.RGBA(1.0, 1.0, 1.0, 1.0))
new_gt_rgb_image[mask] .= alpha.*new_gt_rgb_image[mask] +(1-alpha).*rgb_image_3dp3[mask]
new_gt_rgb_image
# -


# +

@gen function slam_model(T, room_bounds, wall_colors, cov)
    prev_data = nothing
    for t in 1:T
        if t==1
            pose ~ pose_uniform(room_bounds[1,:],room_bounds[2,:])
        else
            pose ~ pose_gaussian(prev_pose, [1.0 0.0;0.0 1.0] * 0.1, deg2rad(20.0))
        end
        prev_pose = pose
    end
    
    rgb, depth = GL.gl_render(renderer, wall_colors, pose)
    sense_depth ~ mvnormal(depth[1,:], cov)
    sense_rgb ~ color_distribution(rgb, 0.999)
end



# +

    
corners = get_corners(get_depth(tr_gt,t))
likely_poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(likely_poses, p)
    end
end

for _ in 1:10
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (pose_mixture_proposal, 
            (likely_poses, t, [1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 10);
end


# +

@gen function generative_model()
    floor = Plane([0., 0., 0.], [0., 0., 1.])
    room_height = @trace(uniform(2.5, 3.0), :room_height)
    ceiling = Plane([0., 0., room_height], [0., 0., -1.])
    objects = [floor, ceiling]

    camera_z = @trace(uniform(0.2, 2.0), :z)
    camera_location = [0., 0., camera_z]

    camera_pitch = @trace(uniform(pi/2 - pi/4, pi/2 + pi/4), :pitch)
    camera_roll = @trace(uniform(-pi/4, pi/4), :roll)
    camera_rotation = make_rotation_matrix(camera_pitch, 0., camera_roll)
  
    depths = render(objects, camera_location, camera_rotation)

    noise = 0.1
    @trace(realsense_sensor(depths, noise), :observation)
end



# initialize trace with first observation
frame = get_frame(depth_camera)
constraints = Gen.choicemap((:observation, frame))
trace, = Gen.generate(generative_model, (), constraints)

while true

    trace, = Gen.mh(trace, Gen.select(:pitch, :roll, :z, :room_height))
    trace, = Gen.mh(trace, random_walk, (pi/64, :pitch))
    trace, = Gen.mh(trace, random_walk, (pi/64, :roll))
    trace, = Gen.mh(trace, random_walk, (0.05, :z))
    trace, = Gen.mh(trace, random_walk, (0.05, :room_height))
    trace, = Gen.mh(trace, Gen.select(:noise))

    # update trace with new observation
    frame = get_frame(depth_camera)
    constraints = Gen.choicemap((:observation, frame))
    trace, = Gen.update(trace, (), (), constraints)
end



