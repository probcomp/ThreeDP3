import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import PoseComposition: Pose
import Serialization
import NearestNeighbors as NN
import Statistics: mean
using Gen
import MeshCatViz

import MeshCatViz
MeshCatViz.setup_visualizer()

function compute_ADDS_error(xyz_models, gt_ids, gt_poses, pred_ids, pred_poses)
    add_s = []
    for (id,gt_pose) in zip(gt_ids, gt_poses)
        idx = findfirst(pred_ids .== id)
        if isnothing(idx)
            push!(add_s, 1.0)
            continue
        end
        pred_pose = pred_poses[idx]

        model = xyz_models[id]

        tree = NN.KDTree(T.move_points_to_frame_b(model, pred_pose))
        _, dists = NN.nn(tree, T.move_points_to_frame_b(model, gt_pose))

        push!(add_s, min(mean(dists)))
    end
    add_s
end

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 1.0
xyz_models = T.load_ycbv_point_xyz(YCB_DIR);
all_ids = collect(1:21);

RESULTS_DIR = "/home/nishadg/mcs/ThreeDVision.jl/pre_neurips_scripts/pybullet_inference/full_massive_run_1/"
TS = collect(0:499);

# +
errors_dense_fusion = Dict([id => [] for id in all_ids])
errors_robust_6d = Dict([id => [] for id in all_ids])
errors_3dp3_star = Dict([id => [] for id in all_ids])
errors_3dp3 = Dict([id => [] for id in all_ids])


for t in TS
    metadata = Serialization.deserialize(joinpath(RESULTS_DIR,"data",lpad(t,6,"0")*"-meta.data"))
    cam_pose = metadata.camera_pose
    ids = metadata.ids
    gt_poses = metadata.object_poses

    dense_poses, dense_ids = T.load_synthetic_dense_fusion_predictions(RESULTS_DIR, t);
    dense_poses = [T.get_c_relative_to_a(cam_pose,p) for p in dense_poses]

    objnet_poses, objnet_ids = T.load_synthetic_objposenet_fusion_predictions(RESULTS_DIR, t)
    objnet_poses = [T.get_c_relative_to_a(cam_pose,p) for p in objnet_poses]

    world_scaling_factor = 100.0
    inf_poses_no_sg, inf_poses = Serialization.deserialize(joinpath(RESULTS_DIR,"sg_results/$(lpad(t,4,"0")).poses"));
    inf_poses = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses]
    inf_poses_no_sg = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses_no_sg]

    df = compute_ADDS_error(xyz_models, ids, gt_poses, dense_ids, dense_poses)
    r6d = compute_ADDS_error(xyz_models, ids, gt_poses, objnet_ids, objnet_poses)
    ablation = compute_ADDS_error(xyz_models, ids, gt_poses, ids, inf_poses_no_sg)
    full = compute_ADDS_error(xyz_models, ids, gt_poses, ids, inf_poses)
    
    for (i,id) in enumerate(ids)
        push!(errors_dense_fusion[id], df[i])
        push!(errors_robust_6d[id], r6d[i])
        push!(errors_3dp3_star[id], ablation[i])
        push!(errors_3dp3[id], full[i])
    end
end

# +
ids = [1,2,3,4,9]

id = 1
df= errors_dense_fusion[id]
r6d= errors_robust_6d[id]
ablation= errors_3dp3_star[id]
full= errors_3dp3[id]

max_plot_val = 0.03
valid_idxs = (df .< max_plot_val) .&  (ablation .< max_plot_val)  .&  (full .< max_plot_val) .&  (full .< max_plot_val);

Plots.scatter(df[valid_idxs], full[valid_idxs],alpha=0.4)
Plots.scatter!(r6d[valid_idxs], full[valid_idxs],alpha=0.4)
Plots.plot!([0.0, max_plot_val],[0.0, max_plot_val],legend=false,
    xlabel="Dense Fusion ADD-S Error",
    ylabel="3DP3 ADD-S Error")
# -

Plots.scatter(r6d[valid_idxs], full[valid_idxs])
Plots.plot!([0.0, max_plot_val],[0.0, max_plot_val],legend=false,
    xlabel="Dense Fusion ADD-S Error",
    ylabel="3DP3 ADD-S Error")



Plots.scatter(df[valid_idxs], full[valid_idxs])
Plots.plot!([0.0, max_plot_val],[0.0, max_plot_val],legend=false,
    xlabel="Dense Fusion ADD-S Error",
    ylabel="3DP3 ADD-S Error")

Plots.scatter(df[valid_idxs], ablation[valid_idxs])
Plots.plot!([0.0, max_plot_val],[0.0, max_plot_val],legend=false,
    xlabel="Dense Fusion ADD-S Error",
    ylabel="3DP3 Ablation ADD-S Error")

Plots.scatter(ablation[valid_idxs], full[valid_idxs])
Plots.plot!([0.0, max_plot_val],[0.0, max_plot_val],legend=false,
    xlabel="3DP3 Ablation ADD-S Error",
    ylabel="3DP3 ADD-S Error")





# +
import Serialization: deserialize


function compute_ADDS_error(xyz_models, gt_ids, gt_poses, pred_ids, pred_poses)
    add_s = []
    for (id,gt_pose) in zip(gt_ids, gt_poses)
        idx = findfirst(pred_ids .== id)
        if isnothing(idx)
            push!(add_s, 1.0)
            continue
        end
        pred_pose = pred_poses[idx]

        model = xyz_models[id]

        tree = KDTree(move_points_to_frame_b(model, pred_pose))
        _, dists = nn(tree, move_points_to_frame_b(model, gt_pose))

        push!(add_s, min(mean(dists)))
    end
    add_s
end

function load_synthetic_results(RESULTS_DIR, T; world_scaling_factor=100.0)
    metadata = Serialization.deserialize(joinpath(RESULTS_DIR,"data",lpad(T,6,"0")*"-meta.data"))

    cam_pose = metadata.camera_pose
    ids = metadata.ids
    gt_poses = metadata.object_poses

    dense_poses, dense_ids = T.load_synthetic_dense_fusion_predictions(RESULTS_DIR, T)
    dense_poses = [Pose(p.pos, p.orientation) for p in dense_poses]
    dense_poses = [T.get_c_relative_to_a(cam_pose,p) for p in dense_poses]

    inf_poses_no_sg, inf_poses = Serialization.deserialize(joinpath(RESULTS_DIR,"sg_results/$(lpad(T,4,"0")).poses"));
    inf_poses = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses]
    inf_poses_no_sg = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses_no_sg]

    ids, gt_poses, dense_ids, dense_poses, inf_poses, inf_poses_no_sg
end

function evaluate_scene(RESULTS_DIR, xyz_models, T; world_scaling_factor=100.0)
    metadata = Serialization.deserialize(joinpath(RESULTS_DIR,"data",lpad(T,6,"0")*"-meta.data"))

    cam_pose = metadata.camera_pose
    ids = metadata.ids
    gt_poses = metadata.object_poses

    dense_poses, dense_ids = load_synthetic_dense_fusion_predictions(RESULTS_DIR, T)
    dense_poses = [Pose(p.pos, p.orientation) for p in dense_poses]
    dense_poses = [T.get_c_relative_to_a(cam_pose,p) for p in dense_poses]

    objnet_poses, objnet_ids = load_synthetic_objposenet_fusion_predictions(RESULTS_DIR, T)
    objnet_poses = [Pose(p.pos, p.orientation) for p in objnet_poses]
    objnet_poses = [T.get_c_relative_to_a(cam_pose,p) for p in objnet_poses]


    inf_poses_no_sg, inf_poses = deserialize(joinpath(RESULTS_DIR,"sg_results/$(lpad(T,4,"0")).poses"));
    inf_poses = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses]
    inf_poses_no_sg = [Pose(p.pos ./ world_scaling_factor, p.orientation) for p in inf_poses_no_sg]

    dense_add = compute_ADDS_error(xyz_models, ids, gt_poses, dense_ids, dense_poses);
    objnet_add = compute_ADDS_error(xyz_models, ids, gt_poses, objnet_ids, objnet_poses);
    my_wo_refinement_add = compute_ADDS_error(xyz_models, ids, gt_poses, ids, inf_poses_no_sg);
    my_add = compute_ADDS_error(xyz_models, ids, gt_poses, ids, inf_poses);

    ids, objnet_add, dense_add, my_wo_refinement_add, my_add
end


# +
YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
original_id_to_cloud = T.load_ycbv_models(YCB_DIR);
xyz_models = T.load_ycbv_point_xyz(YCB_DIR);
model_list = readlines(joinpath(YCB_DIR, "model_list.txt"))

# # +
RESULTS_DIRS = [
    "/home/nishadg/mcs/ThreeDVision.jl/scripts/pybullet_inference/full_massive_run_1/",
    "/home/nishadg/mcs/ThreeDVision.jl/scripts/pybullet_inference/full_massive_run_2/",    
]
TS_LIST = [collect(0:499),collect(500:999)]
# -




