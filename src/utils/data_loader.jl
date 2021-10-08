using PyCall
using FileIO
import MAT
import JSON

function load_rgb(filename)
    py"""
    import numpy as np
    import imageio
    def get_depth(depth_path):
        depth = imageio.imread(depth_path)
        depth = depth.astype(np.float)
        return depth
    def get_rgb(rgb_path):
        rgb = imageio.imread(rgb_path)
        return rgb
    """
    py"get_rgb"(filename)
end

function load_depth(filename)
    py"""
    import numpy as np
    import imageio
    def get_depth(depth_path):
        depth = imageio.imread(depth_path)
        depth = depth.astype(np.float)
        return depth
    def get_rgb(rgb_path):
        rgb = imageio.imread(rgb_path)
        return rgb
    """
    py"get_depth"(filename)
end

export load_rgb, load_depth

function load_ycb_model_list(YCB_DIR)
    model_list = readlines(joinpath(YCB_DIR, "model_list.txt"))
end

function load_ycbv_models(YCB_DIR)
    model_list = load_ycb_model_list(YCB_DIR)
    id_to_cloud = Dict()
    for (i,model_name) in enumerate(model_list)
        model_xyz = load(joinpath(YCB_DIR, "models",model_name,"textured_simple.obj"));
        model_xyz = transpose([i[j] for i in model_xyz.position, j in 1:3])
        id_to_cloud[i] = Matrix(model_xyz)
    end
    id_to_cloud
end

function load_ycbv_point_xyz(YCB_DIR)
    model_list = readlines(joinpath(YCB_DIR, "model_list.txt"))
    id_to_cloud = Dict()
    for (i,model_name) in enumerate(model_list)
        model_xyz = readlines(joinpath(YCB_DIR, "models",model_name,"points.xyz"))
        model_xyz = hcat(map(x->map(y->parse(Float64,y), split(x," ")), model_xyz)...)
        id_to_cloud[i] = Matrix(model_xyz)
    end
    id_to_cloud
end

function load_ycb_model_obj_file_paths(YCB_DIR)
    model_list = readlines(joinpath(YCB_DIR, "model_list.txt"))
    paths = []
    for (i,model_name) in enumerate(model_list)
        path = joinpath(YCB_DIR, "models",model_name,"textured_simple.obj")
        push!(paths, path)
    end
    paths
end

function load_ycb_model_texture_file_paths(YCB_DIR)
    model_list = readlines(joinpath(YCB_DIR, "model_list.txt"))
    paths = []
    for (i,model_name) in enumerate(model_list)
        path = joinpath(YCB_DIR, "models",model_name,"texture_map.png")
        push!(paths, path)
    end
    paths
end

function get_ycb_scene_frame_id_from_idx(YCB_DIR, IDX)
    # Keyframes
    keyframes = readlines(joinpath(YCB_DIR, "keyframe.txt"))
    scene_t = map(u->map(x->parse(Int,x), split(u,"/")), keyframes);

    SCENE, T = scene_t[IDX];
    SCENE,T
end

function load_ycbv_scene(YCB_DIR, IDX)
    # Keyframes
    keyframes = readlines(joinpath(YCB_DIR, "keyframe.txt"))
    scene_t = map(u->map(x->parse(Int,x), split(u,"/")), keyframes);

    SCENE, T = scene_t[IDX];
    load_ycbv_scene(YCB_DIR, SCENE, T)
end

function load_ycbv_scene(YCB_DIR, SCENE, T)
    # Load relevant files
    py"""
    import numpy as np
    import imageio

    def get_depth(depth_path):
        depth = imageio.imread(depth_path)
        depth = depth.astype(np.float)
        return depth

    def get_rgb(rgb_path):
        rgb = imageio.imread(rgb_path)
        return rgb
    """

    rgb_image = py"get_rgb"(joinpath(YCB_DIR,lpad(SCENE,4,"0"),lpad(T,6,"0")*"-color.png"))
    depth_image = py"get_depth"(joinpath(YCB_DIR,lpad(SCENE,4,"0"),lpad(T,6,"0")*"-depth.png"));

    mat = MAT.matopen(joinpath(YCB_DIR,lpad(SCENE,4,"0"),lpad(T,6,"0")*"-meta.mat"));
    keys(mat)

    Rt_mat_to_pose(mat) = Pose(mat[1:3,end],R.RotMatrix3{Float64}(mat[1:3,1:3]));


    cam_Rt = read(mat,"rotation_translation_matrix")
    cam_pose = inverse_pose(Rt_mat_to_pose(cam_Rt))

    depth_scaling_factor = read(mat,"factor_depth")
    # @show depth_scaling_factor

    K = read(mat,"intrinsic_matrix")
    camera = GL.CameraIntrinsics(
        width=640, height=480,
        fx=K[1,1], fy=K[2,2], cx=K[1,3], cy=K[2,3],
        near=1.0, far=50000.0)

    ids = map(Int,[
        read(mat,"cls_indexes")
    ][1][:])

    poses = read(mat,"poses")
    poses = [Rt_mat_to_pose(poses[:,:,i]) for i in 1:(size(poses)[3])];

    poses, ids, rgb_image, depth_image ./ depth_scaling_factor, cam_pose, camera
end

function load_ycbv_dense_fusion_predictions(YCB_DIR, IDX)
    # Load dense fusion results
    json_file = joinpath(YCB_DIR,"densefusion/dense_results",lpad(IDX-1,6,"0")*".json")
    json_data = JSON.parsefile(json_file)
    densefusion_ids = [Int.(x[2]) for x in json_data["rois"]]

    mat_file = joinpath(YCB_DIR,"densefusion/dense_results/Densefusion_iterative_result",lpad(IDX-1,4,"0")*".mat")
    mat_data = MAT.matopen(mat_file)
    densefusion_poses = [
        let
            pos = p[5:7]
            rot = p[1:4]
            if all(pos .== 0.0) && all(rot .== 0.0)
                Pose(zeros(3), IDENTITY_ORIENTATION)
            else
                Pose(p[5:7], Rotations.UnitQuaternion(p[1:4]...))
            end
        end
        for p in eachrow(read(mat_data,"poses"))
    ];
    densefusion_poses, densefusion_ids
end

function load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor)
    original_id_to_cloud = load_ycbv_models(YCB_DIR);

    id_to_cloud = Dict()
    id_to_shift = Dict()
    # id_to_mesh = Dict()
    id_to_box = Dict()
    for id in keys(original_id_to_cloud)
        scaled_cloud = copy(original_id_to_cloud[id]) * world_scaling_factor
        mins,maxs = min_max(scaled_cloud)
        shift = ((maxs .+ mins) ./ 2 )
        id_to_shift[id] = shift
        scaled_cloud .-= shift
        id_to_cloud[id] = scaled_cloud
        # id_to_mesh[id] = voxelized_cloud_to_mesh(
            # voxelize(id_to_cloud[id], resolution), resolution)
    #     register_mesh!(renderer, id_to_mesh[id])
        mins,maxs = min_max(id_to_cloud[id])
        id_to_box[id] = S.Box((maxs .- mins)...)
    end
    id_to_cloud, id_to_shift, id_to_box
end

function load_ycbv_scene_adjusted(YCB_DIR, IDX, world_scaling_factor, id_to_shift)
    gt_poses, ids, rgb_image, depth_image, cam_pose, camera = load_ycbv_scene(YCB_DIR, IDX)
    gt_poses = [get_c_relative_to_a(
        Pose(p.pos * world_scaling_factor, p.orientation),
        Pose(id_to_shift[id]...))
    for (id,p) in zip(ids,gt_poses)]
    cam_pose = Pose(cam_pose.pos * world_scaling_factor, cam_pose.orientation)
    depth_image = depth_image * world_scaling_factor;
    return gt_poses, ids, rgb_image, depth_image, cam_pose, camera
end

function load_ycbv_scene_adjusted(YCB_DIR, SCENE, T, world_scaling_factor, id_to_shift)
    gt_poses, ids, rgb_image, depth_image, cam_pose, camera = load_ycbv_scene(YCB_DIR, SCENE, T)
    gt_poses = [get_c_relative_to_a(
        Pose(p.pos * world_scaling_factor, p.orientation),
        Pose(id_to_shift[id]...))
    for (id,p) in zip(ids,gt_poses)]
    cam_pose = Pose(cam_pose.pos * world_scaling_factor, cam_pose.orientation)
    depth_image = depth_image * world_scaling_factor;
    return gt_poses, ids, rgb_image, depth_image, cam_pose, camera
end

function load_synthetic_scene_adjusted(SYNTHETIC_DIR, T, world_scaling_factor, id_to_shift)
    depth_image = load_depth(joinpath(SYNTHETIC_DIR,"data",lpad(T,6,"0")*"-depth.png")) ./ 10000.0;
    rgb_image = load(joinpath(SYNTHETIC_DIR,"data",lpad(T,6,"0")*"-color.png"))
    metadata = deserialize(joinpath(SYNTHETIC_DIR,"data",lpad(T,6,"0")*"-meta.data"))
    camera = metadata.camera
    cam_pose = metadata.camera_pose
    ids = metadata.ids
    gt_poses = metadata.object_poses
    gt_poses = [get_c_relative_to_a(
        Pose(p.pos * world_scaling_factor, p.orientation),
        Pose(id_to_shift[id]...))
    for (id,p) in zip(ids,gt_poses)]
    cam_pose = Pose(cam_pose.pos * world_scaling_factor, cam_pose.orientation)
    depth_image = depth_image * world_scaling_factor;
    return gt_poses, ids, rgb_image, depth_image, cam_pose, camera
end


function load_ycbv_dense_fusion_predictions_adjusted(YCB_DIR, IDX, world_scaling_factor, id_to_shift)
    dense_poses, dense_ids = load_ycbv_dense_fusion_predictions(YCB_DIR, IDX);
    dense_poses = [get_c_relative_to_a(
        Pose(p.pos * world_scaling_factor, p.orientation),
        Pose(id_to_shift[id]...))
    for (id,p) in zip(dense_ids,dense_poses)]
    return dense_poses, dense_ids
end

export load_ycb_model_list, load_ycbv_scene_adjusted, load_ycbv_dense_fusion_predictions_adjusted, load_ycbv_models_adjusted

"""
This is the directory structure we assume for the above code:

├── ycbv_2
│   ├── 0048
│   ├── 0049
│   ├── 0050
│   ├── 0051
│   ├── 0052
│   ├── 0053
│   ├── 0054
│   ├── 0055
│   ├── 0056
│   ├── 0057
│   ├── 0058
│   ├── 0059
│   ├── densefusion
│   │    ├── dense_results
│   │    └── results_PoseCNN_RSS2018
│   ├── keyframe.txt
│   ├── model_list.txt
│   └── models
"""

function load_synthetic_dense_fusion_predictions(SYTH_DIR, T)
    # Load dense fusion results
    json_file = joinpath(SYTH_DIR,"dense_results",lpad(T,6,"0")*".json")
    json_data = JSON.parsefile(json_file)
    densefusion_ids = [Int.(x[2]) for x in json_data["rois"]]

    mat_file = joinpath(SYTH_DIR,"dense_results/Densefusion_iterative_result",lpad(T,4,"0")*".mat")
    mat_data = MAT.matopen(mat_file)
    densefusion_poses = [
        Pose(p[5:7], Rotations.UnitQuaternion(p[1:4]...))
        for p in eachrow(read(mat_data,"poses"))
    ];
    densefusion_poses, densefusion_ids
end

function load_synthetic_objposenet_fusion_predictions(SYTH_DIR, T)
    # Load dense fusion results
    json_file = joinpath(SYTH_DIR,"dense_results",lpad(T,6,"0")*".json")
    json_data = JSON.parsefile(json_file)
    densefusion_ids = [Int.(x[2]) for x in json_data["rois"]]

    mat_file = joinpath(SYTH_DIR,"objposenet_results",lpad(T,4,"0")*".mat")
    mat_data = MAT.matopen(mat_file)
    densefusion_poses = [
        Pose(p[5:7], Rotations.UnitQuaternion(p[1:4]...))
        for p in eachrow(read(mat_data,"poses"))
    ];
    densefusion_poses, densefusion_ids
end

export load_bop_scene, load_bop_object_ply, load_rgb, load_depth, load_ycbv_models, load_ycbv_scene, load_ycbv_dense_fusion_predictions, load_ycbv_point_xyz, load_ycb_model_obj_file_paths,
            load_synthetic_dense_fusion_predictions, load_synthetic_objposenet_fusion_predictions, get_ycb_scene_frame_id_from_idx, load_synthetic_scene_adjusted
