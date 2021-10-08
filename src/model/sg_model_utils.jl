# Trace utilities

function get_scene_graph(trace)
    get_retval(trace).scene_graph
end
function structure_addr()
    (:scene_graph => :structure)
end
function obs_addr()
    :obs
end
function get_structure(trace)
    trace[structure_addr()]
end
function get_gen_cloud(trace)
    rendered_clouds = get_retval(trace).rendered_clouds
    length(rendered_clouds) == 1 || error(
        "rendered_clouds does not have length 1." *
        " Did you mean to call get_gen_cloud(trace, i)?")
    only(rendered_clouds)
end
function get_gen_cloud(trace, i)
    get_retval(trace).rendered_clouds[i]
end
function get_obs_cloud(trace)
    trace[:obs]
end
function get_object_boxes(trace)
    get_args(trace)[1].boxes
end
function get_object_meshes(trace)
    get_args(trace)[1].meshes
end
function get_num_objects(trace)
    length(get_object_boxes(trace))
end
function get_cam_pose(trace)::Pose
    get_args(trace)[1].camera_pose
end
function get_poses(trace)
    floatingPosesOf(get_scene_graph(trace))
end
function obj_name_from_idx(idx)
    Symbol("obj_", idx)
end

function get_edges(trace::Gen.Trace)
    collect(LG.edges(get_structure(trace)))
end

export scene, get_scene_graph, structure_addr, obs_addr, get_structure, get_gen_cloud, get_obs_cloud, get_object_names,
        get_object_boxes, get_num_objects, get_poses, get_cam_pose, get_edges, obj_name_from_idx, get_object_meshes


function get_unexplained_obs_cloud(trace, radius=1.0)
    obs = get_obs_cloud(trace)
    obs_tree = NearestNeighbors.KDTree(obs)

    valid = fill(true, size(obs)[2])

    idxs = NearestNeighbors.inrange(obs_tree, get_gen_cloud(trace), radius)
    valid[unique(vcat(idxs...))] .= false
    valid
end

export get_unexplained_obs_cloud


floating_pose_addr(i::Int) = (:scene_graph => :params => i => :floating => :pose)
floating_addr(i::Int) = (:scene_graph => :params => i => :floating)
contact_addr(i::Int) = (:scene_graph => :params => i => :contact)
contact_addr(i::Int, addr::Symbol) = (:scene_graph => :params => i => :contact => addr)

export floating_pose_addr, contact_addr

