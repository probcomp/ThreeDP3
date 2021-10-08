using Parameters

function compute_parent_face_categorical_weights(parent_contact_face, mixture_prob)
    @assert 0 < mixture_prob < 1
    if isnothing(parent_contact_face)
        probs = ones(length(S.BOX_SURFACE_IDS))
        probs /= sum(probs)
        return S.BOX_SURFACE_IDS, probs
    else
        faces = [:top :bottom;:left :right;:front :back]
        i,j = let
            idxs = findall(faces .== parent_contact_face)
            @assert length(idxs) == 1
            Tuple(idxs[1])
        end
        j = (j % 2) + 1
        weights = zeros(size(faces))
        weights[i,j] = 1.0

        uniform_weight = ones(size(faces))
        uniform_weight /= sum(uniform_weight)

        weights = weights * mixture_prob + uniform_weight * (1.0 - mixture_prob)

        return reshape(faces, (6,)), reshape(weights, (6,))
    end
end

@gen function sample_contact(boxes, parent_idx, parent_contact_face, hyperparams)
    # For both the parent object and child object, sample which of the 6 faces
    # of the bounding box are in contact.

    faces, probs = compute_parent_face_categorical_weights(parent_contact_face,
            hyperparams.parent_face_mixture_prob)
    parent_face ~ labeled_categorical(faces, probs)

    probs = ones(length(S.BOX_SURFACE_IDS))
    probs /= sum(probs)
    child_face ~ labeled_categorical(S.BOX_SURFACE_IDS, probs)

    parent_shape = boxes[parent_idx]
    if parent_face in [:top, :bottom]
        x ~ uniform(-parent_shape.sizeX/2, parent_shape.sizeX/2)
        y ~ uniform(-parent_shape.sizeY/2, parent_shape.sizeY/2)
    elseif parent_face in [:front, :back]
        x ~ uniform(-parent_shape.sizeX/2, parent_shape.sizeX/2)
        y ~ uniform(-parent_shape.sizeZ/2, parent_shape.sizeZ/2)
    elseif parent_face in [:left, :right]
        x ~ uniform(-parent_shape.sizeZ/2, parent_shape.sizeZ/2)
        y ~ uniform(-parent_shape.sizeY/2, parent_shape.sizeY/2)
    else
        @error "Invalid Parent Face"
    end

    # Sample the (x,y) coordinate along the parent's surface plane at which the contact occurs.
    # if length(shapes) == parent_idx
    # x ~ uniform(-parent_size/2, parent_size/2)
    # y ~ uniform(-parent_size/2, parent_size/2)
    # else
    #     x ~ normal(0.0, 0.25 * parent_size)
    #     y ~ normal(0.0, 0.25 * parent_size)
    # end

    # Sample the angle of rotation of the child object about the contact plane's normal vector.
    angle = R.rotation_angle({:angle} ~ GenDirectionalStats.uniform_rot2())
    # Since the contact may not be perfectly flush, we additionally sample a unit vector
    # representing the child object's outward contact normal (in the parent object's contact frame).
    # (This distribution is concentratedat [0,0,-1], a vector pointing exactly downward, which would
    # mean the contact planes are exactly aligned.
    slack_dir ~ GenDirectionalStats.vmf_3d_direction(GenDirectionalStats.UnitVector3(0.0, 0.0, -1.0), hyperparams.slack_dir_conc)
    # Sample a distance to shift the contact point along the sampled slack_dir
    slack_offset ~ normal(0, hyperparams.slack_offset_var)
    return S.ShapeContact(parent_face, Float64[], child_face, Float64[],
                          make_hopf_planar_contact(x, y, angle, slack_offset, slack_dir))
end

@gen function sample_floating_pose(hyperparams)
    # Sample a floating pose with position uniformly over a 3D region and orientation uniformly.
    pose ~ uniformPose(hyperparams.floating_position_bounds...)
end

@gen function sample_scene_graph_params(structure, boxes, hyperparams)
    scene_graph = S.SceneGraph()

    for (idx, bbox) in enumerate(boxes)
        # Each Shape object has an associated bounding box.
        S.addObject!(scene_graph, obj_name_from_idx(idx), bbox)
    end

    roots = S.rootsOfForest(structure)
    dists = LG.dijkstra_shortest_paths(structure, roots).dists
    unique_dists = sort(unique(dists))

    # Given the structure (a graph) sample parameters of the scene graph.
    for d in unique_dists
        for idx in findall(dists .== d)
            if isFloating(structure, idx)
                # For floating (root nodes in the graph), sample a floating pose and set the pose in the scene graph.
                pose = {idx => :floating} ~ sample_floating_pose(hyperparams)
                S.setPose!(scene_graph, obj_name_from_idx(idx), pose)
            else
                # For contacting objects, sample parameters of the contact and set the contact relationship in the scene graph.
                parent_idx = parent(structure, idx)
                parent_contact_face = let
                    parents_parent_idx = parent(structure, parent_idx)
                    face=nothing
                    if !isnothing(parents_parent_idx)
                        shape_contact = S.getContact(scene_graph,
                            obj_name_from_idx(parents_parent_idx),obj_name_from_idx(parent_idx))
                        face = shape_contact.childFamilyId
                    end
                    face
                end
                contact = {idx => :contact} ~ sample_contact(boxes, parent_idx, parent_contact_face, hyperparams)
                S.setContact!(scene_graph, obj_name_from_idx(parent_idx), obj_name_from_idx(idx), contact)
            end
        end
    end
    return scene_graph
end


@gen (static) function scene_graph_prior(model_params)
    structure ~ uniform_diforest(length(model_params.boxes))
    scene_graph = {:params} ~ sample_scene_graph_params(structure, model_params.boxes, model_params.hyperparams)
    return scene_graph
end


@with_kw struct Hyperparams
    slack_dir_conc::Float32
    slack_offset_var::Float32
    p_outlier::Real
    noise::Real
    resolution::Real
    parent_face_mixture_prob::Real
    floating_position_bounds::Tuple
end

@with_kw struct SceneGraphPriorModelParameters
    boxes::Vector{S.Box}
    hyperparams::Hyperparams
end

@with_kw struct SceneModelParameters
    boxes::Vector{S.Box}
    hyperparams::Hyperparams
    get_cloud_from_poses_and_idx::Function
    camera_pose::Pose
    N::Int
end

export Hyperparams, SceneGraphPriorModelParameters, SceneModelParameters

function render_clouds(model_params::SceneModelParameters, scene_graph)
    # Since each type of object has an associated shape distribution, we approximately integrate out the shape variable
    # by sampling shapes from each object type's shape distribution and then summing the weighted scores for these samples.
    # This sampling does not occur in this generative function, instead the samples are stored in the `meshes` field of each
    # Shape object and we expect there to be `N` of those samples in the `meshes` list. Below, we render `N` point clouds of the
    # scene, where the i-th point cloud is generated by taking the i-th sample from each object's `meshes` list and rendering a
    # scene using those shapes (at the poses sampled in the above code).

    # Floating poses of all objects as sampled above.
    poses = floatingPosesOf(scene_graph)
    # N rendered point clouds
    rendered_clouds = [
        let
            rendered_point_cloud_in_camera_frame = model_params.get_cloud_from_poses_and_idx(
                poses, i, model_params.camera_pose
            )
            rendered_point_cloud_in_world_frame = move_points_to_frame_b(rendered_point_cloud_in_camera_frame,model_params.camera_pose)
            voxelize(rendered_point_cloud_in_world_frame, model_params.hyperparams.resolution)
        end
        for i in 1:model_params.N
    ]
    rendered_clouds
end


"""
Sample a scene
    shapes : Vector of `Shape` objects
    get_cloud_from_objects : Function to render Objects (consisting of a pose and a mesh) from a specifried camera viewpoint
    camera_pose : Pose of camera
    likelihood_params : Hyperparameters of the likelihood model.
    N : Number of shape samples to approximately integrate over
"""
@gen (static) function scene(model_params::SceneModelParameters)
    scene_graph ~ scene_graph_prior(model_params)
    rendered_clouds = render_clouds(model_params, scene_graph)
    p_outlier ~ exponential(1/model_params.hyperparams.p_outlier)
    noise ~ exponential(1/model_params.hyperparams.noise)
    obs_cloud = {:obs} ~ uniform_mixture_from_template_multi_cloud(
                                                rendered_clouds,
                                                p_outlier,
                                                noise,
                                                (-100.0,100.0,-100.0,100.0,-100.0,300.0))
    return (scene_graph=scene_graph,
            rendered_clouds=rendered_clouds,
            obs_cloud=obs_cloud)
end

@gen (static) function scene_graph_prior_model(model_params)
    scene_graph ~ scene_graph_prior(model_params)
    return (scene_graph=scene_graph, temp=1.0)
end


Gen.@load_generated_functions


export scene, scene_graph_prior_model
