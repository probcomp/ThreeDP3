#### Forced version. Not a valid proposal

function force_structure(trace, structure)
    current_poses = floatingPosesOf(get_scene_graph(trace))
    constraints = choicemap();
    num_objects = get_num_objects(trace)
    bboxes = get_object_boxes(trace)

    constraints[structure_addr()] = structure
    for idx in 1:num_objects
        if isFloating(structure, idx)
        constraints[floating_pose_addr(idx)] = current_poses[idx]
        else
            parent_idx = parent(structure, idx)
            (face1,face2), _ = get_closest_planar_contact(current_poses[parent_idx],
                                                bboxes[parent_idx],
                                                current_poses[idx],
                                                bboxes[idx])
            cm = structure_move_get_equivalent_sliding_param(get_scene_graph(trace), parent_idx, idx,
                                                            S.BOX_SURFACE_IDS[face1],
                S.BOX_SURFACE_IDS[face2],
                bboxes
            )
            set_submap!(constraints, contact_addr(idx), cm)
        end
    end
    new_trace, _ = update(trace, constraints)
    new_trace
end

export force_structure

function validate_all_contacts_use_closest_faces(trace)
    current_poses = floatingPosesOf(get_scene_graph(trace))
    structure = get_structure(trace)
    num_objects = get_num_objects(trace)
    bboxes = get_object_boxes(trace)

    for idx in 1:num_objects
        if isFloating(structure, idx)
            continue
        end

        parent_idx = parent(structure, idx)
        (face1,face2), _ = get_closest_planar_contact(current_poses[parent_idx],
            bboxes[parent_idx],
            current_poses[idx],
            bboxes[idx])

        if (
            S.BOX_SURFACE_IDS[face1] != trace[contact_addr(idx, :parent_face)] ||
            S.BOX_SURFACE_IDS[face2] != trace[contact_addr(idx, :child_face)]
        )
            return false
        end

    end
    true
end

export validate_all_contacts_use_closest_faces


###### Involution move

@dist graph_categorical(graphs, weights) = graphs[categorical(weights)]

@gen function full_structure_move_randomness(trace, graphs, weights)
    num_objects = get_num_objects(trace)
    prev_structure = get_structure(trace)
    structure ~ graph_categorical(graphs, weights)
    (prev_structure, structure)
end

function full_structure_move_involution_func(
    trace,
    fwd_choices,
    fwd_ret,
    proposal_args;
    check = false,
)
    scene_graph = get_scene_graph(trace)
    (prev_structure, new_structure) = fwd_ret
    bboxes = get_object_boxes(trace)

    current_poses = floatingPosesOf(get_scene_graph(trace))

    constraints = choicemap()
    constraints[structure_addr()] = new_structure

    bwd_choices = choicemap(:structure => prev_structure)

    num_objects = get_num_objects(trace)

    jacobian_correction = 0.0

    for idx in 1:num_objects
        if !isFloating(new_structure, idx)
            if !isFloating(prev_structure,idx) && (parent(prev_structure,idx) == parent(new_structure, idx))
                continue
            end

            new_parent_idx = parent(new_structure, idx)

            (face1,face2), _ = get_closest_planar_contact(current_poses[new_parent_idx],
                bboxes[new_parent_idx],
                current_poses[idx],
                bboxes[idx])

            parent_face = S.BOX_SURFACE_IDS[face1]
            child_face = S.BOX_SURFACE_IDS[face2]

            cm = structure_move_get_equivalent_sliding_param(
                scene_graph,
                new_parent_idx,
                idx,
                parent_face,
                child_face,
                bboxes,
            )
            set_submap!(constraints, contact_addr(idx), cm)
            jacobian_correction += TO_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION

        else
            if isFloating(prev_structure,idx)
                continue
            end
            constraints[floating_pose_addr(idx)] = current_poses[idx]
            jacobian_correction += FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION

        end
    end

    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight, _, _ = Gen.update(trace, args, argdiffs, constraints)
    weight += jacobian_correction
    return (new_trace, bwd_choices, weight)
end

function full_structure_move(trace, graphs, weights; iterations=10)
    acceptances = false
    for _ in 1:iterations
        trace, acc = mh(
            trace,
            full_structure_move_randomness,
            (graphs, weights,),
            full_structure_move_involution_func,
            check = false,
        )
        acceptances = acc || acceptances
    end
    trace, acceptances
end

export full_structure_move_randomness, full_structure_move_involution_func, full_structure_move