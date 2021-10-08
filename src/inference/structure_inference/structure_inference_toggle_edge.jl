@gen function structure_move_randomness(trace, i::Int, parent_idx)
    num_objects = get_num_objects(trace)
    structure = get_structure(trace)
    floating_to_sliding = isFloating(structure, i)
    world_frame_node_id = num_objects + 1
    if floating_to_sliding

        prev_parent_node = world_frame_node_id
        downstream_nodes = getDownstream(recapitate(structure), Edge(i, prev_parent_node))

        # change to sliding, on top of this object, if possible
        probs = zeros(num_objects)
        for j in downstream_nodes
            if j != nv(structure) + 1
                probs[j] = 1.0
            end
        end

        if isnothing(parent_idx)
            parent_object ~ categorical(probs ./ sum(probs))
        else
            parent_object = parent_idx
        end

        new_parent_node = parent_object
        (parent_face_probs, child_face_probs) =
            structure_move_face_distributions(trace, i, parent_object)
        parent_face ~ labeled_categorical(S.BOX_SURFACE_IDS, parent_face_probs)
        child_face ~ labeled_categorical(S.BOX_SURFACE_IDS, child_face_probs)
    else
        # change to floating
        prev_parent_node = parent(structure, i)
        new_parent_node = world_frame_node_id
    end

    # println(floating_to_sliding, " ", i, " ", prev_parent_node," ", new_parent_node)
    new_tree = replaceEdge(
        recapitate(structure),
        i,
        prev_parent_node,
        new_parent_node,
    )
    return (structure, new_tree, floating_to_sliding)
end

function structure_move_involution(
    trace,
    fwd_choices,
    fwd_ret,
    proposal_args;
    check = false,
)
    i, parent_idx  = proposal_args[1:2]
    (structure, new_tree, floating_to_sliding) = fwd_ret
    scene_graph = get_scene_graph(trace)
    constraints = choicemap()
    bwd_choices = choicemap()

    object_boxes = get_object_boxes(trace)

    local parent_face::Symbol
    local child_face::Symbol

    if isTree(new_tree)

        constraints[structure_addr()] = decapitate(new_tree)
        if floating_to_sliding

            # floating to sliding
            poses = floatingPosesOf(scene_graph)
            pose = poses[i]
            if isnothing(parent_idx)
                parent_object = fwd_choices[:parent_object]
            else
                parent_object = parent_idx
            end

            parent_face = fwd_choices[:parent_face]
            child_face = fwd_choices[:child_face]
            cm = structure_move_get_equivalent_sliding_param(
                scene_graph,
                parent_object,
                i,
                parent_face,
                child_face,
                object_boxes,
            )
            set_submap!(constraints, contact_addr(i), cm)
            jacobian_correction = TO_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION
        else

            # sliding to floating
            parent_object = parent(structure, i)
            (cm, parent_face, child_face) =
                structure_move_get_equivalent_pose(scene_graph, parent_object, i)
            set_submap!(constraints, floating_addr(i), cm)

            if isnothing(parent_idx)
                bwd_choices[:parent_object] = parent_object
            end
            bwd_choices[:parent_face] = parent_face
            bwd_choices[:child_face] = child_face
            jacobian_correction = FROM_DIRECTION_AND_PLANE_ROTATION_JACOBIAN_CORRECTION
        end
    else
        # no-op
        if floating_to_sliding
            if isnothing(parent_idx)
                bwd_choices[:parent_object] = fwd_choices[:parent_object]
            end
            bwd_choices[:parent_face] = fwd_choices[:parent_face]
            bwd_choices[:child_face] = fwd_choices[:child_face]
        end
        jacobian_correction = 0.0
    end
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight, _, _ = Gen.update(trace, args, argdiffs, constraints)
    weight += jacobian_correction
    return (new_trace, bwd_choices, weight)
end

function is_structure_move_valid(trace, i::Int, parent_idx)
    num_objects = get_num_objects(trace)
    structure = get_structure(trace)
    floating_to_sliding = isFloating(structure, i)
    world_frame_node_id = num_objects + 1
    if floating_to_sliding
        prev_parent_node = world_frame_node_id
        downstream_nodes = getDownstream(recapitate(structure), Edge(i, prev_parent_node))
        if isnothing(parent_idx)
            return length(downstream_nodes) > 1
        else
            return (parent_idx in downstream_nodes)
        end
    end
    return true
end

function structure_move(trace, i::Int, parent_idx=nothing; iterations=10)
    acceptances = false
    if is_structure_move_valid(trace, i, parent_idx)
        for _ in 1:iterations
            trace, acc = mh(
                trace,
                structure_move_randomness,
                (i, parent_idx,),
                structure_move_involution,
                check = false,
            )
            acceptances = acc || acceptances
        end
    end
    trace, acceptances
end

export structure_move, is_structure_move_valid, structure_move_randomness, structure_move_involution