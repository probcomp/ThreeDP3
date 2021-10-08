@gen function in_place_drift_randomness(trace, idx::Int, pos_var, rot_conc, sample_contact_faces)
    offset ~ gaussianVMF(IDENTITY_POSE, pos_var, rot_conc)
    offset
end

function in_place_drift_involution(
    trace,
    fwd_choices,
    fwd_ret,
    proposal_args;
    check = false,
)
    constraints = choicemap()
    bwd_choices = choicemap()

    idx, _,_, sample_contact_faces = proposal_args
    offset = fwd_choices[:offset]
    bwd_choices[:offset] = inverse_pose(offset)

    structure = get_structure(trace)

    poses = get_poses(trace)
    object_boxes = get_object_boxes(trace)

    new_pose = get_c_relative_to_a(poses[idx], offset)

    if isFloating(structure, idx)
        constraints[floating_pose_addr(idx)] = new_pose
    else
        parent_idx = parent(structure, idx)
        @assert !isnothing(parent_idx)

        if sample_contact_faces
            (parent_face, child_face), _ =
                get_closest_planar_contact(poses[parent_idx], object_boxes[parent_idx], new_pose, object_boxes[idx])
            parent_face = S.BOX_SURFACE_IDS[parent_face]
            child_face = S.BOX_SURFACE_IDS[child_face]
        else
            parent_face = trace[contact_addr(idx, :parent_face)]
            child_face = trace[contact_addr(idx, :child_face)]
        end
        pc = S.closestApproximatingContact(
            (poses[parent_idx] * S.getContactPlane(object_boxes[parent_idx], parent_face)) \
            (new_pose * S.getContactPlane(object_boxes[idx], child_face)),
        )
        contact = S.ShapeContact(parent_face, (), child_face, (), pc)
        cm = contact_submap_from_shape_contact(contact)
        set_submap!(constraints, (contact_addr(idx)), cm)

        for child_idx in children(structure, idx)
            if sample_contact_faces
                (parent_face, child_face), _ =
                    get_closest_planar_contact(new_pose, object_boxes[idx], poses[child_idx], object_boxes[child_idx])
                parent_face = S.BOX_SURFACE_IDS[parent_face]
                child_face = S.BOX_SURFACE_IDS[child_face]
            else
                parent_face = trace[contact_addr(child_idx, :parent_face)]
                child_face = trace[contact_addr(child_idx, :child_face)]
            end

            pc = S.closestApproximatingContact(
                (new_pose * S.getContactPlane(object_boxes[idx], parent_face)) \
                (poses[child_idx] * S.getContactPlane(object_boxes[child_idx], child_face)),
            )
            contact = S.ShapeContact(parent_face, (), child_face, (), pc)
            cm = contact_submap_from_shape_contact(contact)
            set_submap!(constraints, contact_addr(child_idx), cm)
        end
    end

    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight, _, _ = Gen.update(trace, args, argdiffs, constraints)
    return (new_trace, bwd_choices, weight)
end

export in_place_drift_randomness, in_place_drift_involution