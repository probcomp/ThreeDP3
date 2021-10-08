 function contact_submap_from_shape_contact(contact::S.ShapeContact) :: Gen.ChoiceMap
    pc = contact.planarContact
    angle = RotMatrix{2}(pc.angle)
    if isnothing(pc.slack)
        slack_dir = UnitVector3([0.0, 0.0, -1.0])
        slack_offset = 0.0
    else
        slack_dir, slack_offset = planar_contact_to_slack_dir_and_offset(pc)
    end
    constraints = Gen.choicemap()
    constraints[:x] = pc.x
    constraints[:y] = pc.y
    constraints[:angle] = angle
    constraints[:slack_dir] = slack_dir
    constraints[:slack_offset] = slack_offset
    constraints[:parent_face] = contact.parentFamilyId
    constraints[:child_face] = contact.childFamilyId
    constraints
end

""" Get a contact choicemap capturing the contact relationship between the specified parent and child object_boxes
    and the specified faces of those objects. """
function structure_move_get_equivalent_sliding_param(
    g::MG.MetaDiGraph,
    parent_idx::Int,
    child_idx::Int,
    parent_face::Symbol,
    child_face::Symbol,
    object_boxes,
)
    poses = collect(values(S.floatingPosesOf(g)))
    pc = S.closestApproximatingContact(
        (poses[parent_idx] * S.getContactPlane(object_boxes[parent_idx], parent_face)) \
        (poses[child_idx] * S.getContactPlane(object_boxes[child_idx], child_face)),
    )
    contact = S.ShapeContact(parent_face, (), child_face, (), pc)
    return contact_submap_from_shape_contact(contact)
end

function structure_move_get_equivalent_pose(
    g::MG.MetaDiGraph,
    parent_idx::Int,
    child_idx::Int,
)
    child_object_pose = floatingPosesOf(g)[child_idx]

    contact = S.getContact(g, obj_name_from_idx(parent_idx), obj_name_from_idx(child_idx))
    parent_face = contact.parentFamilyId
    child_face = contact.childFamilyId
    return choicemap(:pose => child_object_pose), parent_face, child_face
end


function structure_move_face_distributions(trace, child_idx::Int, parent_idx::Int)
    object_boxes = get_object_boxes(trace)

    parent_box = object_boxes[parent_idx]
    child_box = object_boxes[child_idx]

    scene_graph = get_scene_graph(trace)

    poses = floatingPosesOf(scene_graph)
    parent_pose = poses[parent_idx]
    child_pose = poses[child_idx]

    (best_parent_face_idx, best_child_face_idx), _ =
        get_closest_planar_contact(parent_pose, parent_box, child_pose, child_box)

    parent_face_probs = fill(1.0, length(S.BOX_SURFACE_IDS))
    parent_face_probs[best_parent_face_idx] = 1000.0
    child_face_probs = fill(1.0, length(S.BOX_SURFACE_IDS))
    child_face_probs[best_child_face_idx] = 1000.0

    parent_face_probs = parent_face_probs / sum(parent_face_probs)
    child_face_probs = child_face_probs / sum(child_face_probs)
    return (parent_face_probs, child_face_probs)
end

export structure_move_get_equivalent_sliding_param
