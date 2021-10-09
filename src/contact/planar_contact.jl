function get_closest_planar_contact(parent_pose, parent_bbox, child_pose, child_bbox)
    possible_face_pairs = [
        (face1, face2) for face1 = 1:length(S.BOX_SURFACE_IDS)
        for face2 = 1:length(S.BOX_SURFACE_IDS)
    ]
    corresponding_planar_contacts = [
        S.closestApproximatingContact(
            (parent_pose * S.getContactPlane(parent_bbox, S.BOX_SURFACE_IDS[face1])) \
            (child_pose * S.getContactPlane(child_bbox, S.BOX_SURFACE_IDS[face2])),
        ) for (face1, face2) in possible_face_pairs
    ]
    error = [pose_distance(pc.slack, PC.IDENTITY_POSE; r=5.0) for pc in corresponding_planar_contacts]
    best_contact_index = argmin(error)
    return possible_face_pairs[best_contact_index], corresponding_planar_contacts[best_contact_index]
end

export get_closest_planar_contact
