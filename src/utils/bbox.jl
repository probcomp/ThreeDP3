import GeometryBasics: Point

function get_dims(box::S.Box)
    [box.sizeX, box.sizeY, box.sizeZ]
end

function axis_aligned_bbox_from_point_cloud(point_cloud::Matrix)
    x_bounds = min_max(point_cloud[1, :])
    y_bounds = min_max(point_cloud[2, :])
    z_bounds = min_max(point_cloud[3, :])

    return Pose([mean(x_bounds), mean(y_bounds), mean(z_bounds)], IDENTITY_ORIENTATION),
        S.Box(x_bounds[2] - x_bounds[1],
                            y_bounds[2] - y_bounds[1],
                            z_bounds[2] - z_bounds[1])
end

function get_bbox_corners(bbox::S.Box)
    x,y,z = get_dims(bbox) ./ 2
    nominal_corners = collect([
        -x -y -z;
        -x y -z;
        x y -z;
        x -y -z;
        -x -y  z;
        -x y  z;
        x y  z;
        x -y  z;
    ]')
    nominal_corners
end

function get_bbox_segments_point_list(bbox::S.Box, pose::Pose)
    x,y,z = get_dims(bbox) ./ 2
    nominal_corners = collect([
        # bottom square
        x y -z;
        x -y -z;

        x -y -z;
        -x -y -z;

        -x -y -z;
        -x y -z;

        -x y -z;
        x y -z;

        # top square
        x y z;
        x -y z;

        x -y z;
        -x -y z;

        -x -y z;
        -x y z;

        -x y z;
        x y z;

        # top to bottom
        -x -y z;
        -x -y -z;

        x -y z;
        x -y -z;

        -x y z;
        -x y -z;

        x y z;
        x y -z;
    ]')
    corners = move_points_to_frame_b(nominal_corners, pose)
    Point.(corners[1,:],corners[2,:], corners[3,:])
end


export axis_aligned_bbox_from_point_cloud, get_bbox_corners, get_bbox_segments_point_list
