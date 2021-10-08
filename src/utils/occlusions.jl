function get_occ_ocl_free(cloud, camera, depth_image, resolution)
    pixels = GL.point_cloud_to_pixel_coordinates(cloud, camera)

    THRESHOLD = 0.01

    occluded = fill(true, size(pixels)[2])
    occupied = fill(false, size(pixels)[2])

    for (i,(x,y)) in enumerate(eachcol(pixels))
        if 1 <= x <= camera.width && 1 <= y <= camera.height
            if abs(cloud[3,i] - depth_image[y,x]) <= resolution
                occupied[i] = true
            end
            if cloud[3, i] < depth_image[y,x] + 0.0
                occluded[i] = false
            end
        end
    end
    occluded = occluded .& (.!(occupied));
    free = .!(occluded) .& .!(occupied)
    occupied, occluded, free
end

export get_occ_ocl_free

