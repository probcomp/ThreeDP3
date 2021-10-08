function scale_down_camera(camera, FACTOR)
    camera_modified = GL.CameraIntrinsics(
        width=round(Int,camera.width/FACTOR), height=round(Int,camera.height/FACTOR),
        fx=camera.fx/FACTOR, fy=camera.fy/FACTOR, cx=camera.cx/FACTOR, cy=camera.cy/FACTOR,
        near=camera.near, far=camera.far)
    camera_modified
end

export scale_down_camera
