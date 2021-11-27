import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import MiniGSG as S
import PoseComposition: Pose
using Gen
import MeshCatViz

import MeshCatViz
MeshCatViz.setup_visualizer()

import PyCall
PyCall.py"""

import os
import numpy as np
import h5py as h5
from imageio import imread
import IPython
import math

def im2col(im, psize):
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(im_pad.shape[1] / psize, psize,
                                     im_pad.shape[2] / psize, psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])

def filterDiscontinuities(depthMap):
    filt_size = 7
    thresh = 1000

    # Ensure that filter sizes are okay
    assert filt_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = (filt_size - 1) / 2
    patches = 1.0 * im2col(depthMap, filt_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depthMap * (1 - final_mark)

def registerDepthMap(unregisteredDepthMap,
                     rgbImage,
                     depthK,
                     rgbK,
                     H_RGBFromDepth):


    unregisteredHeight = unregisteredDepthMap.shape[0]
    unregisteredWidth = unregisteredDepthMap.shape[1]

    registeredHeight = rgbImage.shape[0]
    registeredWidth = rgbImage.shape[1]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))

    xyzDepth = np.empty((4,1))
    xyzRGB = np.empty((4,1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / depthK[0,0]
    invDepthFy = 1.0 / depthK[1,1]
    depthCx = depthK[0,2]
    depthCy = depthK[1,2]

    rgbFx = rgbK[0,0]
    rgbFy = rgbK[1,1]
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
      for u in range(unregisteredWidth):

            depth = unregisteredDepthMap[v,u]
            if depth == 0:
                continue

            xyzDepth[0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[2] = depth

            xyzRGB[0] = (H_RGBFromDepth[0,0] * xyzDepth[0] +
                         H_RGBFromDepth[0,1] * xyzDepth[1] +
                         H_RGBFromDepth[0,2] * xyzDepth[2] +
                         H_RGBFromDepth[0,3])
            xyzRGB[1] = (H_RGBFromDepth[1,0] * xyzDepth[0] +
                         H_RGBFromDepth[1,1] * xyzDepth[1] +
                         H_RGBFromDepth[1,2] * xyzDepth[2] +
                         H_RGBFromDepth[1,3])
            xyzRGB[2] = (H_RGBFromDepth[2,0] * xyzDepth[0] +
                         H_RGBFromDepth[2,1] * xyzDepth[1] +
                         H_RGBFromDepth[2,2] * xyzDepth[2] +
                         H_RGBFromDepth[2,3])

            invRGB_Z  = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > registeredDepthMap[vRGB,uRGB]:
                registeredDepthMap[vRGB,uRGB] = registeredDepth

    return registeredDepthMap

def registeredDepthMapToPointCloud(depthMap, rgbImage, rgbK, organized=True):
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]
    invRGBFx = 1.0/rgbK[0,0]
    invRGBFy = 1.0/rgbK[1,1]

    height = depthMap.shape[0]
    width = depthMap.shape[1]

    if organized:
      cloud = np.empty((height, width, 6), dtype=np.float)
    else:
      cloud = np.empty((1, height*width, 6), dtype=np.float)

    goodPointsCount = 0
    for v in range(height):
        for u in range(width):

            depth = depthMap[v,u]

            if organized:
              row = v
              col = u
            else:
              row = 0
              col = goodPointsCount

            if depth <= 0:
                if organized:
                    if depth <= 0:
                       cloud[row,col,0] = float('nan')
                       cloud[row,col,1] = float('nan')
                       cloud[row,col,2] = float('nan')
                       cloud[row,col,3] = 0
                       cloud[row,col,4] = 0
                       cloud[row,col,5] = 0
                continue

            cloud[row,col,0] = (u - rgbCx) * depth * invRGBFx
            cloud[row,col,1] = (v - rgbCy) * depth * invRGBFy
            cloud[row,col,2] = depth
            cloud[row,col,3] = rgbImage[v,u,0]
            cloud[row,col,4] = rgbImage[v,u,1]
            cloud[row,col,5] = rgbImage[v,u,2]
            if not organized:
              goodPointsCount += 1

    if not organized:
      cloud = cloud[:,:goodPointsCount,:]

    return cloud

def writePLY(filename, cloud, faces=[]):
    if len(cloud.shape) != 3:
        print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(cloud.shape))
        return

    color = True if cloud.shape[2] == 6 else False
    num_points = cloud.shape[0]*cloud.shape[1]

    header_lines = [
        'ply',
        'format ascii 1.0',
        'element vertex %d' % num_points,
        'property float x',
        'property float y',
        'property float z',
        ]
    if color:
        header_lines.extend([
        'property uchar diffuse_red',
        'property uchar diffuse_green',
        'property uchar diffuse_blue',
        ])
    if faces != None:
        header_lines.extend([
        'element face %d' % len(faces),
        'property list uchar int vertex_indices'
        ])

    header_lines.extend([
      'end_header',
      ])

    f = open(filename, 'w+')
    f.write('\n'.join(header_lines))
    f.write('\n')

    lines = []
    for i in range(cloud.shape[0]):
        for j in range(cloud.shape[1]):
            if color:
                lines.append('%s %s %s %d %d %d' % tuple(cloud[i, j, :].tolist()))
            else:
                lines.append('%s %s %s' % tuple(cloud[i, j, :].tolist()))

    for face in faces:
        lines.append(('%d' + ' %d'*len(face)) % tuple([len(face)] + list(face)))

    f.write('\n'.join(lines) + '\n')
    f.close()

def writePCD(pointCloud, filename, ascii=False):
    if len(pointCloud.shape) != 3:
        print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(pointCloud.shape))
        return
    with open(filename, 'w') as f:
        height = pointCloud.shape[0]
        width = pointCloud.shape[1]
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if pointCloud.shape[2] == 3:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % width)
        f.write("HEIGHT %d\n" % height)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % (height * width))
        if ascii:
          f.write("DATA ascii\n")
          for row in range(height):
            for col in range(width):
                if pointCloud.shape[2] == 3:
                    f.write("%f %f %f\n" % tuple(pointCloud[row, col, :]))
                else:
                    f.write("%f %f %f" % tuple(pointCloud[row, col, :3]))
                    r = int(pointCloud[row, col, 3])
                    g = int(pointCloud[row, col, 4])
                    b = int(pointCloud[row, col, 5])
                    rgb_int = (r << 16) | (g << 8) | b
                    packed = pack('i', rgb_int)
                    rgb = unpack('f', packed)[0]
                    f.write(" %.12e\n" % rgb)
        else:
          f.write("DATA binary\n")
          if pointCloud.shape[2] == 6:
              # These are written as bgr because rgb is interpreted as a single
              # little-endian float.
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('b', np.uint8),
                             ('g', np.uint8),
                             ('r', np.uint8),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z', 'r', 'g', 'b']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)
          else:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)

def getRGBFromDepthTransform(calibration, camera, referenceCamera):
    irKey = "H_{0}_ir_from_{1}".format(camera, referenceCamera)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)

    rgbFromRef = calibration[rgbKey][:]
    irFromRef = calibration[irKey][:]

    return np.dot(rgbFromRef, np.linalg.inv(irFromRef))


def get_data(ycb_data_folder, target_object, viewpoint_camera, viewpoint_angle):
    referenceCamera = "NP5"
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    depthFilename = os.path.join(ycb_data_folder+ target_object, basename + ".h5")
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")

    calibrationFilename = os.path.join(ycb_data_folder+target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)

    rgbImage = imread(rgbFilename)
    depthK = calibration["{0}_ir_K".format(viewpoint_camera)][:]
    rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
    depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters
    H_RGBFromDepth = getRGBFromDepthTransform(calibration, viewpoint_camera, referenceCamera)

    unregisteredDepthMap = h5.File(depthFilename)["depth"][:]
    unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale

    registeredDepthMap = registerDepthMap(unregisteredDepthMap,
                                          rgbImage,
                                          depthK,
                                          rgbK,
                                          H_RGBFromDepth)

    pointCloud = registeredDepthMapToPointCloud(registeredDepthMap, rgbImage, rgbK)
    return pointCloud
"""

PyCall.py"get_data"("/home/nishadg/Downloads/003_cracker_box_berkeley_rgbd/","003_cracker_box","NP3",15)

# +
import HDF5
filename = "/home/nishadg/Downloads/003_cracker_box_berkeley_rgbd/003_cracker_box/NP2_0.h5"
d = HDF5.h5open(filename, "r+")
depth = Float64.(HDF5.read(d["depth"]))
filename = "/home/nishadg/Downloads/003_cracker_box_berkeley_rgbd/003_cracker_box/calibration.h5"
c = HDF5.h5open(filename, "r+");
K = HDF5.read(c["NP2_ir_K"])
depthScale = HDF5.read(c["NP2_ir_depth_scale"]) * .0001 # 100um to meters

depth .*= depthScale;
depth = collect(transpose(depth))
camera = GL.CameraIntrinsics(size(depth)[2], size(depth)[1], K[1,1], K[2,2],K[3,1],K[3,2], 0.01, 100.0)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth, camera))
MeshCatViz.viz(cloud)

# -
import Plots
Plots.heatmap(depth)

import Pkg; Pkg.add("Plots")


YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

IDX =100
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
camera = T.scale_down_camera(original_camera, 4)
@show camera
img = I.colorview(I.Gray, gt_depth_image ./ maximum(gt_depth_image))
I.colorview(I.RGB, permutedims(Float64.(rgb_image)./255.0, (3,1,2)))


# +
resolution = 0.5

renderer = GL.setup_renderer(camera, GL.DepthMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, f)
end
depth_image = GL.gl_render(
    renderer, ids, gt_poses, T.IDENTITY_POSE)
depth_image[depth_image .== 50000.0] .= 200.0

function get_cloud_from_ids_and_poses(ids, poses, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_image, camera))
    cloud = cloud[:,cloud[3,:] .< (camera.far - 1.0)]
    if size(cloud)[2] == 0
        cloud = zeros(3,1)
    end
    cloud
end

img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))

# +

hypers = T.Hyperparams(slack_dir_conc=300.0, slack_offset_var=0.5, p_outlier=0.01, 
    noise=0.2, resolution=0.5, parent_face_mixture_prob=0.99, floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))

params = T.SceneModelParameters(
    boxes=vcat([id_to_box[id] for id in ids],[S.Box(40.0,40.0,0.1)]),
    get_cloud_from_poses_and_idx=
    (poses, idx, p) -> get_cloud_from_ids_and_poses(ids, poses[1:end-1], cam_pose),
    camera_pose=cam_pose,
    hyperparams=hypers, N=1)

table_pose = Pose(0.0, 0.0, -0.1)
num_obj = length(params.boxes)
g = T.graph_with_edges(num_obj, [])
constraints = Gen.choicemap((T.structure_addr()) => g, T.floating_pose_addr(num_obj) => table_pose)
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
end

obs_cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(gt_depth_image, original_camera))
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 1.2 .< obs_cloud[3, :] .< 40.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
constraints[T.obs_addr()] = T.voxelize(obs_cloud, params.hyperparams.resolution)
MeshCatViz.reset_visualizer()
MeshCatViz.viz(constraints[T.obs_addr()])

trace, _ = generate(T.scene, (params,), constraints);
# -

MeshCatViz.reset_visualizer()
MeshCatViz.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)

traces = [trace];

for i in 1:length(ids)
    if !T.isFloating(T.get_structure(trace),i)
        continue
    end
    pose_addr = T.floating_pose_addr(i)
    for _ in 1:50
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0)
        if acc traces = push!(traces, trace) end

        trace, acc = T.drift_move(trace, pose_addr, 1.0, 100.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 1.5, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 100.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 5000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 1, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 2, 1000.0)
        if acc traces = push!(traces, trace) end
        trace, acc = T.pose_flip_move(trace, pose_addr, 3, 1000.0)
        if acc traces = push!(traces, trace) end
    end
end


CRAZY_POSE = Pose(-100.0 * ones(3), T.IDENTITY_ORIENTATION)
for i in 1:length(ids)
    if !T.isFloating(T.get_structure(trace),i)
        continue
    end
    addr = T.floating_pose_addr(i)

    t, = Gen.update(trace, Gen.choicemap(T.floating_pose_addr(i) => CRAZY_POSE))
    valid = T.get_unexplained_obs_cloud(t)

    p = T.icp_object_pose(trace[addr], trace[:obs][:,valid],
                          p-> get_cloud_from_ids_and_poses(
                            [ids[i]], [p], cam_pose), false)
    for _ in 1:3
        trace, acc = T.pose_mixture_move(trace, addr, [trace[addr], p], [0.5, 0.5], 0.001, 5000.0)
        if acc traces = push!(traces, trace) end
    end
end

MeshCatViz.reset_visualizer()
MeshCatViz.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
MeshCatViz.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)

depth_image = GL.gl_render(
    renderer, ids, T.get_poses(trace)[1:end-1], cam_pose)
depth_image[depth_image .== 50000.0] .= 200.0
img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))

# +
renderer_texture = GL.setup_renderer(original_camera, GL.TextureMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,t = renderer_texture.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer_texture, v, n, f, t,
        texture_paths[id]
    )
end


# -

rgb_image, depth_image = GL.gl_render(
    renderer_texture, ids, T.get_poses(trace)[1:end-1], cam_pose)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))


