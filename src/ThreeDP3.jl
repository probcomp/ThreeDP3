module ThreeDP3

greet() = print("Hello World!")

using Gen

import GLRenderer as GL
import GenDirectionalStats
import GenSceneGraphs as S
import Rotations as R
import PoseComposition as PC
import PoseComposition: Pose
import StaticArrays:SVector, @SVector, StaticVector, SMatrix
import Colors
import PyCall
import LightGraphs as LG
import MetaGraphs as MG

include("utils/bbox.jl")
include("utils/clustering.jl")
include("utils/data_loader.jl")
include("utils/icp.jl")
include("utils/occlusions.jl")
include("utils/pose.jl")
include("utils/pose_utils.jl")
include("utils/scale_camera.jl")
include("utils/utils.jl")

include("distributions/distributions.jl")

include("contact/contact.jl")
include("graphs/graphs.jl")
include("shape/shape.jl")

include("model/sg_model.jl")
include("model/sg_model_utils.jl")

include("inference/inference.jl")

end # module
