# +
import Revise
import GLRenderer as GL
import ThreeDP3 as T
import Images as I
import GenSceneGraphs as S
import PoseComposition: Pose
using Gen


hypers = T.Hyperparams(slack_dir_conc=300.0, slack_offset_var=0.5, p_outlier=0.01, 
    noise=0.2, resolution=0.5, parent_face_mixture_prob=0.99, floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))

params = T.SceneModelParameters(
    boxes=[S.Box(1.0,1.0,1.0), S.Box(1.0,1.0,1.0), S.Box(1.0,1.0,1.0)],
    get_cloud_from_poses_and_idx=(x,y,z) -> rand(3,100),
    camera_pose=T.IDENTITY_POSE,
    hyperparams=hypers, N=1)


num_obj = length(params.boxes)
g = T.graph_with_edges(3, [3=>1])
constraints = Gen.choicemap((T.structure_addr()) => g)

constraints[T.obs_addr()] = rand(3,100)

trace, _ = generate(T.scene, (params,), constraints);
