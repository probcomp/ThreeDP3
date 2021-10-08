import LinearAlgebra: norm, cross, dot

""" Construct a Planar Contact from our Hopf contact parametrization. """
function make_hopf_planar_contact(x::Real, y::Real, angle::Real,
    slack_offset::Real, slack_dir::GenDirectionalStats.UnitVector3
   ) :: S.PlanarContact
    rot = S.OUTWARD_NORMAL_FLIP * S.geodesicHopf(S.OUTWARD_NORMAL_FLIP \ slack_dir.v, angle)
    offset = inv(rot) * slack_dir.v * slack_offset
    relative_pose = Pose([x,y,0.0] .+ offset, rot)
    relative_pose_no_slack = Pose(x, y, 0.0, S.OUTWARD_NORMAL_FLIP * R.AngleAxis(angle, [0, 0, 1]...))
    slack = inv(relative_pose_no_slack) * relative_pose
    pc = S.PlanarContact(x,y,angle, slack)
    return pc
end

function geodesicHopf(newZ::StaticVector{3, <:Real}, planarAngle::Real, atol=1e-3)
  @assert (abs(norm(newZ) - 1.0) < atol) "newZ $(newZ) $(norm(newZ))"
  zUnit = @SVector([0, 0, 1])
  if newZ ≈ -zUnit
    @warn "Singularity: anti-parallel z-axis, rotation has an undetermined degree of freedom"
    axis = @SVector([1, 0, 0])
    geodesicAngle = π
  elseif newZ ≈ zUnit
    # Choice of axis doesn't matter here as long as it's nonzero
    axis = @SVector([1, 0, 0])
    geodesicAngle = 0
  else
    axis = cross(zUnit, newZ)
    @assert !(axis ≈ zero(axis)) || newZ ≈ zUnit
    geodesicAngle = let θ = asin(clamp(norm(axis), -1, 1))
      dot(zUnit, newZ) > 0 ? θ : π - θ
    end
  end
  return (R.AngleAxis(geodesicAngle, axis...) *
          R.AngleAxis(planarAngle, zUnit...))
end

# This is copy pasted from GenSceneGraphs besides the modification to expose the atol parameter.
function myInvGeodesicHopf(r::R.Rotation{3}, atol=1e-8)
  zUnit = @SVector([0, 0, 1])
  # print("myInvGeodesicHopf r: $(r) \n")
  newZ = r * zUnit
  # print("myInvGeodesicHopf newZ: $(newZ) \n")
  newZ = newZ ./ norm(newZ)
  if newZ ≈ -zUnit
    @warn "Singularity: anti-parallel z-axis, planarAngle is undetermined"
    planarAngle = 0
  else
    # Solve `planarRot == R.AngleAxis(planarAngle, zUnit...)` for `planarAngle`
    planarRot = R.AngleAxis(geodesicHopf(newZ, 0) \ r)
    axis = @SVector([planarRot.axis_x, planarRot.axis_y, planarRot.axis_z])
    # `axis` is either `zUnit` or `-zUnit`, and we need to ensure that it's
    # `zUnit`.  (Exception: the degenerate case `planarAngle == 0`)
    if axis[3] < 0
      axis = -axis
      planarAngle = -planarRot.theta
    else
      planarAngle = planarRot.theta
    end

    @assert isapprox(axis, zUnit; atol=atol) ||
            abs(rem2pi(planarAngle, RoundNearest)) < atol
  end
  return (newZ=newZ, planarAngle=planarAngle)
end

""" Recover the slack direction and slack offset from a PlanarContact. """
function planar_contact_to_slack_dir_and_offset(pc::S.PlanarContact)
    p = S.planarContactTo6DOF(pc)
    rot = p.orientation
    recovered_slack_dir = GenDirectionalStats.UnitVector3(myInvGeodesicHopf(rot).newZ...)
    recovered_offset = p.pos .- [pc.x, pc.y, 0.0]
    recovered_slack_offset = sum(recovered_offset .* (inv(rot) * recovered_slack_dir.v))
    recovered_slack_dir, recovered_slack_offset
end

export make_hopf_planar_contact, planar_contact_to_slack_dir_and_offset
