
function cardinal_rotations()
    six_dirs = [let o=zeros(3);o[i] = sign*1.0;o end for sign in [-1,1] for i in 1:3]
    cardinal_rots = []
    for dir in six_dirs
        for dir2 in six_dirs
            if dot(dir, dir2) == 0.0
                dir3 = cross(dir, dir2)
                rot = vcat(dir', dir2', dir3')
                push!(cardinal_rots, RotMatrix3{Float64}(rot))
            end
        end
    end
    cardinal_rots
end

function rotations_90()
    [
        let
            rot = zeros(3)
            rot[i] = sign * pi/2
            RotXYZ(rot...)
        end
        for i in 1:3 for sign in [-1, 1]
    ]
end

export cardinal_rotations, rotations_90