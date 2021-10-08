import Parameters: @with_kw

@with_kw struct Shape
    bbox::S.Box
    p::Array{Real}
    meshes::Array{Tuple}
end

export Shape
