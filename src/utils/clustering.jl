import Clustering

function dbscan_cluster(cloud::Matrix; radius=0.25, min_cluster_size=10)
    if size(cloud, 2) <= 3
        return Int[]
    end
    clusters = Clustering.dbscan(cloud, radius, min_neighbors = 1, min_cluster_size = min_cluster_size)
    assign = zeros(Int, size(cloud)[2])
    for (i,r) in enumerate(clusters)
        assign[r.core_indices] .= i
    end
    assign
end
function get_entities_from_assignment(cloud::Matrix, assign)
    @assert size(cloud)[2] == length(assign)
    [cloud[:, assign .== i] for i in sort(unique(assign)) if i != 0]
end



using Statistics
function centroid(cloud::Matrix)
    mean(cloud,dims=2)[:]
end

export dbscan_cluster, get_entities_from_assignment, centroid
