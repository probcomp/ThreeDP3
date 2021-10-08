
isTree(g::LG.AbstractGraph) = nv(g) == ne(g)+1 && is_connected(g)
isFloating(diforest::LG.SimpleDiGraph, i::Int) = i ∈ S.rootsOfForest(diforest)

export isTree, isFloating

import Base.parent
function parent(diforest::LG.SimpleDiGraph, i::Int)
    # Forests guarantee all vertices only have a single incoming edge.
    parents = LightGraphs.inneighbors(diforest, i)
    @assert length(parents) <= 1
    return isempty(parents) ? nothing : first(parents)
end

function children(diforest::LG.SimpleDiGraph, i::Int)
  # Forests guarantee all vertices only have a single incoming edge.
  children = LightGraphs.outneighbors(diforest, i)
  return children
end

export parent, children


# Construct directed graph with n vertices and no edges.
function no_edge_graph(n::Int)::LG.SimpleDiGraph
    g = LG.SimpleDiGraph()
    LG.add_vertices!(g, n)
    g
end

# Construct directed graph with n vertices and edges specfied as a list of (parent, child) tuples.
function graph_with_edges(n::Int, edges)::LG.SimpleDiGraph
    g = LG.SimpleDiGraph()
    LG.add_vertices!(g, n)
    for (i,j) in edges
        LG.add_edge!(g, i, j)
    end
    g
end

export no_edge_graph, graph_with_edges

function getDownstream(tree::LG.SimpleGraph, e::LG.Edge) :: Set{Int}
    nodes = Set{Int}()
    _getDownstream!(nodes, tree, e)
    nodes
end

function _getDownstream!(nodes::Set{Int}, tree::LG.SimpleGraph, e::LG.Edge)
    i, j = src(e), dst(e)
    if !((j in neighbors(tree, i)) && (i in neighbors(tree, j)))
        error("No edge (i, j)")
    end
    push!(nodes, j)
    for n in neighbors(tree, j)
        if n != i
        _getDownstream!(nodes, tree, LG.Edge(j, n))
        end
    end
end

# Delete the edge from i->j and add an edge  from i->k.
function replaceEdge(tree::LG.SimpleGraph, i::Int, j::Int, k::Int)
    @assert has_edge(tree, LG.Edge(i, j))
    @assert k in getDownstream(tree, LG.Edge(i, j))
    newTree = deepcopy(tree)
    LG.rem_edge!(newTree, i, j)
    LG.add_edge!(newTree, i, k)
    newTree
end

# Peform a BFS from a specified node, to assign directions to each of the edges.
# Then, remove that specified node from the directed graph and return what is left behind.
function decapitate(tree::LG.SimpleGraph; root::Union{Int,Nothing}=nothing)
    root = isnothing(root) ? nv(tree) : root
    diforest = bfs_tree(tree, root)
    rem_vertex!(diforest, root)
    diforest
end

# Add a new node to a directed forest. Then, for each existing root node of the forest, add
# an edge from the newly added node to that root. This turns the forest into a tree.
function recapitate(diforest::LG.SimpleDiGraph)
    forestRoots = S.rootsOfForest(diforest)
    tree = LG.SimpleGraph(diforest)
    add_vertex!(tree)
    root = nv(tree)
    for forestRoot in forestRoots
        LG.add_edge!(tree, root, forestRoot)
    end
    @assert ne(tree) == nv(tree) - 1
    tree
end

function edges(g::LG.SimpleDiGraph)
    collect(LG.edges(g))
end


export no_edge_graph, graph_with_edges, isTree, decapitate, uniform_diforest, isFloating, edges
