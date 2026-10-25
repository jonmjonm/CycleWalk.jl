## move these functions to MetropolizeForestRecom.jl at some point

"""
    multi_level_graph(base_graph, levels)

Create a [`MultiLevelGraph`](@ref) from `base_graph` and a set of level names
(keys in `base_graph.node_attributes`), reordered from coarsest to finest.

The hierarchical ordering is inferred from the node attributes: level `A` is treated
as a parent of level `B` when each value of `B` maps to exactly one value of `A`.
The hierarchy must be strict and connected with a single root level; ambiguous
parents, missing levels, or multiple roots raise an error.
"""
function multi_level_graph(base_graph::BaseGraph, levels::Vector{String})
    """
    Create a MultiLevelGraph with hierarchically ordered levels.
    
    Given a set of level names that are keys in base_graph.node_attributes,
    determine the strict hierarchical ordering (coarsest to finest) and 
    return a MultiLevelGraph.
    
    # Arguments
    - base_graph::BaseGraph: The base graph with node attributes
    - levels::Vector{String}: Names of hierarchical levels (keys in node_attributes)
    
    # Returns
    - MultiLevelGraph with levels ordered from coarsest to finest
    
    # Errors
    - If the hierarchy is not strict (ambiguous parent-child relationships)
    - If any level name is not found in node_attributes
    """
    
    # Validate all levels exist in node_attributes
    for level in levels
        if !all(haskey(node_attrs, level) 
                for node_attrs in values(base_graph.node_attributes))
            error("Level '$level' not found in all node attributes")
        end
    end
    
    # Single level is trivial
    if length(levels) == 1
        return MultiLevelGraph(base_graph, levels)
    end
    
    # Build parent-child relationships
    parent = fill(-1, length(levels)) 
    # parent[i] = j means levels[j] is parent of levels[i]
    
    for i in 1:length(levels)
        for j in 1:length(levels)
            i == j && continue
            
            # Check if levels[j] is parent of levels[i]
            # i.e., each value of levels[j] maps to exactly one value of levels[i]
            if is_parent(base_graph, levels[j], levels[i])
                if parent[i] == -1
                    parent[i] = j
                else
                    error("Level $(levels[i]) has multiple parents: "*
                          "$(levels[parent[i]]) and $(levels[j]). "*
                          "Hierarchy must be strict.")
                end
            end
        end
    end
    
    # Find roots (levels with no parent) and build ordering from coarse to fine
    ordered_indices = []
    visited = fill(false, length(levels))
    
    # Find all roots
    roots = findall(x -> x == -1, parent)
    
    if length(roots) != 1
        error("Hierarchy must have exactly one root level. "*
              "Found $(length(roots)) levels with no parent.")
    end
    
    # Topological sort from root
    function dfs_order(idx)
        if visited[idx]
            return
        end
        visited[idx] = true
        push!(ordered_indices, idx)
        
        # Find children of this level
        for i in 1:length(levels)
            if parent[i] == idx
                dfs_order(i)
            end
        end
    end
    
    dfs_order(roots[1])
    
    # Validate we got all nodes (no cycles or disconnected nodes)
    if length(ordered_indices) != length(levels)
        error("Hierarchy is not connected. Some levels are isolated or form cycles.")
    end
    
    # Reorder levels from coarse to fine
    ordered_levels = levels[ordered_indices]
    
    return MultiLevelGraph(base_graph, ordered_levels)
end

"""
    is_parent(base_graph, potential_parent, potential_child)::Bool

Return `true` if the level `potential_parent` is a (strict) parent of the level
`potential_child` in `base_graph`'s node attributes, i.e. `potential_child` is finer:
every distinct child value maps to exactly one parent value.
"""
function is_parent(base_graph::BaseGraph, potential_parent::String, potential_child::String)::Bool
    """
    Check if potential_parent level is a parent of potential_child level.
    
    A level A is parent of level B if each distinct value of B maps to exactly
    one value of A (B is finer than A).
    """
    
        child_to_parent = Dict()
    
    for node_attrs in values(base_graph.node_attributes)
        parent_val = node_attrs[potential_parent]
        child_val = node_attrs[potential_child]
        
            if haskey(child_to_parent, child_val)
            # Check consistency
                if child_to_parent[child_val] != parent_val
                return false  # One parent value maps to multiple child values
            end
        else
                child_to_parent[child_val] = parent_val
        end
    end
    
    return true 
end

"""
    modify_edge_weights!(graph, edge_weight_func)

Recompute and overwrite every edge weight of `graph` in place. For each edge,
`edge_weight_func(base_graph, src, dst)` is called and its return value is stored
both in the weighted adjacency matrix and in the edge-attribute table.

Two methods are provided: one for a [`BaseGraph`](@ref), and one for a
[`MultiLevelGraph`](@ref) that updates the underlying base graph and rebuilds the
per-level views so the multi-level structure stays consistent.
"""
function modify_edge_weights!(
    base_graph::BaseGraph,
    edge_weight_func::Function
)
    for e in edges(base_graph.simple_graph)
        new_weight = edge_weight_func(base_graph, src(e), dst(e))
        base_graph.simple_graph.weights[src(e), dst(e)] = new_weight
        base_graph.simple_graph.weights[dst(e), src(e)] = new_weight
        key = Set([src(e), dst(e)])
        @assert haskey(base_graph.edge_attributes, key)
        edge_weights = base_graph.edge_weights
        base_graph.edge_attributes[key][edge_weights] = new_weight
    end
end

function modify_edge_weights!( 
    graph::MultiLevelGraph,
    edge_weight_func::Function
)
    base_graph = graph.graphs_by_level[end]
    modify_edge_weights!(base_graph, edge_weight_func)
    g = MultiLevelGraph(base_graph, graph.levels)
    for level in 1:length(graph.levels)
        graph.graphs_by_level[level] = g.graphs_by_level[level]
        # the finest level has no coarse-to-fine graphs (its slot is unassigned)
        isassigned(graph.coarse_to_fine_graphs, level) || continue
        for ii in 1:length(graph.coarse_to_fine_graphs[level])
            new_graph = g.coarse_to_fine_graphs[level][ii]
            graph.coarse_to_fine_graphs[level][ii] = new_graph
        end
    end

    # Replace dictionary contents in place to avoid altering immutable struct.
    empty!(graph.mixed_nbr_weights)
    for (k, v) in g.mixed_nbr_weights
        graph.mixed_nbr_weights[k] = v
    end
end