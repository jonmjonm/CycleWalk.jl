using RandomNumbers
using CycleWalk
using IOBuffer


function generate_grid_graph_json(n::Int)
    nodes = []
    # For assigning counties, alternate 'A' and 'B' for each side as in your example
    for x = 0:n-1
        for y = 0:n-1
            border_length = 0
            # Border on edge? Add 1 for each side that's a border
            border_length += x == 0 ? 1 : 0
            border_length += x == n-1 ? 1 : 0
            border_length += y == 0 ? 1 : 0
            border_length += y == n-1 ? 1 : 0
            node = Dict(
                "node_name" => "($(x),$(y))",
                "id" => x * n + y,
                "border_length" => border_length,
                "x_location" => x,
                "y_location" => y,
                "area" => 1,
                "population" => 1,
                "county" => x < n÷2 ? "A" : "B"
            )
            push!(nodes, node)
        end
    end

    adjacency = [ [] for _ in 1:n*n ]
    for x = 0:n-1
        for y = 0:n-1
            id = x * n + y
            adj = []
            for (dx, dy) in ((-1,0),(1,0),(0,-1),(0,1))
                nx, ny = x+dx, y+dy
                if 0 ≤ nx < n && 0 ≤ ny < n
                    push!(adj, Dict("id"=>nx*n+ny, "length"=>1))
                end
            end
            adjacency[id+1] = adj
        end
    end

    graph_dict = Dict(
        "directed" => false,
        "multigraph" => false,
        "graph" => [],
        "nodes" => nodes,
        "adjacency" => adjacency
    )
    return JSON.json(graph_dict,4)
end
