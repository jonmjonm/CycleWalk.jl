# Derived node columns — computing a new node attribute from existing ones (e.g. a
# unique node name joined from columns that aren't unique alone, or a population
# adjusted by another) via the same safe expression grammar `weight`/`weight_start`
# already use (see `evaluate_column_expression` in `measure/weight_expressions.jl`).
#
# This exists at the graph layer, not the measure layer, because it has to run before
# `MultiLevelGraph` is built: `geo_units` (the node-naming column `MultiLevelGraph`
# keys everything by) has to already exist as an ordinary node attribute by the time
# the graph is constructed, and `BaseGraph`/`MultiLevelGraph` themselves come from
# MetropolizedForestRecom, a separate package — this writes into
# `BaseGraph.node_attributes` (an ordinary `Vector{Dict{String,Any}}`) beforehand
# rather than changing how that package names nodes.

"""
    derive_node_columns!(base_graph, specs) -> base_graph

For each `name => expr` pair in `specs` (order matters — see below), evaluate `expr`
([`evaluate_column_expression`](@ref)) against every node's own attributes and store
the result under `name`, mutating `base_graph.node_attributes` in place. Once done,
`name` is an ordinary node attribute — usable as `geo_units`, `pop_col`, a
constraint's unit column, or an energy's column argument, exactly like a raw one,
because to everything downstream it is one.

```julia
derive_node_columns!(base_graph, OrderedDict(
    "uid"     => "county * \"_\" * prec_id",   # a unique node name from two that aren't
    "adj_pop" => "pop2020cen - prison_pop",     # a population adjustment
))
```

Each `expr` may reference only RAW columns already on the graph, not another entry
of `specs` — `specs` is evaluated independently per name, in the order given, so one
derived column depending on another would silently depend on dict/table iteration
order rather than being resolved deliberately. Pass an `OrderedDict` (or a
`Vector{Pair}`) if the order of `name => expr` pairs matters to you for any other
reason (e.g. which of two colliding names' errors surfaces first); plain `Dict`
iteration order is otherwise unspecified.

Refuses to silently shadow an existing column: `name` colliding with a column
already on the graph (raw, or derived by an earlier entry of `specs`) is an
`ArgumentError` naming the collision, since overwriting a raw column by accident
would be very hard to notice afterward.
"""
function derive_node_columns!(base_graph::BaseGraph, specs)
    for (name, expr) in specs
        name isa AbstractString || throw(ArgumentError(
            "derived column name $(repr(name)) is not a string"))
        expr isa AbstractString || throw(ArgumentError(
            "derived column \"$name\"'s expression $(repr(expr)) is not a string"))
        if !isempty(base_graph.node_attributes) && haskey(base_graph.node_attributes[1], name)
            throw(ArgumentError("derived column \"$name\" collides with a column " *
                                "already on the graph (raw, or derived earlier in " *
                                "this same list) — pick a different name"))
        end
        for attrs in base_graph.node_attributes
            attrs[name] = evaluate_column_expression(expr, attrs)
        end
    end
    return base_graph
end
