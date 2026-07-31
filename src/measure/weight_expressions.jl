# Arithmetic over named run parameters, for weights that are written in a config file
# rather than in code (see the `[measure.weights]` table in examples/parameterUtils.jl).
#
# A config's weight can be a plain number, or an expression over the run's named
# parameters — "gamma", "2*gamma + 1", "-iso_weight/2" — so a second energy can ride
# the same swept knob without a new config key.
#
# These strings arrive from files that are read back out of Atlases written by other
# people, so they are *interpreted*, never `eval`ed: `Meta.parse` builds an inert AST
# and the walker below refuses every node that is not arithmetic over known parameters.
# Parsing alone runs nothing; handing the result to `eval` would run anything.

"""
    WEIGHT_EXPRESSION_OPS

The only operations a weight expression may use (see
[`evaluate_weight_expression`](@ref)). Deliberately arithmetic and nothing else: every
addition widens both what a config can express and what someone reading a stranger's
config has to reason about.
"""
const WEIGHT_EXPRESSION_OPS = Dict{Symbol, Function}(
    :+ => +, :- => -, :* => *, :/ => /, :^ => ^)

"""
    MAX_WEIGHT_EXPRESSION_DEPTH

How deeply a weight expression may nest before it is rejected (see
[`evaluate_weight_expression`](@ref)).
"""
const MAX_WEIGHT_EXPRESSION_DEPTH = 16

"""
    MAX_WEIGHT_EXPRESSION_NODES

How many nodes a weight expression may contain before it is rejected (see
[`evaluate_weight_expression`](@ref)).
"""
const MAX_WEIGHT_EXPRESSION_NODES = 100

"""
    evaluate_weight_expression(expr, parameters) -> Float64

Evaluate the arithmetic expression `expr` over the named `parameters`, a `Dict` from
name to number. Used for measure weights written in a config file:

```julia
evaluate_weight_expression("2*gamma + 1", Dict("gamma" => 1.5))   # 4.0
```

Only `+ - * / ^`, numeric literals, and names present in `parameters` are permitted;
anything else — a function call, an index, a field access, a second statement, an
unknown name — is an `ArgumentError` naming what was rejected. The expression is
parsed to an AST and then *interpreted* by this function; it is never `eval`ed, so a
string like `"run(\\`rm -rf /\\`)"` is refused at inspection time rather than run. That
matters because these expressions come from config files, which CycleWalk reads back
out of Atlas headers written elsewhere.

Nesting deeper than [`MAX_WEIGHT_EXPRESSION_DEPTH`](@ref) or larger than
[`MAX_WEIGHT_EXPRESSION_NODES`](@ref) nodes is rejected, as is a result that is not a
finite real number — an infinite or `NaN` weight would silently change the target
rather than fail.
"""
function evaluate_weight_expression(
    expr::AbstractString,
    parameters::AbstractDict
)::Float64
    ast = try
        Meta.parse(strip(expr))
    catch err
        throw(ArgumentError("could not parse the expression \"$expr\""))
    end
    counter = Ref(0)
    value = _eval_weight_node(ast, parameters, expr, 1, counter)
    isa(value, Real) && isfinite(value) ||
        throw(ArgumentError("the expression \"$expr\" evaluates to $value, which is " *
                            "not a usable weight"))
    return Float64(value)
end

# Interpret one AST node. Everything not explicitly allowed is rejected, so new Julia
# syntax cannot quietly become permissible.
function _eval_weight_node(
    node,
    parameters::AbstractDict,
    src::AbstractString,
    depth::Int,
    counter::Ref{Int}
)
    depth > MAX_WEIGHT_EXPRESSION_DEPTH &&
        throw(ArgumentError("the expression \"$src\" nests deeper than " *
                            "$MAX_WEIGHT_EXPRESSION_DEPTH"))
    counter[] += 1
    counter[] > MAX_WEIGHT_EXPRESSION_NODES &&
        throw(ArgumentError("the expression \"$src\" has more than " *
                            "$MAX_WEIGHT_EXPRESSION_NODES terms"))

    if node isa Bool
        # `true` is a Real in Julia; a boolean weight is a config mistake, not a 1.0
        throw(ArgumentError("the expression \"$src\" uses a boolean"))
    elseif node isa Real
        return float(node)
    elseif node isa Symbol
        name = String(node)
        if !haskey(parameters, name)
            known = join(sort!(collect(String.(keys(parameters)))), ", ")
            throw(ArgumentError("the expression \"$src\" refers to \"$name\", which " *
                                "is not a known parameter (known: $known)"))
        end
        value = parameters[name]
        value isa Real && !(value isa Bool) ||
            throw(ArgumentError("parameter \"$name\" is $(repr(value)), not a number"))
        return float(value)
    elseif node isa Expr && node.head === :call
        op = length(node.args) >= 1 ? node.args[1] : nothing
        # `Base.exit()` parses as a call whose callee is an Expr, not a Symbol, so
        # requiring a Symbol here also rules out every qualified name.
        (op isa Symbol && haskey(WEIGHT_EXPRESSION_OPS, op)) ||
            throw(ArgumentError("the expression \"$src\" uses \"$op\", which is not " *
                                "allowed (only + - * / ^)"))
        args = [_eval_weight_node(a, parameters, src, depth+1, counter)
                for a in node.args[2:end]]
        return try
            WEIGHT_EXPRESSION_OPS[op](args...)
        catch err
            throw(ArgumentError("the expression \"$src\" applies \"$op\" to " *
                                "$(length(args)) arguments"))
        end
    else
        # :toplevel (a second statement), :ref, :., :(=), :macrocall, strings, …
        what = node isa Expr ? string(node.head) : string(typeof(node))
        throw(ArgumentError("the expression \"$src\" contains $what, which is not " *
                            "allowed (only arithmetic over named parameters)"))
    end
end
