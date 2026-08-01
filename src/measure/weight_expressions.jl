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
    value = _walk_weight_ast(ast, expr, 1, counter) do name
        if !haskey(parameters, name)
            known = join(sort!(collect(String.(keys(parameters)))), ", ")
            throw(ArgumentError("the expression \"$expr\" refers to \"$name\", which " *
                                "is not a known parameter (known: $known)"))
        end
        value = parameters[name]
        value isa Real && !(value isa Bool) ||
            throw(ArgumentError("parameter \"$name\" is $(repr(value)), not a number"))
        float(value)
    end
    isa(value, Real) && isfinite(value) ||
        throw(ArgumentError("the expression \"$expr\" evaluates to $value, which is " *
                            "not a usable weight"))
    return Float64(value)
end

"""
    weight_path_closure(expr, parameters) -> Function   # t::Float64 -> Float64

Parse and validate `expr` ONCE — the same restricted grammar as
[`evaluate_weight_expression`](@ref) (only `+ - * / ^`, numeric literals, and named
parameters), with `t` additionally available as a bound name — and return a closure
evaluating it against `t` alone. `parameters` (e.g. `gamma`, `iso_weight`) are
resolved once, here, not re-looked-up on every call; the returned closure's only
per-call cost is walking the small pre-parsed tree with `t` bound. Used to build a
per-energy tempering schedule (`[[measure.energy]] weight_path = "..."`) without
re-parsing the expression once per round.

`t` is reserved for this purpose: a `[measure]` parameter named `t` is rejected by
[`measure_parameters`](@ref) before it would ever reach here.
"""
function weight_path_closure(expr::AbstractString, parameters::AbstractDict)
    ast = try
        Meta.parse(strip(expr))
    catch err
        throw(ArgumentError("could not parse the expression \"$expr\""))
    end
    resolve = function (t::Float64)
        counter = Ref(0)
        value = _walk_weight_ast(ast, expr, 1, counter) do name
            name == "t" && return t
            if !haskey(parameters, name)
                known = join(sort!(collect(String.(keys(parameters)))), ", ")
                throw(ArgumentError("the expression \"$expr\" refers to \"$name\", " *
                                    "which is not a known parameter (known: t, $known)"))
            end
            pval = parameters[name]
            pval isa Real && !(pval isa Bool) ||
                throw(ArgumentError("parameter \"$name\" is $(repr(pval)), not a number"))
            float(pval)
        end
        isa(value, Real) && isfinite(value) ||
            throw(ArgumentError("the expression \"$expr\" evaluates to $value at " *
                                "t=$t, which is not a usable weight"))
        return Float64(value)
    end
    resolve(0.0)   # validate once now (bad names / bad syntax fail at build time,
                   # not on the first round of a run); a value that is only finite
                   # away from t=0 still surfaces the moment that round is reached.
    return resolve
end

"""
    weight_expression_parameters(expr) -> Vector{String}

The names a weight expression reads, sorted and without repeats:
`weight_expression_parameters("2*gamma + gamma/iso_weight")` is
`["gamma", "iso_weight"]`. Refuses exactly what [`evaluate_weight_expression`](@ref)
refuses — it is the same walk over the same parsed tree, with the names collected
instead of looked up — so it can be called on an expression before any parameters
exist.

Used to decide which of a run's parameters actually reach its measure: a parameter no
expression names cannot change the target, and one that is named must appear in the
atlas file name, or two runs differing only in it would collide on one path.
"""
function weight_expression_parameters(expr::AbstractString)
    ast = try
        Meta.parse(strip(expr))
    catch err
        throw(ArgumentError("could not parse the expression \"$expr\""))
    end
    found = String[]
    _walk_weight_ast(ast, expr, 1, Ref(0)) do name
        name in found || push!(found, name)
        1.0                     # a stand-in value; only the names matter here
    end
    return sort!(found)
end

# Walk one AST node, calling `at_symbol(name)` for each name and combining the results
# with the allowed operators. Everything not explicitly allowed is rejected, so new
# Julia syntax cannot quietly become permissible. Evaluating and collecting names are
# the same traversal with different `at_symbol`s, which is what keeps the two from
# ever disagreeing about what an expression is allowed to contain.
function _walk_weight_ast(
    at_symbol::Function,
    node,
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
        return at_symbol(String(node))
    elseif node isa Expr && node.head === :call
        op = length(node.args) >= 1 ? node.args[1] : nothing
        # `Base.exit()` parses as a call whose callee is an Expr, not a Symbol, so
        # requiring a Symbol here also rules out every qualified name.
        (op isa Symbol && haskey(WEIGHT_EXPRESSION_OPS, op)) ||
            throw(ArgumentError("the expression \"$src\" uses \"$op\", which is not " *
                                "allowed (only + - * / ^)"))
        args = [_walk_weight_ast(at_symbol, a, src, depth+1, counter)
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
