# Optional human-readable names for proposal closures, surfaced in run metadata.
# The `build_*` proposal constructors register their returned closure here and
# [`describe_proposal`](@ref) reads it back. Proposals stay plain closures (rather than
# a wrapper type), so every proposal dispatch and the weighted-mixture element type are
# exactly as before; only the name lookup is added.
const _proposal_names = Base.IdDict{Any, String}()

"""
    name_proposal!(f, name) -> f

Tag proposal closure `f` with a human-readable `name` and return it unchanged. The name
is recorded in a run's Atlas header (see [`describe_proposal`](@ref)) so the output reads
e.g. `two_tree_cycle_walk` instead of the anonymous closure name `f`. Used by the
`build_*` proposal constructors; call it on your own closures to name custom proposals.
"""
name_proposal!(f, name::AbstractString) = (_proposal_names[f] = String(name); f)

"""
    proposal_name(f) -> String

Human-readable name registered for proposal `f` via [`name_proposal!`](@ref), falling
back to `string(f)` when none was registered.
"""
proposal_name(f) = get(_proposal_names, f, string(f))
