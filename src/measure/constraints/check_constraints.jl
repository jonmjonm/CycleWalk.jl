"""
    satisfies_constraints(partition, constraints, districts=...;
                          check_population=false, update=nothing)::Bool

Return `true` if `partition` (optionally as modified by `update`) satisfies every
constraint in `constraints` for the given `districts`. The population constraint is
only checked when `check_population=true`, since during sampling it is enforced
separately inside the proposal; the other constraints are always checked.
"""
function satisfies_constraints(
	partition::LinkCutPartition,
	constraints::Constraints,
	districts::Union{Tuple{Vararg{T}}, Vector{T}}
        = collect(1:partition.num_dists);
	check_population::Bool = false,
	update::Union{Update, Nothing}=nothing
)::Bool where T<:Int
	if check_population && !satisfies_constraint(partition, 
		                                    constraints.population_constraint,
		                                    districts; update=update)
		return false
	end
	
	for constraint in constraints.constraints
		if !satisfies_constraint(partition, constraint, districts;
                             update=update)
			return false
		end
	end
	return true
end
