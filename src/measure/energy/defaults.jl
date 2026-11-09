"""
    update_energy_data!(eData::AbstractEnergyData, partition, update)

Default fallback invoked by `update_partition!` for every cache after a move is
accepted. Does nothing; energy caches that maintain incremental state (e.g.
[`LogForestEnergyData`](@ref)) override this to commit the changed districts' proposed
values into their synced fields.
"""
function update_energy_data!(
	eData::ED,
	partition::LinkCutPartition,
	update::Update{T}
) where {ED <: AbstractEnergyData, T<:Int}
	# @show "in default energy data", typeof(eData)
	# by default, do nothing with the energy data unless the update is specified 
	return
end

"""
    reset_update_buffer!(eData::AbstractEnergyData)

Invalidate any proposed (staged) values held by `eData` so the next query with an
`update` recomputes them, leaving the current values untouched. Called on every cache
before a proposal's constraints are checked, so a staged buffer left by an earlier
proposal from the same state can never be mistaken for the new one.

Default fallback: does nothing. This is correct for caches that overwrite the staged
values of the changed districts on every `update` query (e.g.
[`LogForestEnergyData`](@ref)). A cache that skips recomputation when the `update`
matches the one it last staged (e.g. [`BalanceData`](@ref), keyed on
`update.identifier`, which does not determine the labeling) must override this to
drop that key.
"""
function reset_update_buffer!(eData::AbstractEnergyData)
    return
end
