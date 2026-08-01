# =============================================================================
# Parallel tempering — advance backends
#
# See docs/parallel_tempering_implementation.md (tasks 4, 7, 9). `SerialBackend` is a
# plain loop and is needed by the driver itself (task 4); `ThreadedBackend` (task 7)
# and `DistributedBackend` (task 9) are a performance layer on top of the same
# `advance!` seam and must reproduce `SerialBackend` bit for bit.
# =============================================================================

abstract type PTBackend end

"""
    advance!(backend, ensemble, steps)

Advance every replica `steps` MH steps at its current β. Returns nothing. Implementations
must leave `ensemble` otherwise untouched.
"""
function advance! end

backend_name(::PTBackend) = "unknown"
backend_workers(::PTBackend) = 1
backend_straggler_gap(::PTBackend) = 0.0

"""
    advance_replica!(replica, ensemble, proposal, work_measure, steps)

Advance one replica `steps` MH steps at the measure configured to ITS OWN rung
(`ensemble.lattice[replica.beta_index]`). `work_measure` is mutated in place by
`configure_measure!`, so it must be private to this replica/worker (see the module
invariants) — never share one `work_measure` across replicas advanced concurrently.
"""
function advance_replica!(replica::Replica, ensemble::PTEnsemble, proposal,
                          work_measure::Measure, steps::Int)
    configure_measure!(ensemble.path, work_measure, ensemble.scores,
                       ensemble.lattice[replica.beta_index])
    run_metropolis_hastings!(replica.state, proposal, work_measure, steps, replica.rng;
                             output_initial=false, run_diagnostics=replica.diagnostics)
end

"""
    SerialBackend(proposal)

Advance every replica in a plain loop on the calling thread. The reference
implementation every other backend must reproduce bit for bit (§7.1 test 5).
"""
struct SerialBackend{P} <: PTBackend
    proposal::P
end

backend_name(::SerialBackend) = "serial"

function advance!(b::SerialBackend, e::PTEnsemble, steps::Int)
    for replica in e.replicas
        advance_replica!(replica, e, b.proposal, replica.work_measure, steps)
    end
    return nothing
end

"""
    ThreadedBackend(proposal, M)

Advance every replica with one `Threads.@spawn` task per replica (`M` = number of
rungs, needed up front only to size the per-replica timing buffer used for the
straggler-gap diagnostic), then block until all finish. Dynamic scheduling absorbs
the load imbalance between rungs (accepted moves cost more than rejected ones, and
acceptance varies with β) at negligible spawn overhead relative to a replica-block
(design §1.5) — give the process threads with `julia -t N`.

Must reproduce [`SerialBackend`](@ref) bit for bit for the same seed (§7.1 test 5):
replica rngs are seeded sequentially from the driver's rng regardless of backend (see
the module invariants), and each replica's state/rng/diagnostics/work_measure are
never shared, so which thread runs which replica cannot affect the result. Never
index anything by `Threads.threadid()` — index by replica id, as below.
"""
mutable struct ThreadedBackend{P} <: PTBackend
    proposal::P
    times::Vector{Float64}
    last_gap::Float64
end
ThreadedBackend(proposal::P, M::Int) where {P} =
    ThreadedBackend{P}(proposal, zeros(Float64, M), 0.0)

backend_name(::ThreadedBackend) = "threaded"
backend_workers(::ThreadedBackend) = Threads.nthreads()
backend_straggler_gap(b::ThreadedBackend) = b.last_gap

function advance!(b::ThreadedBackend, e::PTEnsemble, steps::Int)
    length(b.times) == length(e.replicas) ||
        throw(ArgumentError("ThreadedBackend was sized for $(length(b.times)) " *
                            "replicas but the ensemble has $(length(e.replicas))"))
    times = b.times
    tasks = map(1:length(e.replicas)) do w
        Threads.@spawn begin
            t0 = time()
            advance_replica!(e.replicas[w], e, b.proposal, e.replicas[w].work_measure, steps)
            times[w] = time() - t0        # indexed by REPLICA, never threadid()
        end
    end
    foreach(wait, tasks)
    b.last_gap = maximum(times) - sum(times)/length(times)
    return nothing
end
