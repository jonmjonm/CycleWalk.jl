"""
    MutableFloat

A mutable box holding a single `Float64` in `value`, so a running scalar (e.g. the
accumulated log importance weight during annealing) can be updated in place through
a closure or callback argument list.
"""
mutable struct MutableFloat
    value::Float64
end

"""
    set_all!(vec, value=0)

Recursively set every scalar entry of a (possibly nested) vector to `value`,
descending into inner vectors and leaving the nesting structure intact. Only
entries whose type matches `typeof(value)` are overwritten. Returns `vec`.
"""
function set_all!(
	vec::Vector, value::T = 0
) where {T <: Real}
	for ii = 1:length(vec)
		if typeof(vec[ii]) <: Vector
			vec[ii] = set_all!(vec[ii], value)
		elseif typeof(vec[ii]) <: T
			vec[ii] = value
		end
	end
    return vec
end

"""
    recursive_copy!(dst, src)

Deep-copy the contents of `src` into `dst` in place, recursing into nested vectors
so that `dst`'s inner buffers are reused rather than reallocated. Assumes `dst` and
`src` have identical nesting structure, lengths, and element types (asserted at each
level).
"""
function recursive_copy!(
	dst::Vector,
	src::Vector
)
	@assert length(dst) == length(src)
	for ii = 1:length(dst)
		if typeof(dst[ii]) <: Vector
			@assert typeof(dst[ii]) == typeof(src[ii])
			recursive_copy!(dst[ii], src[ii])
		else
			dst[ii] = src[ii]
		end
	end
end