


abstract type AbstractDensitySolver end


struct KPM_Solver <: AbstractDensitySolver
    params::NamedTuple
end

struct ED_Solver <: AbstractDensitySolver
    params::NamedTuple
end

struct Purification_Solver <: AbstractDensitySolver
    params::NamedTuple
end