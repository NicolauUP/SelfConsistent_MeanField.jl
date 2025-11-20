


abstract type AbstractDensitySolver end


struct KPM_Solver <: AbstractDensitySolver
    params::NamedTuple
end

struct ED_Solver <: AbstractDensitySolver
    params::NamedTuple
end

struct SP2_Solver <: AbstractDensitySolver
    params::NamedTuple
end