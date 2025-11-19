


abstract type Density_Solver end


struct KPM_Solver <: Density_Solver
    params::NamedTuple
end

struct ED_Solver <: Density_Solver
    params::NamedTuple
end

struct SP2_Solver <: Density_Solver
    params::NamedTuple
end