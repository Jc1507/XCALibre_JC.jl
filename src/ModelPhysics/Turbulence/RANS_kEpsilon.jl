export KEpsilon

"""
    KEpsilon <: AbstractTurbulenceModel

kEpsilon model containing all kEpsilon field parameters.

### Fields
- 'k' -- Turbulent kinetic energy ScalarField.
- 'epsilon' -- Specific dissipation rate ScalarField.
- 'nut' -- Eddy viscosity ScalarField.
- 'kf' -- Turbulent kinetic energy FaceScalarField.
- 'epsilonf' -- Specific dissipation rate FaceScalarField.
- 'nutf' -- Eddy viscosity FaceScalarField.
- 'coeffs' -- Model coefficients.

"""


struct KEpsilon{S1, S2, S3, F1, F2, F3, C} <: AbstractRANSModel
    k::S1
    epsilon::S2
    nut::S3
    kf::F1
    epsilonf::F2
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KEpsilon

struct KEpsilonModel{E1, E2}
    k_eqn::E1 
    epsilon_eqn::E2
end
Adapt.@adapt_structure KEpsilonModel

RANS{KEpsilon}(; Cμ=0.09, C1ε=1.44, C2ε=1.92, σk=1.0, σε=1.3) = begin 
    coeffs = (Cμ=Cμ, C1ε=C1ε, C2ε=C2ε, σk=σk, σε=σε)
    ARG = typeof(coeffs)
    RANS{KEpsilon, ARG}(coeffs)
end

# Model initialisation
"""
    initialise(turbulence::KEpsilon, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `KEpsilonModel(k_eqn, e_eqn)`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KEpsilon, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
) where {T,F,M,Tu,E,D,BI}

    (; k, epsilon, nut) = turbulence
    (; rho) = model.fluid
    (; solvers, schemes, runtime) = config
    mesh = mdotf.mesh
    eqn = peqn.equation

    # define fluxes and sources for k and epsilon
    mueffk = FaceScalarField(mesh)
    mueffe = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Depsilonf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pe = ScalarField(mesh)
    Epsilon(k) = C1ε*(Pk)^2/k-C2ε*ε
    
    # k equation
    k_eqn = (
    Time{schemes.k.time}(rho, k)  # Time derivative of k
    + Divergence{schemes.k.divergence}(mdotf, k)  # Convective transport term
    - Laplacian{schemes.k.laplacian}(mueffk, k)  # Diffusive transport term
    + Pk  # Production term (Pk remains unchanged)
    - Epsilon(k)  # Dissipation term (epsilon, related to k and turbulence characteristics)
    ==
    Source(Pk)  # Source term remains unchanged
) → eqn


    # epsilon equation
    epsilon_eqn = (
    Time{schemes.epsilon.time}(rho, epsilon)  # Time derivative of epsilon
    + Divergence{schemes.epsilon.divergence}(mdotf, epsilon)  # Convective transport term
    - Laplacian{schemes.epsilon.laplacian}(mueffe, epsilon)  # Diffusion term
    + Si(Depsilonf, epsilon)  # Source term involving Dεf = rho * C₁ε * k / ε
    ==
    Source(Pe)  # Source term for epsilon
) → eqn


    # Set up preconditioners
    @reset k_eqn.preconditioner = set_preconditioner(
        solvers.k.preconditioner, k_eqn, k.BCs, config)
    @reset epsilon_eqn.preconditioner = k_eqn.preconditioner

    # Preallocate solvers
    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    @reset epsilon_eqn.solver = solvers.epsilon.solver(_A(epsilon_eqn), _b(epsilon_eqn))

    return KEpsilonModel(k_eqn, epsilon_eqn)
end

# Model solver call (implementation)
"""
    turbulence!(rans::KEpsilonModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:KEpsilon,E,D,BI,E1,E2}

Run turbulence model transport equations.

### Input
- `rans::KEpsilonModel{E1,E2}` -- KEpsilon turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""

function turbulence!(
    rans::KEpsilonModel{E1,E2}, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
) where {T,F,M,Tu<:KEpsilon,E,D,BI,E1,E2}

    mesh = model.domain
    
    (; rho, rhof, nu, nuf) = model.fluid
    (;k, epsilon, nut, kf, epsilonf, nutf, coeffs) = model.turbulence
    (;k_eqn, epsilon_eqn) = rans
    (; solvers, runtime) = config

    # Get fluxes and sources
    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    Pk = get_source(k_eqn, 1)

    mueffe = get_flux(epsilon_eqn, 3)
    Depsilonf = get_flux(epsilon_eqn, 4)
    Pe = get_source(epsilon_eqn, 1)

    # Update fluxes and sources
    magnitude2!(Pk, S, config, scale_factor=2.0)
    constrain_boundary!(epsilon, epsilon.BCs, model, config)
    correct_production!(Pk, k.BCs, model, S.gradU, config)
    @. Pe.values = rho.values * coeffs.C1ε * Pk.values
    @. Pk.values = rho.values * nut.values * Pk.values
    @. Depsilonf.values = rho.values * coeffs.C2ε * epsilon.values
    @. mueffe.values = rhof.values * (nuf.values + coeffs.σε * nutf.values)
    @. Depsilonf.values = rho.values * coeffs.Cμ * epsilon.values
    @. mueffk.values = rhof.values * (nuf.values + coeffs.σk * nutf.values)

    # Solve epsilon equation
    prev .= epsilon.values
    discretise!(epsilon_eqn, epsilon, config)
    apply_boundary_conditions!(epsilon_eqn, epsilon.BCs, nothing, time, config)
    update_preconditioner!(epsilon_eqn.preconditioner, mesh, config)
    solve_system!(epsilon_eqn, solvers.epsilon, epsilon, nothing, config)

    # Solve k equation
    prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, time, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    solve_system!(k_eqn, solvers.k, k, nothing, config)

    @. nut.values = k.values / epsilon.values

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)
end

# Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
    ) where {T,F,M,Tu<:KEpsilon,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("epsilon", model.turbulence.epsilon),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("epsilon", model.turbulence.epsilon),
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end

"done"