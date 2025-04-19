using XCALibre

mesh_file = "prototype/TestMesh/TurbulentPipeFlow.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)


# mesh_dev = adapt(CUDABackend(), mesh)
# Using Water
d = 0.01
velocity = [1, 0.0, 0.0]
nu = 1e-6
Re = d*velocity[1]/nu
cp = 4184.0 
gamma = 4.18
Pr = 6.9
Tu = 3.0
nuR = nu*Re
k_inlet = 3/2*(Tu*50)^2
epsilon_inlet = k_inlet/(nuR*nu)

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu,),
    turbulence = RANS{KEpsilon}(),
    energy = Energy{Isothermal}(),
    domain = mesh
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0])
)

 @assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall, 0.0),
    Neumann(:top, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    KWallFunction(:wall),
    KWallFunction(:top)
)

@assign! model turbulence epsilon (
    Dirichlet(:inlet, epsilon_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    OmegaWallFunction(:top)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet/epsilon_inlet),
    Neumann(:outlet, 0.0),
    NutWallFunction(:wall),
    NutWallFunction(:top)
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind),
    p = set_schemes(time=Euler, divergence=Upwind),
    k = set_schemes(time=Euler, divergence=Upwind),
    epsilon = set_schemes(time=Euler, divergence=Upwind)
)


solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
    ),
    p = set_solver(
        model.momentum.p;
        solver      = GmresSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    epsilon = set_solver(
        model.turbulence.epsilon;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.5,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=10000, write_interval=10, time_step=0.1e-5)

# hardware = set_hardware(backend=CUDABackend(), workgroup=32)
hardware = set_hardware(backend=CPU(), workgroup=cld(length(mesh.cells),4))

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, 3.84e-3)
initialise!(model.turbulence.epsilon, 3.059e-2)
initialise!(model.turbulence.nut, 3.84e-3/3.059e-2)

residuals = run!(model, config)
