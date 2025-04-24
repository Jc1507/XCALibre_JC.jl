using XCALibre

mesh_file = "Downloads/BFS.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = adapt(CUDABackend(), mesh)


d = 800e-3
ReTarget = 36000
nu = 1e-3
u_mag_cal = ReTarget*nu/d
u_mag = u_mag_cal
velocity = [u_mag, 0.0, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 3/2*(Tu*u_mag)^2
epsilon_inlet = k_inlet/(nuR*nu)
νt_inlet = k_inlet/ω_inlet
Re = d*u_mag_cal/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KOmega}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Symmetry(:extra, 0.0),
    Symmetry(:extra2, 0.0),
    Wall(:wall1, [0.0, 0.0, 0.0]),
    Wall(:wall2, [0.0, 0.0, 0.0]),
    Wall(:side, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0])
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall1, 0.0),
    Neumann(:top, 0.0),
    Neumann(:wall2, 0.0),
    Neumann(:side, 0.0),
    Neumann(:extra, 0.0),
    Neumann(:extra2, 0.0)
)

@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Symmetry(:extra, 0.0),
    Symmetry(:extra2, 0.0),
    KWallFunction(:wall1),
    KWallFunction(:wall2),
    KWallFunction(:side),
    KWallFunction(:top)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    Symmetry(:extra, 0.0),
    Symmetry(:extra2, 0.0),
    OmegaWallFunction(:wall1),
    OmegaWallFunction(:wall2),
    OmegaWallFunction(:side),
    OmegaWallFunction(:top)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, νt_inlet),
    Neumann(:outlet, 0.0),
    Symmetry(:extra, 0.0),
    Symmetry(:extra2, 0.0),
    NutWallFunction(:wall1),
    NutWallFunction(:wall2),
    NutWallFunction(:side),
    NutWallFunction(:top)
)

schemes = (
    U = set_schemes(time=Euler, divergence=Upwind),
    p = set_schemes(time=Euler, divergence=Upwind),
    k = set_schemes(time=Euler, divergence=Upwind),
    omega = set_schemes(time=Euler, divergence=Upwind)
)

solvers = (
    U = set_solver(
        model.momentum.U;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.7,
        rtol = 1e-2,
        atol = 1e-10
    ),
    p = set_solver(
        model.momentum.p;
        solver      = CgSolver, #GmresSolver, #CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-3,
        atol = 1e-10
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax       = 0.3,
        rtol = 1e-2,
        atol = 1e-10
    )
)

runtime = set_runtime(iterations=100000, write_interval=5000, time_step=0.4e-5)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.momentum.U, velocity)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, νt_inlet)

residuals = run!(model, config)