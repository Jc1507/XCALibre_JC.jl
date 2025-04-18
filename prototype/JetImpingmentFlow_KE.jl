using XCALibre
using CUDA

# mesh_file = "examples/0_GRIDS/backwardFacingStep_5mm.unv"
mesh_file = "prototype/TestMesh/JetImpingment5.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = adapt(CUDABackend(), mesh)
# mesh_dev = mesh

d = 101.6e-3
ReTarget = 70000
nu = 1e-5
u_mag_cal = ReTarget*nu/d
u_mag = -u_mag_cal
# u_mag = 3.5 # 2mm mesh
velocity = [0.0, u_mag, 0.0]
Tu = 0.05
nuR = 100
k_inlet = 3/2*(Tu*u_mag)^2
epsilon_inlet = k_inlet/(nuR*nu)
Re = d*u_mag_cal/nu

model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu = nu),
    turbulence = RANS{KEpsilon}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )

@assign! model momentum U (
        Dirichlet(:inlet, velocity),
        Neumann(:outlet1, 0.0),
        Neumann(:outlet2, 0.0),
        Wall(:wall, [0.0, 0.0, 0.0]),
        # Dirichlet(:wall, [0.0, 0.0, 0.0]),
        Neumann(:top1, 0.0),
        Neumann(:top2, 0.0),
        Wall(:side1, [0.0, 0.0, 0.0]),
        Wall(:side2, [0.0, 0.0, 0.0])
    )
    
    @assign! model momentum p (
        Neumann(:inlet, 0.0),
        Dirichlet(:outlet1, 0.0),
        Dirichlet(:outlet2, 0.0),
        Neumann(:wall, 0.0),
        Dirichlet(:top1, 0.0),
        Dirichlet(:top2, 0.0),
        Neumann(:side1, 0.0),
        Neumann(:side2, 0.0)
    )
    
    @assign! model turbulence k (
        Dirichlet(:inlet, k_inlet),
        Neumann(:outlet1, 0.0),
        Neumann(:outlet2, 0.0),
        KWallFunction(:wall),
        # Neumann(:wall, 0.0),
        Neumann(:top1, 0.0),
        Neumann(:top2, 0.0),
        KWallFunction(:side1),
        KWallFunction(:side2)
    )
    
    @assign! model turbulence epsilon (
        Dirichlet(:inlet, epsilon_inlet),
        Neumann(:outlet1, 0.0),
        Neumann(:outlet2, 0.0),
        EpsilonWallFunction(:wall),
        Neumann(:top1, 0.0),
        Neumann(:top2, 0.0),
        EpsilonWallFunction(:side1),
        EpsilonWallFunction(:side2)
    )

    @assign! model turbulence nut (
        Dirichlet(:inlet, k_inlet/epsilon_inlet),
        Neumann(:outlet1, 0.0),
        Neumann(:outlet2, 0.0),
        NutWallFunction(:wall), 
        Neumann(:top1, 0.0),
        Neumann(:top2, 0.0),
        NutWallFunction(:side1),
        NutWallFunction(:side2)
    )

schemes = (
    U = set_schemes(divergence=Upwind, time=Euler),
    p = set_schemes(divergence=Upwind, time=Euler),
    k = set_schemes(divergence=Upwind, time=Euler),
    epsilon = set_schemes(divergence=Upwind, time=Euler)
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
        relax       = 0.5,
        rtol = 1e-3,
        atol = 1e-10
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

runtime = set_runtime(iterations=10000, write_interval=5, time_step=0.00003)
# runtime = set_runtime(iterations=2, write_interval=-1, time_step=1)

hardware = set_hardware(backend=CUDABackend(), workgroup=32)
# hardware = set_hardware(backend=CPU(), workgroup=4)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


GC.gc()

initialise!(model.momentum.U, [0.0, 0.0, 0.0])
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.epsilon, epsilon_inlet)
initialise!(model.turbulence.nut, k_inlet/epsilon_inlet)

residuals = run!(model, config)