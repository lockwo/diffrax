# Extending Diffrax

It's completely possible to extend Diffrax with your own custom solvers, step size controllers, and so on.

The main points of extension are as follows:

- **Custom solvers** should inherit from [`diffrax_extensions.AbstractSolver`][].
    - If you are making a new Runge--Kutta method then this is particularly easy; you can use the existing base classes Diffrax already uses for its own Runge--Kutta methods, and supply them with an appropriate `diffrax_extensions.ButcherTableau`.
        - For explicit-Runge--Kutta methods (ERK) then inherit from `diffrax_extensions.AbstractERK`.
        - For general diagonal-implicit-Runge--Kutta methods (DIRK) then inherit from `diffrax_extensions.AbstractDIRK`.
        - For singly-diagonal-implicit-Runge--Kutta methods (SDIRK) then inherit from `diffrax_extensions.AbstractSDIRK`.
        - For explicit-singly-diagonal-implicit-Runge--Kutta methods (ESDIRK) then inherit from `diffrax_extensions.AbstractESDIRK`.
    - Several abstract base classes are available to specify the behaviour of the solver:
        - `diffrax_extensions.AbstractImplicitSolver` are those solvers that solve implicit problems (and therefore take a `root_finder` argument).
        - `diffrax_extensions.AbstractAdaptiveSolver` are those solvers capable of providing error estimates (and thus can be used with adaptive step size controllers).
        - `diffrax_extensions.AbstractItoSolver` and `diffrax_extensions.AbstractStratonovichSolver` are used to specify which SDE solution a particular solver is known to converge to.
        - `diffrax_extensions.AbstractWrappedSolver` indicates that the solver is used to wrap another solver, and so e.g. it will be treated as an implicit solver/etc. if the wrapped solver is also an implicit solver/etc.

- **Custom step size controllers** should inherit from [`diffrax_extensions.AbstractStepSizeController`][].
    - The abstract base class `diffrax_extensions.AbstractAdaptiveStepSizeController` can be used to specify that this controller uses error estimates to adapt step sizes.

- **Custom Brownian motion simulations** should inherit from [`diffrax_extensions.AbstractBrownianPath`][].

- **Custom controls** (e.g. **custom interpolation schemes** analogous to [`diffrax_extensions.CubicInterpolation`][]) should inherit from [`diffrax_extensions.AbstractPath`][].

- **Custom terms** should inherit from [`diffrax_extensions.AbstractTerm`][].
    - For example, if the vector field - control interaction is a matrix-vector product, but the matrix is known to have special structure, then you may wish to create a custom term that can calculate this interaction more efficiently than is given by a full matrix-vector product. Given the large suite of linear operators [lineax](https://docs.kidger.site/lineax/) implements (which are fully supported by [`diffrax_extensions.ControlTerm`][]), this is likely rarely necessary.

In each case we recommend looking up existing solvers/etc. in Diffrax to understand how to implement them.

!!! tip "Contributions"

    If you implement a technique that you'd like to see merged into the main Diffrax library then open a [pull request on GitHub](https://github.com/patrick-kidger/diffrax). We're very happy to take contributions.

    Also, if you implement a technique that you'd like to see merged into the main Diffrax Extensions library then open a [pull request on GitHub](https://github.com/lockwo/diffrax_extensions). We're very happy to take contributions.
