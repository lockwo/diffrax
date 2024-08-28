# SDE solvers

See also [How to choose a solver](../../usage/how-to-choose-a-solver.md#stochastic-differential-equations).

!!! info "Term structure"

    The type of solver chosen determines how the `terms` argument of `diffeqsolve` should be laid out.
    
    Most solvers handle both ODEs and SDEs in the same way, and expect a single term. So for an ODE you would pass `terms=ODETerm(vector_field)`, and for an SDE you would pass `terms=MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))`. For example:

    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))
    diffeqsolve(terms, solver=Euler(), ...)
    ```

    For any individual solver then this is documented below, and is also available programatically under `<solver>.term_structure`.

    For advanced users, note that we typically accept any `AbstractTerm` for the diffusion, so it could be a custom one that implements more-efficient behaviour for the structure of your diffusion matrix.

---

## Explicit Runge--Kutta (ERK) methods

These solvers can be used to solve SDEs just as well as they can be used to solve ODEs.

::: diffrax_extensions.Euler
    selection:
        members: false

::: diffrax_extensions.Heun
    selection:
        members: false

::: diffrax_extensions.Midpoint
    selection:
        members: false

::: diffrax_extensions.Ralston
    selection:
        members: false

!!! info

    In addition to the solvers above, then most higher-order ODE solvers can actually also be used as SDE solvers. They will typically converge to the Stratonovich solution. In practice this is computationally wasteful as they will not obtain more accurate solutions when applied to SDEs.

---

## SDE-only solvers

!!! info "Term structure"

    These solvers are SDE-specific. For these, `terms` must specifically be of the form `MultiTerm(ODETerm(...), SomeOtherTerm(...))` (Typically `SomeOTherTerm` will be a `ControlTerm` representing the drift and diffusion specifically.


::: diffrax_extensions.EulerHeun
    selection:
        members: false

::: diffrax_extensions.ItoMilstein
    selection:
        members: false

::: diffrax_extensions.StratonovichMilstein
    selection:
        members: false

### Stochastic Runge--Kutta (SRK)

These are a particularly important class of SDE-only solvers.

::: diffrax_extensions.SEA
    selection:
        members: false

::: diffrax_extensions.SRA1
    selection:
        members: false

::: diffrax_extensions.ShARK
    selection:
        members: false

::: diffrax_extensions.GeneralShARK
    selection:
        members: false

::: diffrax_extensions.SlowRK
    selection:
        members: false

::: diffrax_extensions.SPaRK
    selection:
        members: false

---

### Reversible methods

These are reversible in the same way as when applied to ODEs. [See here.](./ode_solvers.md#reversible-methods)

::: diffrax_extensions.ReversibleHeun
    selection:
        members: false

---

### Wrapper solvers

::: diffrax_extensions.HalfSolver
    selection:
        members:
            - __init__
