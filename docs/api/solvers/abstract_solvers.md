# Abstract solvers

All of the solvers (both ODE and SDE solvers) implement the following interface specified by [`diffrax_extensions.AbstractSolver`][].

The exact details of this interface are only really useful if you're using the [Manual stepping](../../usage/manual-stepping.md) interface or defining your own solvers; otherwise this is all just internal to the library.

Also see [Extending Diffrax](../../usage/extending.md) for more information on defining your own solvers.

In addition [`diffrax_extensions.AbstractSolver`][] has several subclasses that you can use to mark your custom solver as exhibiting particular behaviour.

---

::: diffrax_extensions.AbstractSolver
    selection:
        members:
            - order
            - strong_order
            - error_order
            - init
            - step
            - func

---

::: diffrax_extensions.AbstractImplicitSolver
    selection:
        members:
          - __init__

---

::: diffrax_extensions.AbstractAdaptiveSolver
    selection:
        members: false

---

::: diffrax_extensions.AbstractItoSolver
    selection:
        members: false

---

::: diffrax_extensions.AbstractStratonovichSolver
    selection:
        members: false

---

::: diffrax_extensions.AbstractWrappedSolver
    selection:
        members:
            - __init__

---

### Abstract Runge--Kutta solvers

::: diffrax_extensions.AbstractRungeKutta
    selection:
        members: false

::: diffrax_extensions.AbstractERK
    selection:
        members: false

::: diffrax_extensions.AbstractDIRK
    selection:
        members: false

::: diffrax_extensions.AbstractSDIRK
    selection:
        members: false

::: diffrax_extensions.AbstractESDIRK
    selection:
        members: false

::: diffrax_extensions.ButcherTableau
    selection:
        members:
            - __init__

::: diffrax_extensions.CalculateJacobian
    selection:
        members: false

---

### Abstract Stochastic Runge--Kutta (SRK) solvers

::: diffrax_extensions.AbstractSRK
    selection:
        members: false

::: diffrax_extensions.StochasticButcherTableau
    selection:
        members:
            - __init__