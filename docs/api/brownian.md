# Brownian controls

SDEs are simulated using a Brownian motion as a control. (See the neural SDE example.)

??? abstract "`diffrax_extensions.AbstractBrownianPath`"

    ::: diffrax_extensions.AbstractBrownianPath
        selection:
            members:
                - evaluate

---

::: diffrax_extensions.UnsafeBrownianPath
    selection:
        members:
            - __init__
            - evaluate

::: diffrax_extensions.VirtualBrownianTree
    selection:
        members:
            - __init__
            - evaluate

---

## Lévy areas

Brownian controls can return certain types of Lévy areas. These are iterated integrals
of the Brownian motion, and are used by some SDE solvers. When a solver requires a 
Lévy area, it will have a `minimal_levy_area` attribute, which will always return an
abstract Lévy area type, and it can accept any subclass of that type.
The inheritance hierarchy is as follows:
```
AbstractBrownianIncrement
│   └── BrownianIncrement
└── AbstractSpaceTimeLevyArea
    │   └── SpaceTimeLevyArea
    └── AbstractSpaceTimeTimeLevyArea
            └── SpaceTimeTimeLevyArea
```
For example if `solver.minimal_levy_area` returns an `AbstractSpaceTimeLevyArea`, then
the Brownian motion (which is either an `UnsafeBrownianPath` or 
a `VirtualBrownianTree`) should be initialized with `levy_area=SpaceTimeLevyArea` or 
`levy_area=SpaceTimeTimeLevyArea`. Note that for the Brownian motion,
a concrete class must be used, not its abstract parent.

::: diffrax_extensions.AbstractBrownianIncrement
    selection:
        members: false

::: diffrax_extensions.BrownianIncrement
    selection:
        members: false

::: diffrax_extensions.AbstractSpaceTimeLevyArea
    selection:
        members: false

::: diffrax_extensions.SpaceTimeLevyArea
    selection:
        members: false

::: diffrax_extensions.AbstractSpaceTimeTimeLevyArea
    selection:
        members: false

::: diffrax_extensions.SpaceTimeTimeLevyArea
    selection:
        members: false