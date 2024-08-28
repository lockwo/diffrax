# Progress meters

As the solve progresses, progress meters offer the ability to have some kind of output indicating how far along the solve has progressed. For example, to display a text output every now and again, or to fill a [tqdm](https://github.com/tqdm/tqdm) progress bar.

??? abstract "`diffrax_extensions.AbstractProgressMeter`"

    ::: diffrax_extensions.AbstractProgressMeter
        selection:
            members:
                - init
                - step
                - close

---

::: diffrax_extensions.NoProgressMeter
    selection:
        members:
            - __init__

::: diffrax_extensions.TextProgressMeter
    selection:
        members:
            - __init__

::: diffrax_extensions.TqdmProgressMeter
    selection:
        members:
            - __init__
