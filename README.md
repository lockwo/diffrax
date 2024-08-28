<h1 align='center'>Diffrax Extensions</h1>
<h2 align='center'>(Extensions to) Diffrax: Numerical differential equation solvers in JAX. Autodifferentiable and GPU-capable.</h2>

Diffrax Extensions is a strict superset of [Diffrax](https://github.com/patrick-kidger/diffrax). Everything in diffrax is included in diffrax extensions (that is to say, any code you have `import diffrax` in, you can replace with `import diffrax_extensions as diffrax` and it will work). However, diffrax extensions (as the name implies) contains features beyond that of diffrax. These currently include:

- Additional SDE tooling

and will include:

- Additional solvers
- Additional levy area approximation

The following is required per the Apache 2.0 license. The original copyright notice of diffrax is available here: https://github.com/patrick-kidger/diffrax/blob/main/LICENSE#L189. A copy of the license is available in the `DIFFRAX_LICENSE` file. The significant changes made to the code that we are republishing with are available above. Diffrax does not contain a NOTICE file.

## Installation

```
pip install diffrax_extensions
```

Requires Python 3.9+, JAX 0.4.13+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.10.11+.

## Documentation

Available at [https://lockwo.github.io/diffrax_extensions](https://lockwo.github.io/diffrax_extensions).

## Why a fork?

Why maintain a fork of Diffrax, as opposed to strictly building on top of it as a dependency? There are several reasons:
- Features in diffrax_extensions may depend on changes to diffrax core, and we can shorten bottlenecks to deploying features, by rolling out changes in extensions while waiting for the changes to be made to core (rather than waiting for those changes to roll out our features that depend on them)
- Features will alter code within diffrax. Features will not strictly depend on the primitives diffrax provides, but also modify these primitives.
- Ease of interoperability. By forking diffrax main, and explicitly supersetting it, we enable diffrax_extensions to be drag and drop in older code bases, and trivial to use the new features of in research code.