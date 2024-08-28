import diffrax_extensions as diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import Array, PyTree, Shaped

from .helpers import tree_allclose


def test_ode_term():
    def vector_field(t, y, args) -> Array:
        return -y

    term = diffrax.ODETerm(vector_field)
    dt = term.contr(0, 1)
    vf = term.vf(0, 1, None)
    vf_prod = term.vf_prod(0, 1, None, dt)
    assert tree_allclose(vf_prod, term.prod(vf, dt))

    # `# type: ignore` is used for contrapositive static type checking as per:
    # https://github.com/microsoft/pyright/discussions/2411#discussioncomment-2028001
    _: diffrax.ODETerm[Array] = term
    __: diffrax.ODETerm[bool] = term  # type: ignore


def test_control_term(getkey):
    vector_field = lambda t, y, args: jr.normal(args, (3, 2))
    derivkey = getkey()

    class Control(diffrax.AbstractPath[Shaped[Array, "2"]]):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1=None, left=True):
            return jr.normal(getkey(), (2,))

        def derivative(self, t, left=True):
            return jr.normal(derivkey, (2,))

    control = Control()
    term = diffrax.ControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    y = jnp.array([1.0, 2.0, 3.0])
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    if isinstance(dx, jax.Array) and isinstance(vf, jax.Array):
        assert dx.shape == (2,)
        assert vf.shape == (3, 2)
    else:
        raise TypeError("dx/vf is not an array")
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dx))

    # `# type: ignore` is used for contrapositive static type checking as per:
    # https://github.com/microsoft/pyright/discussions/2411#discussioncomment-2028001
    _: diffrax.ControlTerm[PyTree[Array], Array] = term
    __: diffrax.ControlTerm[PyTree[Array], diffrax.BrownianIncrement] = term  # type: ignore

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dt))


def test_weakly_diagional_control_term(getkey):
    vector_field = lambda t, y, args: jr.normal(args, (3,))
    derivkey = getkey()

    class Control(diffrax.AbstractPath):
        t0 = 0
        t1 = 1

        def evaluate(self, t0, t1=None, left=True):
            return jr.normal(getkey(), (3,))

        def derivative(self, t, left=True):
            return jr.normal(derivkey, (3,))

    control = Control()
    term = diffrax.WeaklyDiagonalControlTerm(vector_field, control)
    args = getkey()
    dx = term.contr(0, 1)
    y = jnp.array([1.0, 2.0, 3.0])
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dx)
    if isinstance(dx, jax.Array) and isinstance(vf, jax.Array):
        assert dx.shape == (3,)
        assert vf.shape == (3,)
    else:
        raise TypeError("dx/vf is not an array")
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dx))

    term = term.to_ode()
    dt = term.contr(0, 1)
    vf = term.vf(0, y, args)
    vf_prod = term.vf_prod(0, y, args, dt)
    assert vf.shape == (3,)
    assert vf_prod.shape == (3,)
    assert tree_allclose(vf_prod, term.prod(vf, dt))


def test_ode_adjoint_term(getkey):
    vector_field = lambda t, y, args: -y
    term = diffrax.ODETerm(vector_field)
    adjoint_term = diffrax._term.AdjointTerm(term)
    t, y, a_y, dt = jr.normal(getkey(), (4,))
    ode_term = diffrax.ODETerm(lambda t, y, args: None)
    ode_term = eqx.tree_at(lambda m: m.vector_field, ode_term, None)
    aug = (y, a_y, None, ode_term)
    args = None
    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert tree_allclose(vf_prod1, vf_prod2)


def test_cde_adjoint_term(getkey):
    class VF(eqx.Module):
        mlp: eqx.nn.MLP

        def __call__(self, t, y, args):
            (y,) = y
            in_ = jnp.concatenate([t[None], y, *args])
            out = self.mlp(in_)
            return (out.reshape(2, 3),)

    mlp = eqx.nn.MLP(in_size=5, out_size=6, width_size=3, depth=1, key=getkey())
    vector_field = VF(mlp)
    control = lambda t0, t1: (jr.normal(getkey(), (3,)),)
    term = diffrax.ControlTerm(vector_field, control)
    adjoint_term = diffrax._term.AdjointTerm(term)
    t = jr.normal(getkey(), ())
    y = (jr.normal(getkey(), (2,)),)
    args = (jr.normal(getkey(), (1,)), jr.normal(getkey(), (1,)))
    a_y = (jr.normal(getkey(), (2,)),)
    a_args = (jr.normal(getkey(), (1,)), jr.normal(getkey(), (1,)))
    randlike = lambda a: jr.normal(getkey(), a.shape)
    a_term = jtu.tree_map(randlike, eqx.filter(term, eqx.is_array))
    aug = (y, a_y, a_args, a_term)
    dt = adjoint_term.contr(t, t + 1)

    vf_prod1 = adjoint_term.vf_prod(t, aug, args, dt)
    vf = adjoint_term.vf(t, aug, args)
    vf_prod2 = adjoint_term.prod(vf, dt)
    assert tree_allclose(vf_prod1, vf_prod2)


def test_weaklydiagonal_deprecate():
    with pytest.warns(match="WeaklyDiagonalControlTerm"):
        _ = diffrax.WeaklyDiagonalControlTerm(
            lambda t, y, args: 0.0, lambda t0, t1: jnp.array(t1 - t0)
        )


def test_correction():
    drift = lambda t, y, args: jnp.sin(y / 2)
    diffusion = lambda t, y, args: jax.vmap(lambda a, b: a * b, in_axes=(None, 0))(
        y, jnp.ones((3, 10))
    ).T

    t0, t1 = 0.0, 0.1
    y0 = jnp.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5])

    def solve(key, solver, dt):
        brownian_motion = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt / 2, shape=(3,), key=key
        )
        drift_term = diffrax.ODETerm(drift)
        diffusion_term = diffrax.ControlTerm(diffusion, brownian_motion)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        return diffrax.diffeqsolve(
            terms, solver, t0=t0, t1=t1, dt0=dt, y0=y0, saveat=diffrax.SaveAt(t1=True)
        )

    def solve_corrected(key, solver, dt):
        brownian_motion = diffrax.VirtualBrownianTree(
            t0, t1, tol=dt / 2, shape=(3,), key=key
        )
        diffusion_term = diffrax.ControlTerm(diffusion, brownian_motion)
        drift_term = diffrax.stratonovich_to_ito(diffrax.ODETerm(drift), diffusion_term)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        return diffrax.diffeqsolve(
            terms, solver, t0=t0, t1=t1, dt0=dt, y0=y0, saveat=diffrax.SaveAt(t1=True)
        )

    solve_fn = eqx.filter_jit(eqx.filter_vmap(solve, in_axes=(0, None, None)))
    solve_corrected_fn = eqx.filter_jit(
        eqx.filter_vmap(solve_corrected, in_axes=(0, None, None))
    )

    euler_solver = diffrax.Euler()
    heun_solver = diffrax.Heun()

    key = jax.random.key(0)
    keys = jax.random.split(key, 100)

    em = solve_fn(keys, euler_solver, 1e-3).ys.squeeze()
    h = solve_fn(keys, heun_solver, 1e-2).ys.squeeze()
    h_corrected = solve_corrected_fn(keys, heun_solver, 1e-2).ys.squeeze()

    assert ((em - h) ** 2).mean() > 1e-4
    assert ((em - h_corrected) ** 2).mean() < 1e-4
