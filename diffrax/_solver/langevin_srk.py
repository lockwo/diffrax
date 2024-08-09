import abc
from typing import Any, Callable, Generic, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jax import vmap
from jax._src.tree_util import PyTreeDef
from jaxtyping import Array, ArrayLike, PyTree

from .._custom_types import (
    AbstractBrownianIncrement,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import (
    _LangevinDiffusionTerm,
    _LangevinDriftTerm,
    _LangevinTuple,
    _LangevinX,
    AbstractTerm,
    MultiTerm,
    WrapTerm,
)
from .base import AbstractItoSolver, AbstractStratonovichSolver


_ErrorEstimate = TypeVar("_ErrorEstimate", None, _LangevinTuple)
_LangevinArgs = tuple[_LangevinX, _LangevinX, Callable[[_LangevinX], _LangevinX]]


def get_args_from_terms(
    terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
) -> _LangevinArgs:
    drift, diffusion = terms.terms
    if isinstance(drift, WrapTerm):
        unwrapped_drift = drift.term
        assert isinstance(diffusion, WrapTerm)
        unwrapped_diffusion = diffusion.term
        assert isinstance(unwrapped_drift, _LangevinDriftTerm)
        assert isinstance(unwrapped_diffusion, _LangevinDiffusionTerm)
        gamma = unwrapped_drift.gamma
        u = unwrapped_drift.u
        f = unwrapped_drift.grad_f
    else:
        assert isinstance(drift, _LangevinDriftTerm)
        assert isinstance(diffusion, _LangevinDiffusionTerm)
        gamma = drift.gamma
        u = drift.u
        f = drift.grad_f
    return gamma, u, f


class AbstractCoeffs(eqx.Module):
    dtype: AbstractVar[jnp.dtype]


_Coeffs = TypeVar("_Coeffs", bound=AbstractCoeffs)


# TODO: I'm not sure if I can use the _Coeffs type here,
# given that I do not use Generic[_Coeffs] in the class definition.
# How should I work around this?
class SolverState(eqx.Module, Generic[_Coeffs]):
    h: RealScalarLike
    taylor_coeffs: PyTree[_Coeffs, "_LangevinX"]
    coeffs: _Coeffs
    rho: _LangevinX
    prev_f: _LangevinX


# CONCERNING COEFFICIENTS:
# The coefficients used in a step of this SRK depend on
# the time increment h, and the parameter gamma.
# Assuming the modelled SDE stays the same (i.e. gamma is fixed),
# then these coefficients must be recomputed each time h changes.
# Furthermore, for very small h, directly computing the coefficients
# via the function below can cause large floating point errors.
# Hence, we pre-compute the Taylor expansion of the SRK coefficients
# around h=0. Then we can compute the SRK coefficients either via
# the Taylor expansion, or via direct computation.
# In short the Taylor coefficients give a Taylor expansion with which
# one can compute the SRK coefficients more precisely for a small h.


class AbstractLangevinSRK(
    AbstractStratonovichSolver[SolverState],
    AbstractItoSolver[SolverState],
    Generic[_Coeffs, _ErrorEstimate],
):
    """Abstract class for Stochastic Runge Kutta methods specifically designed
    for Underdamped Langevin Diffusion of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2gammau)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $gamma$ and $u$ are the
    friction and momentum parameters, and $W$ is a Brownian motion.

    Solvers which inherit from this class include ALIGN, SORT, ShOULD, and
    QUIC_SORT.
    """

    term_structure = MultiTerm[tuple[_LangevinDriftTerm, _LangevinDiffusionTerm]]
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _coeffs_structure: eqx.AbstractClassVar[PyTreeDef]
    minimal_levy_area: eqx.AbstractClassVar[type[AbstractBrownianIncrement]]

    @abc.abstractmethod
    def _directly_compute_coeffs_leaf(self, h, c) -> _Coeffs:
        raise NotImplementedError

    @abc.abstractmethod
    def _tay_coeffs_single(self, c: Array) -> _Coeffs:
        raise NotImplementedError

    def _comp_taylor_coeffs_leaf(self, c: Array) -> _Coeffs:
        # c is a leaf of gamma

        # When the step-size h is small the coefficients (which depend on h) need
        # to be computed via Taylor expansion to ensure numerical stability.
        # This precomputes the Taylor coefficients (depending on gamma and u), which
        # are then multiplied by powers of h, to get the SRK coefficients.
        out = self._tay_coeffs_single(c)

        def check_shape(coeff_leaf):
            permitted_shapes = [c.shape + (3, 6), c.shape + (1, 6), c.shape + (6,)]
            assert (
                coeff_leaf.shape in permitted_shapes
            ), f"leaf shape: {coeff_leaf.shape}, c shape: {c.shape}"

        jtu.tree_map(check_shape, out)
        return out

    @staticmethod
    def _eval_taylor(h, tay_coeffs: _Coeffs) -> _Coeffs:
        # Multiplies the pre-computed Taylor coefficients by powers of h.
        dtype = tay_coeffs.dtype
        h_powers = jnp.power(h, jnp.arange(0, 6, dtype=h.dtype)).astype(dtype)
        return jtu.tree_map(
            lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_coeffs
        )

    def _recompute_coeffs(
        self, h, gamma: _LangevinX, tay_coeffs: PyTree[_Coeffs], state_h
    ) -> _Coeffs:
        def recompute_coeffs_leaf(c: ArrayLike, _tay_coeffs: _Coeffs):
            # Used when the step-size h changes and coefficients need to be recomputed
            # Depending on the size of h*gamma choose whether the Taylor expansion or
            # direct computation is more accurate.
            cond = h * c < self.taylor_threshold  # c is a leaf of gamma
            if jnp.ndim(c) == 0:
                return lax.cond(
                    cond,
                    lambda h_: self._eval_taylor(h_, _tay_coeffs),
                    lambda h_: self._directly_compute_coeffs_leaf(h_, c),
                    h,
                )
            else:
                tay_out = self._eval_taylor(h, _tay_coeffs)
                if cond.ndim < jtu.tree_leaves(tay_out)[0].ndim:
                    cond = jnp.expand_dims(cond, axis=-1)

                def select_tay_or_direct(dummy):
                    fun = lambda _c: self._directly_compute_coeffs_leaf(h, _c)
                    direct_out = vmap(fun)(c)

                    def _choose(tay_leaf, direct_leaf):
                        assert tay_leaf.ndim == direct_leaf.ndim == cond.ndim, (
                            f"tay_leaf.ndim: {tay_leaf.ndim},"
                            f" direct_leaf.ndim: {direct_leaf.ndim},"
                            f" cond.ndim: {cond.ndim}"
                        )
                        return jnp.where(cond, tay_leaf, direct_leaf)

                    return jtu.tree_map(_choose, tay_out, direct_out)

                # If all entries of h*gamma are below threshold, only compute tay_out
                # otherwise, compute both tay_out and direct_out and select the
                # correct one for each dimension
                return lax.cond(
                    eqxi.unvmap_all(cond), lambda _: tay_out, select_tay_or_direct, None
                )

        tree_with_coeffs = jtu.tree_map(recompute_coeffs_leaf, gamma, tay_coeffs)
        outer = jtu.tree_structure(gamma)
        inner = self._coeffs_structure
        coeffs_with_tree = jtu.tree_transpose(outer, inner, tree_with_coeffs)
        return coeffs_with_tree

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _LangevinTuple,
        args: PyTree,
    ) -> SolverState:
        """Precompute _SolverState which carries the Taylor coefficients and the
        SRK coefficients (which can be computed from h and the Taylor coeffs).
        Some solvers of this type are FSAL, so _SolverState also carries the previous
        evaluation of grad_f.
        """
        drift, diffusion = terms.terms
        gamma, u, f = get_args_from_terms(terms)

        h = drift.contr(t0, t1)
        x0, v0 = y0

        def _check_shapes(_c, _u, _x, _v):
            assert _c.shape == _u.shape == _x.shape == _v.shape

        assert jtu.tree_all(jtu.tree_map(_check_shapes, gamma, u, x0, v0))

        tay_coeffs = jtu.tree_map(self._comp_taylor_coeffs_leaf, gamma)
        # tay_coeffs have the same tree structure as gamma, with each leaf being a
        # _Coeffs and the arrays have an extra trailing dimension of 6

        coeffs = self._recompute_coeffs(h, gamma, tay_coeffs, -1.0)
        rho = jtu.tree_map(lambda c, _u: jnp.sqrt(2 * c * _u), gamma, u)

        state_out = SolverState(
            h=h,
            taylor_coeffs=tay_coeffs,
            coeffs=coeffs,
            rho=rho,
            prev_f=f(x0),
        )

        return state_out

    @staticmethod
    @abc.abstractmethod
    def _compute_step(
        h: RealScalarLike,
        levy,
        x0: _LangevinX,
        v0: _LangevinX,
        langevin_args: _LangevinArgs,
        coeffs: _Coeffs,
        st: SolverState,
    ) -> tuple[_LangevinX, _LangevinX, _LangevinX, _ErrorEstimate]:
        raise NotImplementedError

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _LangevinTuple,
        args: PyTree,
        solver_state: SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[_LangevinTuple, _ErrorEstimate, DenseInfo, SolverState, RESULTS]:
        del made_jump, args
        st = solver_state
        drift, diffusion = terms.terms
        gamma, u, f = get_args_from_terms(terms)

        h = drift.contr(t0, t1)
        h_state = st.h
        tay: PyTree[_Coeffs] = st.taylor_coeffs
        coeffs: _Coeffs = st.coeffs

        # If h changed recompute coefficients
        cond = jnp.isclose(h_state, h, rtol=1e-10, atol=1e-12)
        coeffs = lax.cond(
            cond,
            lambda x: x,
            lambda _: self._recompute_coeffs(h, gamma, tay, h_state),
            coeffs,
        )

        # compute the Brownian increment and space-time Levy area
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, self.minimal_levy_area), (
            f"The Brownian motion must have"
            f" `levy_area={self.minimal_levy_area.__name__}`"
        )

        x0, v0 = y0
        x_out, v_out, f_fsal, error = self._compute_step(
            h, levy, x0, v0, (gamma, u, f), coeffs, st
        )

        def check_shapes_dtypes(_x, _v, _f, _x0):
            assert _x.dtype == _v.dtype == _f.dtype == _x0.dtype, (
                f"dtypes don't match. x0: {x0.dtype},"
                f" v_out: {_v.dtype}, x_out: {_x.dtype}, f_fsal: {_f.dtype}"
            )
            assert _x.shape == _v.shape == _f.shape == _x0.shape, (
                f"Shapes don't match. x0: {x0.shape},"
                f" v_out: {_v.shape}, x_out: {_x.shape}, f_fsal: {_f.shape}"
            )

        jtu.tree_map(check_shapes_dtypes, x_out, v_out, f_fsal, x0)

        y1 = (x_out, v_out)

        dense_info = dict(y0=y0, y1=y1)
        st = SolverState(
            h=h,
            taylor_coeffs=tay,
            coeffs=coeffs,
            rho=st.rho,
            prev_f=f_fsal,
        )
        return y1, error, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        y0: _LangevinTuple,
        args: PyTree,
    ):
        return terms.vf(t0, y0, args)
