"""This module implements a reconstruction operation that takes
a :class:`.Function`, defined on a broken function space via
:class:`ufl.BrokenElement`, and provides its representation in
the corresponding non-broken space.
"""
from __future__ import absolute_import, print_function, division
import ufl

from firedrake.function import Function
from firedrake.parloops import par_loop, READ, INC


__all__ = ["reconstruct", "Reconstructor"]


def reconstruct(v_b, V):
    """Reconstruct a :class:`.Function`, defined on a broken function
    space and transfer its data into a function defined on an unbroken
    finite element space.

    In other words: suppose we have a function v defined on a space constructed
    from a :class:`ufl.BrokenElement`. This methods allows one to "project"
    the data into an unbroken function space.

    This method avoids assembling a mass matrix system to solve a Galerkin
    projection problem; instead kernels are generated which computes weighted
    averages between facet degrees of freedom.

    :arg v_b: the :class:`.Function` to reconstruct.
    :arg V: the target function space.
    """

    if not isinstance(v_b, Function):
        raise RuntimeError(
            "Can only reconstruct functions. Not %s" % type(v_b)
        )

    reconstructor = Reconstructor(v_b, V)
    result = reconstructor.reconstruct()
    return result


class Reconstructor(object):
    """A reconstructor takes a Firedrake function, defined on a
    "broken" function space (through means of :class:`ufl.BrokenElement`),
    and returns its representation in a non-broken space. Unlike a
    projection, this operation does not require assembling and solving
    a mass matrix system. Instead, a weight function is created by spinning
    over the cells of the mesh, and then is used to average the contribution
    of facet dofs of the "broken" function to create its continuous
    representation.

    :arg v_b: the :class:`.Function` defined on the broken finite
              element space.
    :arg V: the :class:`.FunctionSpace`, which is not broken, of the
            reconstructed function.
    """

    def __init__(self, v_b, V):

        if not isinstance(v_b.function_space().ufl_element(),
                          ufl.BrokenElement):
            raise ValueError(
                "Function space must be defined on a broken element."
            )

        super(Reconstructor, self).__init__()
        self._vb = v_b
        self._weight_function = Function(V)
        self._v_rec = Function(V)

    def reconstruct(self):
        """Performs two loops over mesh cells. The first
        will populate the weight function. The second will
        perform the averaging and assign values to the
        reconstructed function.

        Returns: The reconstructed function.
        """

        weight_kernel = """
        for (int i=0; i<weight.dofs; ++i) {
            weight[i][0] += 1.0;
        }"""

        average_kernel = """
        for (int i=0; i<vrec.dofs; ++i) {
            vrec[i][0] += v_b[i][0]/weight[i][0];
        }"""

        par_loop(weight_kernel, ufl.dx, {"weight": (self._weight_function, INC)})
        par_loop(average_kernel, ufl.dx, {"vrec": (self._v_rec, INC),
                                          "v_b": (self._vb, READ),
                                          "weight": (self._weight_function, READ)})
        return self._v_rec
