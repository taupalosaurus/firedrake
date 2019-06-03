import numpy as np
from firedrake import *


def test_constant():
    cg1 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 1)
    f = interpolate(Constant(1.0), cg1)
    assert np.allclose(1.0, f.dat.data)


def test_function():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    f = interpolate(x[0]*x[0], V1)
    g = interpolate(f, V2)

    # g shall be equivalent to:
    h = interpolate(x[0], V2)

    assert np.allclose(g.dat.data, h.dat.data)


def test_inner():
    m = UnitTriangleMesh()
    V1 = FunctionSpace(m, 'P', 1)
    V2 = FunctionSpace(m, 'P', 2)

    x, y = SpatialCoordinate(m)
    f = interpolate(inner(x, x), V1)
    g = interpolate(f, V2)

    # g shall be equivalent to:
    h = interpolate(x, V2)

    assert np.allclose(g.dat.data, h.dat.data)


def test_coordinates():
    cg2 = FunctionSpace(UnitSquareMesh(5, 5), "CG", 2)
    x = SpatialCoordinate(cg2.mesh())
    f = interpolate(x[0]*x[0], cg2)

    x = SpatialCoordinate(cg2.mesh())
    g = interpolate(x[0]*x[0], cg2)

    assert np.allclose(f.dat.data, g.dat.data)


def test_piola():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = interpolate(f[0], V)

    # g shall be equivalent to:
    h = project(f[0], V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_vector():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = VectorFunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], Constant(0.0))), U)
    g = interpolate(f, V)

    # g shall be equivalent to:
    h = project(f, V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_constant_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 1)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = interpolate(div(f), V)

    assert np.allclose(2.0, g.dat.data)


def test_compound_expression():
    m = UnitTriangleMesh()
    x = SpatialCoordinate(m)
    U = FunctionSpace(m, 'RT', 2)
    V = FunctionSpace(m, 'P', 2)

    f = project(as_vector((x[0], x[1])), U)
    g = interpolate(Constant(1.5)*div(f) + sin(x[0] * np.pi), V)

    # g shall be equivalent to:
    h = interpolate(3.0 + sin(pi * x[0]), V)

    assert np.allclose(g.dat.data, h.dat.data)


def test_cell_orientation():
    m = UnitCubedSphereMesh(2)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)
    x = m.coordinates
    U = FunctionSpace(m, 'RTCF', 1)
    V = VectorFunctionSpace(m, 'DQ', 1)

    f = project(as_tensor([x[1], -x[0], 0.0]), U)
    g = interpolate(f, V)

    # g shall be close to:
    h = project(f, V)

    assert abs(g.dat.data - h.dat.data).max() < 1e-2


def test_cellvolume():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'DG', 0)

    f = interpolate(CellVolume(m), V)

    assert np.allclose(f.dat.data_ro, 0.125)


def test_cellvolume_higher_order_coords():
    m = UnitTriangleMesh()
    V = VectorFunctionSpace(m, 'P', 3)
    f = Function(V)
    f.interpolate(m.coordinates)

    # Warp mesh so that the bottom triangle line is:
    # x(x - 1)(x + a) with a = 19/12.0
    def warp(x):
        return x * (x - 1)*(x + 19/12.0)

    f.dat.data[1, 1] = warp(1.0/3.0)
    f.dat.data[2, 1] = warp(2.0/3.0)

    mesh = Mesh(f)
    g = interpolate(CellVolume(mesh), FunctionSpace(mesh, 'DG', 0))

    assert np.allclose(g.dat.data_ro, 0.5 - (1.0/4.0 - (1 - 19.0/12.0)/3.0 - 19/24.0))


def test_mixed():
    m = UnitTriangleMesh()
    x = m.coordinates
    V1 = FunctionSpace(m, 'BDFM', 2)
    V2 = VectorFunctionSpace(m, 'P', 2)
    f = Function(V1 * V2)
    f.sub(0).project(as_tensor([x[1], -x[0]]))
    f.sub(1).interpolate(as_tensor([x[0], x[1]]))

    V = FunctionSpace(m, 'P', 1)
    g = interpolate(dot(grad(f[0]), grad(f[3])), V)

    assert np.allclose(1.0, g.dat.data)


def test_lvalue_rvalue():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.assign(1.0)
    u.interpolate(u + 1.0)
    assert np.allclose(u.dat.data_ro, 2.0)

def test_interpolator2():
    mesh = UnitSquareMesh(10, 10)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    P2 = FunctionSpace(mesh, "CG", 2)
    expr = x[0] + x[1]
    x_P1 = interpolate(expr, P1)
    interpolator = Interpolator(TestFunction(P1), P2)
    x_P2 = interpolator.interpolate(x_P1)
    x_P2_direct = interpolate(expr, P2)
    assert np.allclose(x_P2.dat.data, x_P2_direct.dat.data)

def test_interpolator3():
    mesh = UnitSquareMesh(10, 10)
    x = SpatialCoordinate(mesh)
    P2 = FunctionSpace(mesh, "CG", 2)
    P3 = FunctionSpace(mesh, "CG", 3)
    expr = x[0]**2 + x[1]**2
    x_P3 = interpolate(expr, P2)
    interpolator = Interpolator(TestFunction(P2), P3)
    x_P3 = interpolator.interpolate(x_P3)
    x_P3_direct = interpolate(expr, P3)
    assert np.allclose(x_P3.dat.data, x_P3_direct.dat.data)

def test_adjoint():
    mesh = UnitSquareMesh(10,10)
    P2 = FunctionSpace(mesh, "CG", 2)
    P1 = FunctionSpace(mesh, "CG", 1)
    u = TestFunction(P1)
    v = TestFunction(P2)
    u_P1 = assemble(TestFunction(P1) * dx)
    interpolator = Interpolator(TestFunction(P1), P2)
    v_adj = interpolator.interpolate(assemble(v * dx), transpose=True)
    assert np.allclose(u_P1.dat.data, v_adj.dat.data)
