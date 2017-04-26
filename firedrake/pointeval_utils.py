from __future__ import absolute_import, print_function, division

from pyop2.datatypes import IntType, as_cstr

from coffee import base as ast


def compile_element(expression, coordinates, parameters=None):
    """Generates C code for point evaluations.

    :arg ufl_element: UFL expression
    :returns: C code as string
    """
    from ufl.algorithms import extract_arguments, extract_coefficients
    from tsfc import ufl_utils, fem

    import tsfc.kernel_interface.firedrake as firedrake_interface
    from tsfc.parameters import default_parameters
    import gem
    import gem.impero_utils as impero_utils
    from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee

    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # No arguments, please!
    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression)

    # Collect required coefficients
    coefficient, = extract_coefficients(expression)

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        expression = ufl_utils.replace_coordinates(expression, coordinates)

    # Initialise kernel builder
    builder = firedrake_interface.KernelBuilderBase()
    funargs = []
    funargs.append(builder._coefficient(coordinates, "x"))
    funargs.append(builder._coefficient(coefficient, "f"))

    # # Split mixed coefficients
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    point = gem.Variable('X', (domain.ufl_cell().topological_dimension(),))
    funargs.insert(0, ast.Decl(SCALAR_TYPE, ast.Symbol('X', rank=(domain.ufl_cell().topological_dimension(),))))

    # point = gem.Variable('X', (2,))  # FIXME
    # point_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('X', rank=(2,)))  # FIXME
    config = dict(interface=builder,
                  ufl_cell=coordinates.ufl_domain().ufl_cell(),
                  precision=parameters["precision"],
                  point_expr=point)
    # config["cellvolume"] = cellvolume_generator(coordinates.ufl_domain(), coordinates, config)
    context = fem.GemPointContext(**config)

    # Abs-simplification
    expression = ufl_utils.simplify_abs(expression)

    # Translate UFL to GEM, lowering finite element specific nodes
    translator = fem.Translator(context)
    from ufl.corealg.map_dag import map_expr_dags
    result, = map_expr_dags(translator, [expression])

    tensor_indices = ()
    if expression.ufl_shape:
        tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
        retvar = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
        R_sym = ast.Symbol('R', rank=expression.ufl_shape)
        result = gem.Indexed(result, tensor_indices)
    else:
        R_sym = ast.Symbol('R', rank=(1,))
        retvar = gem.Indexed(gem.Variable('R', (1,)), (0,))
    funargs.insert(0, ast.Decl(SCALAR_TYPE, R_sym))

    result, = impero_utils.preprocess_gem([result])
    impero_c = impero_utils.compile_gem([(retvar, result)], tensor_indices)
    body = generate_coffee(impero_c, {}, parameters["precision"])

    # Build kernel tuple
    kernel_code = builder.construct_kernel("evaluate_kernel", funargs, body)

    from ufl import TensorProductCell

    # Create FIAT element
    cell = domain.ufl_cell()
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "extruded_arg": ", %s nlayers" % as_cstr(IntType) if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """static inline void wrap_evaluate(double *result, double *X, double *coords, %(IntType)s *coords_map, double *f, %(IntType)s *f_map%(extruded_arg)s, %(IntType)s cell);

int evaluate(struct Function *f, double *x, double *result)
{
    struct ReferenceCoords reference_coords;
    %(IntType)s cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }

    wrap_evaluate(result, reference_coords.X, f->coords, f->coords_map, f->f, f->f_map%(nlayers)s, cell);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code.gencode()
