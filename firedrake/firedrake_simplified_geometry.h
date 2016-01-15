/* --- Computation of Jacobian matrices --- */

/* Compute Jacobian J for interval embedded in R^1 */
#define compute_jacobian_interval_1d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0];

/* Compute Jacobian J for interval embedded in R^2 */
#define compute_jacobian_interval_2d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[3] - coordinate_dofs[2];

/* Compute Jacobian J for quad embedded in R^2 */
#define compute_jacobian_quad_2d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[1] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[2]); \
  J[2] = 0.5*(coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[4] - coordinate_dofs[5]); \
  J[3] = 0.5*(coordinate_dofs[5] + coordinate_dofs[7] - coordinate_dofs[4] - coordinate_dofs[6]);

/* Compute Jacobian J for quad embedded in R^3 */
#define compute_jacobian_quad_3d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[1] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[2]); \
  J[2] = 0.5*(coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[4] - coordinate_dofs[5]); \
  J[3] = 0.5*(coordinate_dofs[5] + coordinate_dofs[7] - coordinate_dofs[4] - coordinate_dofs[6]); \
  J[4] = 0.5*(coordinate_dofs[10] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[9]); \
  J[5] = 0.5*(coordinate_dofs[9] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[10]);

/* Compute Jacobian J for interval embedded in R^3 */
#define compute_jacobian_interval_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[3] - coordinate_dofs[2]; \
  J[2] = coordinate_dofs[5] - coordinate_dofs[4];

/* Compute Jacobian J for triangle embedded in R^2 */
#define compute_jacobian_triangle_2d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[2] - coordinate_dofs[0]; \
  J[2] = coordinate_dofs[4] - coordinate_dofs[3]; \
  J[3] = coordinate_dofs[5] - coordinate_dofs[3];

/* Compute Jacobian J for triangle embedded in R^3 */
#define compute_jacobian_triangle_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[2] - coordinate_dofs[0]; \
  J[2] = coordinate_dofs[4] - coordinate_dofs[3]; \
  J[3] = coordinate_dofs[5] - coordinate_dofs[3]; \
  J[4] = coordinate_dofs[7] - coordinate_dofs[6]; \
  J[5] = coordinate_dofs[8] - coordinate_dofs[6];

/* Compute Jacobian J for tetrahedron embedded in R^3 */
#define compute_jacobian_tetrahedron_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1]  - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[2]  - coordinate_dofs[0]; \
  J[2] = coordinate_dofs[3]  - coordinate_dofs[0]; \
  J[3] = coordinate_dofs[5]  - coordinate_dofs[4]; \
  J[4] = coordinate_dofs[6]  - coordinate_dofs[4]; \
  J[5] = coordinate_dofs[7]  - coordinate_dofs[4]; \
  J[6] = coordinate_dofs[9]  - coordinate_dofs[8]; \
  J[7] = coordinate_dofs[10] - coordinate_dofs[8]; \
  J[8] = coordinate_dofs[11] - coordinate_dofs[8];

/* Compute Jacobian J for tensor product prism embedded in R^3 */
/* Explanation: the CG1 x CG1 basis functions are, in order,
(1-X-Y)(1-Z), (1-X-Y)Z, X(1-Z), XZ, Y(1-Z), YZ.  Each row of the
Jacobian is the derivatives of these w.r.t. X, Y and Z in turn,
evaluated at the midpoint (1/3, 1/3, 1/2).  This gives the
coefficients below.*/
#define compute_jacobian_prism_3d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[4] + coordinate_dofs[5] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[2] = (coordinate_dofs[1] + coordinate_dofs[3] + coordinate_dofs[5] - coordinate_dofs[0] - coordinate_dofs[2] - coordinate_dofs[4])/3.0; \
  J[3] = 0.5*(coordinate_dofs[8] + coordinate_dofs[9] - coordinate_dofs[6] - coordinate_dofs[7]); \
  J[4] = 0.5*(coordinate_dofs[10] + coordinate_dofs[11] - coordinate_dofs[6] - coordinate_dofs[7]); \
  J[5] = (coordinate_dofs[7] + coordinate_dofs[9] + coordinate_dofs[11] - coordinate_dofs[6] - coordinate_dofs[8] - coordinate_dofs[10])/3.0; \
  J[6] = 0.5*(coordinate_dofs[14] + coordinate_dofs[15] - coordinate_dofs[12] - coordinate_dofs[13]); \
  J[7] = 0.5*(coordinate_dofs[16] + coordinate_dofs[17] - coordinate_dofs[12] - coordinate_dofs[13]); \
  J[8] = (coordinate_dofs[13] + coordinate_dofs[15] + coordinate_dofs[17] - coordinate_dofs[12] - coordinate_dofs[14] - coordinate_dofs[16])/3.0;

/* Compute Jacobian J for tensor product hexahedron embedded in R^3 */
/* Explanation: the CG1 x CG1 basis functions are, in order, (1-X)(1-Y)(1-Z),
(1-X)(1-Y)Z, (1-X)Y(1-Z), (1-X)YZ, X(1-Y)(1-Z), X(1-Y)Z, XY(1-Z), XYZ.
Each row of the Jacobian is the derivatives of these w.r.t. X, Y and Z in turn,
evaluated at the midpoint (1/2, 1/2, 1/2). This gives the coefficients below. */
#define compute_jacobian_hex_3d(J, coordinate_dofs) \
  J[0] = 0.25*(coordinate_dofs[4] + coordinate_dofs[5] + coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[1] - coordinate_dofs[2] - coordinate_dofs[3]); \
  J[1] = 0.25*(coordinate_dofs[2] + coordinate_dofs[3] + coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[1] - coordinate_dofs[4] - coordinate_dofs[5]); \
  J[2] = 0.25*(coordinate_dofs[1] + coordinate_dofs[3] + coordinate_dofs[5] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[2] - coordinate_dofs[4] - coordinate_dofs[6]); \
  J[3] = 0.25*(coordinate_dofs[12] + coordinate_dofs[13] + coordinate_dofs[14] + coordinate_dofs[15] - coordinate_dofs[8] - coordinate_dofs [9] - coordinate_dofs[10] - coordinate_dofs[11]); \
  J[4] = 0.25*(coordinate_dofs[10] + coordinate_dofs[11] + coordinate_dofs[14] + coordinate_dofs[15] - coordinate_dofs[8] - coordinate_dofs [9] - coordinate_dofs[12] - coordinate_dofs[13]); \
  J[5] = 0.25*(coordinate_dofs [9] + coordinate_dofs[11] + coordinate_dofs[13] + coordinate_dofs[15] - coordinate_dofs[8] - coordinate_dofs[10] - coordinate_dofs[12] - coordinate_dofs[14]); \
  J[6] = 0.25*(coordinate_dofs[20] + coordinate_dofs[21] + coordinate_dofs[22] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[17] - coordinate_dofs[18] - coordinate_dofs[19]); \
  J[7] = 0.25*(coordinate_dofs[18] + coordinate_dofs[19] + coordinate_dofs[22] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[17] - coordinate_dofs[20] - coordinate_dofs[21]); \
  J[8] = 0.25*(coordinate_dofs[17] + coordinate_dofs[19] + coordinate_dofs[21] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[18] - coordinate_dofs[20] - coordinate_dofs[22]);

/* Jacobians for interior facets of different sorts */

/* Compute Jacobian J for interval embedded in R^1 */
#define compute_jacobian_interval_int_1d compute_jacobian_interval_1d

/* Compute Jacobian J for interval embedded in R^2 */
#define compute_jacobian_interval_int_2d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[5] - coordinate_dofs[4];

/* Compute Jacobian J for quad embedded in R^2 */
#define compute_jacobian_quad_int_2d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[1] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[2]); \
  J[2] = 0.5*(coordinate_dofs[10] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[9]); \
  J[3] = 0.5*(coordinate_dofs[9] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[10]);

/* Compute Jacobian J for quad embedded in R^3 */
#define compute_jacobian_quad_int_3d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[1] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[2]); \
  J[2] = 0.5*(coordinate_dofs[10] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[9]); \
  J[3] = 0.5*(coordinate_dofs[9] + coordinate_dofs[11] - coordinate_dofs[8] - coordinate_dofs[10]); \
  J[4] = 0.5*(coordinate_dofs[18] + coordinate_dofs[19] - coordinate_dofs[16] - coordinate_dofs[17]); \
  J[5] = 0.5*(coordinate_dofs[17] + coordinate_dofs[19] - coordinate_dofs[16] - coordinate_dofs[18]);

/* Compute Jacobian J for interval embedded in R^3 */
#define compute_jacobian_interval_int_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[5] - coordinate_dofs[4]; \
  J[2] = coordinate_dofs[9] - coordinate_dofs[8];

/* Compute Jacobian J for triangle embedded in R^2 */
#define compute_jacobian_triangle_int_2d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1] - coordinate_dofs[0]; \
  J[1] = coordinate_dofs[2] - coordinate_dofs[0]; \
  J[2] = coordinate_dofs[7] - coordinate_dofs[6]; \
  J[3] = coordinate_dofs[8] - coordinate_dofs[6];

/* Compute Jacobian J for triangle embedded in R^3 */
#define compute_jacobian_triangle_int_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1]  - coordinate_dofs[0] ; \
  J[1] = coordinate_dofs[2]  - coordinate_dofs[0] ; \
  J[2] = coordinate_dofs[7]  - coordinate_dofs[6] ; \
  J[3] = coordinate_dofs[8]  - coordinate_dofs[6] ; \
  J[4] = coordinate_dofs[13] - coordinate_dofs[12]; \
  J[5] = coordinate_dofs[14] - coordinate_dofs[12];

/* Compute Jacobian J for tetrahedron embedded in R^3 */
#define compute_jacobian_tetrahedron_int_3d(J, coordinate_dofs) \
  J[0] = coordinate_dofs[1]  - coordinate_dofs[0] ; \
  J[1] = coordinate_dofs[2]  - coordinate_dofs[0] ; \
  J[2] = coordinate_dofs[3]  - coordinate_dofs[0] ; \
  J[3] = coordinate_dofs[9]  - coordinate_dofs[8] ; \
  J[4] = coordinate_dofs[10] - coordinate_dofs[8] ; \
  J[5] = coordinate_dofs[11] - coordinate_dofs[8] ; \
  J[6] = coordinate_dofs[17] - coordinate_dofs[16]; \
  J[7] = coordinate_dofs[18] - coordinate_dofs[16]; \
  J[8] = coordinate_dofs[19] - coordinate_dofs[16];

/* Compute Jacobian J for tensor product prism embedded in R^3 */
#define compute_jacobian_prism_int_3d(J, coordinate_dofs) \
  J[0] = 0.5*(coordinate_dofs[2] + coordinate_dofs[3] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[1] = 0.5*(coordinate_dofs[4] + coordinate_dofs[5] - coordinate_dofs[0] - coordinate_dofs[1]); \
  J[2] = (coordinate_dofs[1] + coordinate_dofs[3] + coordinate_dofs[5] - coordinate_dofs[0] - coordinate_dofs[2] - coordinate_dofs[4])/3.0; \
  J[3] = 0.5*(coordinate_dofs[14] + coordinate_dofs[15] - coordinate_dofs[12] - coordinate_dofs[13]); \
  J[4] = 0.5*(coordinate_dofs[16] + coordinate_dofs[17] - coordinate_dofs[12] - coordinate_dofs[13]); \
  J[5] = (coordinate_dofs[13] + coordinate_dofs[15] + coordinate_dofs[17] - coordinate_dofs[12] - coordinate_dofs[14] - coordinate_dofs[16])/3.0; \
  J[6] = 0.5*(coordinate_dofs[26] + coordinate_dofs[27] - coordinate_dofs[24] - coordinate_dofs[25]); \
  J[7] = 0.5*(coordinate_dofs[28] + coordinate_dofs[29] - coordinate_dofs[24] - coordinate_dofs[25]); \
  J[8] = (coordinate_dofs[25] + coordinate_dofs[27] + coordinate_dofs[29] - coordinate_dofs[24] - coordinate_dofs[26] - coordinate_dofs[28])/3.0;

/* Compute Jacobian J for tensor product hexahedron embedded in R^3 */
#define compute_jacobian_hex_int_3d(J, coordinate_dofs) \
  J[0] = 0.25*(coordinate_dofs[4] + coordinate_dofs[5] + coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[1] - coordinate_dofs[2] - coordinate_dofs[3]); \
  J[1] = 0.25*(coordinate_dofs[2] + coordinate_dofs[3] + coordinate_dofs[6] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[1] - coordinate_dofs[4] - coordinate_dofs[5]); \
  J[2] = 0.25*(coordinate_dofs[1] + coordinate_dofs[3] + coordinate_dofs[5] + coordinate_dofs[7] - coordinate_dofs[0] - coordinate_dofs[2] - coordinate_dofs[4] - coordinate_dofs[6]); \
  J[3] = 0.25*(coordinate_dofs[20] + coordinate_dofs[21] + coordinate_dofs[22] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[17] - coordinate_dofs[18] - coordinate_dofs[19]); \
  J[4] = 0.25*(coordinate_dofs[18] + coordinate_dofs[19] + coordinate_dofs[22] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[17] - coordinate_dofs[20] - coordinate_dofs[21]); \
  J[5] = 0.25*(coordinate_dofs[17] + coordinate_dofs[19] + coordinate_dofs[21] + coordinate_dofs[23] - coordinate_dofs[16] - coordinate_dofs[18] - coordinate_dofs[20] - coordinate_dofs[22]); \
  J[6] = 0.25*(coordinate_dofs[36] + coordinate_dofs[37] + coordinate_dofs[38] + coordinate_dofs[39] - coordinate_dofs[32] - coordinate_dofs[33] - coordinate_dofs[34] - coordinate_dofs[35]); \
  J[7] = 0.25*(coordinate_dofs[34] + coordinate_dofs[35] + coordinate_dofs[38] + coordinate_dofs[39] - coordinate_dofs[32] - coordinate_dofs[33] - coordinate_dofs[36] - coordinate_dofs[37]); \
  J[8] = 0.25*(coordinate_dofs[33] + coordinate_dofs[35] + coordinate_dofs[37] + coordinate_dofs[39] - coordinate_dofs[32] - coordinate_dofs[34] - coordinate_dofs[36] - coordinate_dofs[38]);

/* --- Computation of Jacobian inverses --- */

/* Compute Jacobian inverse K for interval embedded in R^1 */
#define compute_jacobian_inverse_interval_1d(K, det, J) \
  det = J[0]; \
  K[0] = 1.0 / det;

/* Compute Jacobian (pseudo)inverse K for interval embedded in R^2 */
#define compute_jacobian_inverse_interval_2d(K, det, J) \
  do { const double det2 = J[0]*J[0] + J[1]*J[1]; \
  det = sqrt(det2); \
  K[0] = J[0] / det2; \
  K[1] = J[1] / det2; } while (0)

/* Compute Jacobian (pseudo)inverse K for interval embedded in R^3 */
#define compute_jacobian_inverse_interval_3d(K, det, J) \
  do { const double det2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2]; \
  det = sqrt(det2); \
  K[0] = J[0] / det2; \
  K[1] = J[1] / det2; \
  K[2] = J[2] / det2; } while (0)

/* Compute Jacobian inverse K for triangle embedded in R^2 */
#define compute_jacobian_inverse_triangle_2d(K, det, J) \
  det = J[0]*J[3] - J[1]*J[2]; \
  K[0] =  J[3] / det; \
  K[1] = -J[1] / det; \
  K[2] = -J[2] / det; \
  K[3] =  J[0] / det;

/* Compute Jacobian (pseudo)inverse K for triangle embedded in R^3 */
#define compute_jacobian_inverse_triangle_3d(K, det, J) \
  do { const double d_0 = J[2]*J[5] - J[4]*J[3]; \
  const double d_1 = J[4]*J[1] - J[0]*J[5]; \
  const double d_2 = J[0]*J[3] - J[2]*J[1]; \
  const double c_0 = J[0]*J[0] + J[2]*J[2] + J[4]*J[4]; \
  const double c_1 = J[1]*J[1] + J[3]*J[3] + J[5]*J[5]; \
  const double c_2 = J[0]*J[1] + J[2]*J[3] + J[4]*J[5]; \
  const double den = c_0*c_1 - c_2*c_2; \
  const double det2 = d_0*d_0 + d_1*d_1 + d_2*d_2; \
  det = sqrt(det2); \
  K[0] = (J[0]*c_1 - J[1]*c_2) / den; \
  K[1] = (J[2]*c_1 - J[3]*c_2) / den; \
  K[2] = (J[4]*c_1 - J[5]*c_2) / den; \
  K[3] = (J[1]*c_0 - J[0]*c_2) / den; \
  K[4] = (J[3]*c_0 - J[2]*c_2) / den; \
  K[5] = (J[5]*c_0 - J[4]*c_2) / den; } while (0)

/* Compute Jacobian (pseudo)inverse K for quad embedded in R^2 */
#define compute_jacobian_inverse_quad_2d compute_jacobian_inverse_triangle_2d

/* Compute Jacobian (pseudo)inverse K for quad embedded in R^3 */
#define compute_jacobian_inverse_quad_3d compute_jacobian_inverse_triangle_3d

/* Compute Jacobian inverse K for tetrahedron embedded in R^3 */
#define compute_jacobian_inverse_tetrahedron_3d(K, det, J) \
  do { const double d_00 = J[4]*J[8] - J[5]*J[7]; \
  const double d_01 = J[5]*J[6] - J[3]*J[8]; \
  const double d_02 = J[3]*J[7] - J[4]*J[6]; \
  const double d_10 = J[2]*J[7] - J[1]*J[8]; \
  const double d_11 = J[0]*J[8] - J[2]*J[6]; \
  const double d_12 = J[1]*J[6] - J[0]*J[7]; \
  const double d_20 = J[1]*J[5] - J[2]*J[4]; \
  const double d_21 = J[2]*J[3] - J[0]*J[5]; \
  const double d_22 = J[0]*J[4] - J[1]*J[3]; \
  det = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; \
  K[0] = d_00 / det; \
  K[1] = d_10 / det; \
  K[2] = d_20 / det; \
  K[3] = d_01 / det; \
  K[4] = d_11 / det; \
  K[5] = d_21 / det; \
  K[6] = d_02 / det; \
  K[7] = d_12 / det; \
  K[8] = d_22 / det; } while(0)

/* Compute Jacobian inverse K for tensor product prism embedded in R^3 - identical to tetrahedron */
#define compute_jacobian_inverse_prism_3d compute_jacobian_inverse_tetrahedron_3d

/* Compute Jacobian inverse K for tensor product hexahedron embedded in R^3 - identical to tetrahedron */
#define compute_jacobian_inverse_hex_3d compute_jacobian_inverse_tetrahedron_3d

/* --- Compute facet edge lengths --- */

#define compute_facet_edge_length_tetrahedron_3d(facet, coordinate_dofs) \
  const unsigned int tetrahedron_facet_edge_vertices[4][3][2] = { \
    {{2, 3}, {1, 3}, {1, 2}}, \
    {{2, 3}, {0, 3}, {0, 2}}, \
    {{1, 3}, {0, 3}, {0, 1}}, \
    {{1, 2}, {0, 2}, {0, 1}}, \
    }; \
  double edge_lengths_sqr[3]; \
  for (unsigned int edge = 0; edge < 3; ++edge) \
  { \
    const unsigned int vertex0 = tetrahedron_facet_edge_vertices[facet][edge]; \
    const unsigned int vertex1 = tetrahedron_facet_edge_vertices[facet][edge][1]; \
    edge_lengths_sqr[edge] = (coordinate_dofs[vertex1 + 0] - coordinate_dofs[vertex0 + 0])*(coordinate_dofs[vertex1 + 0] - coordinate_dofs[vertex0 + 0]) \
                           + (coordinate_dofs[vertex1 + 4] - coordinate_dofs[vertex0 + 4])*(coordinate_dofs[vertex1 + 4] - coordinate_dofs[vertex0 + 4]) \
                           + (coordinate_dofs[vertex1 + 8] - coordinate_dofs[vertex0 + 8])*(coordinate_dofs[vertex1 + 8] - coordinate_dofs[vertex0 + 8]); \
  }

/* Compute min edge length in facet of tetrahedron embedded in R^3 */
#define compute_min_facet_edge_length_tetrahedron_3d(min_edge_length, facet, coordinate_dofs) \
  compute_facet_edge_length_tetrahedron_3d(facet, coordinate_dofs); \
  min_edge_length = sqrt(fmin(fmin(edge_lengths_sqr[1], edge_lengths_sqr[1]), edge_lengths_sqr[2]));

/* Compute max edge length in facet of tetrahedron embedded in R^3 */
/*
 * FIXME: we can't call compute_facet_edge_length_tetrahedron_3d again, so we
 * rely on the fact that max is always computed after min
 */
#define compute_max_facet_edge_length_tetrahedron_3d(max_edge_length, facet, coordinate_dofs) \
  max_edge_length = sqrt(fmax(fmax(edge_lengths_sqr[1], edge_lengths_sqr[1]), edge_lengths_sqr[2]));
