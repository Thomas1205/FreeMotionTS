/** -*-c++-*- **/
/**** written by Thomas Schoenemann as an employee of Lund University, September 2010 ****/

#ifndef CUDA_CONV_LP_CUH
#define CUDA_CONV_LP_CUH


template<typename T>
void cuda_eq_constrained_lp_solving_auglagrange_nesterov(uint nVars, uint nConstraints, const double* cost, const double* var_lb, const double* var_ub,
							 const T* row_sorted_coeffs, const uint* column_indices, const uint* row_start, 
							 const double* rhs,  double* solution, double start_penalty = 100.0, uint inner_iter = 1000,
							 uint outer_iter = 15, double stepsize_coeff = 1.0, double penalty_factor = 1.25);


template<typename T>
void cuda_eq_and_simplex_constr_lp_solving_auglag_nesterov(uint nVars, uint nConstraints, const double* cost, uint nSimplices, uint simplex_width,
							   const double* var_lb, const double* var_ub,
							   const T* row_sorted_coeffs, const uint* column_indices, const uint* row_start, 
							   const double* rhs,  double* solution, double start_penalty = 100.0, uint inner_iter = 1000,
							   uint outer_iter = 15, double stepsize_coeff = 1.0, double penalty_factor = 1.25);


#endif
