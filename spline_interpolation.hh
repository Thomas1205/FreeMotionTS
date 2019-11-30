/**** written by Thomas Schoenemann as an employee of Lund University, Sweden, 2011 ****/

#ifndef SPLINE_INTERPOLATION_HH
#define SPLINE_INTERPOLATION_HH

#include "vector.hh"
#include "matrix.hh"
#include "tensor.hh"

double quadratic_spline_1D(double x);

double quadratic_spline_prime_1D(double x);

double cubic_spline_1D(double x);

double cubic_spline_prime_1D(double x);

void compute_qspline_coefficients(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficients);

//comparing with symmetric discrete gradients
void compute_qspline_coefficients_overdetermined(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficients);

//comparing with asymmetric discrete gradients
void compute_qspline_coefficients_overdetermined2(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficients);

void compute_cubic_spline_coefficients(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficients);

void compute_cubic_spline_coefficients_overdetermined2(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficients);
			   
double interpolate_qspline(const Math1D::Vector<double>& coefficients, double x);

double interpolate_cubic_spline(const Math1D::Vector<double>& coefficients, double x);

/************** 2-D functionality ***************/

double quadratic_spline_xprime_2D(double x, double y);

double quadratic_spline_yprime_2D(double x, double y);

double interpolate_2D_qspline(const Math2D::Matrix<double>& coefficients, double x, double y);

Math1D::Vector<double> interpolate_2D_qspline_grad(const Math2D::Matrix<double>& coefficients, double x, double y);

template<typename T>
void compute_2D_qspline_grad(const Math2D::Matrix<double>& coefficients, Math3D::Tensor<T>& grad); 

void compute_2D_qspline_coefficients(const Math2D::Matrix<float>& input, Math2D::Matrix<double>& coefficients);

//comparing with asymmetric discrete gradients
void compute_2D_qspline_coefficients_overdetermined2(const Math2D::Matrix<float>& input, Math2D::Matrix<double>& coefficients);

/************* implementation of templates ***************/

template<typename T>
void compute_2D_qspline_grad(const Math2D::Matrix<double>& coefficients, Math3D::Tensor<T>& grad) {

  uint xDim = coefficients.xDim();
  uint yDim = coefficients.yDim();

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Math1D::Vector<double> cur_grad = interpolate_2D_qspline_grad(coefficients,x,y);

      grad(x,y,0) = cur_grad[0];
      grad(x,y,1) = cur_grad[1];
    }
  }
}


#endif
