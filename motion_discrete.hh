/**** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ****/

#ifndef MOTION_DISCRETE_HH
#define MOTION_DISCRETE_HH

#include "matrix.hh"
#include "tensor.hh"

double expmove_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                 int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                 uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                 Math2D::Matrix<uint>* labeling = 0);


#endif
