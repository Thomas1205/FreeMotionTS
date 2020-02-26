/**** written by Thomas Schoenemann as an employee of Lund University, September 2010 ****/

#ifndef MOTION_TRWS
#define MOTION_TRWS

#include "tensor.hh"

double trws_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                              int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                              uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);

double message_passing_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                         int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                         uint neighborhood, double lambda, std::string method, 
                                         Math3D::Tensor<double>& velocity);

#endif
