/**** written by Thomas Schoenemann as an employee of Lund University, February 2010 ****/

#ifndef MOTION_LP_HH
#define MOTION_LP_HH

#include "matrix.hh"
#include "tensor.hh"

double lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);

double lp_motion_estimation_standard_relax(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                           int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                           uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);


double conv_lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                 int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                 uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);

double implicit_conv_lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                          int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                          uint neighborhood, uint spacing, double lambda, Math3D::Tensor<double>& velocity,
                                          Math2D::Matrix<uint>* labeling = 0);

//block coordinate descent version
double lp_motion_estimation_bcd(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);


#endif
