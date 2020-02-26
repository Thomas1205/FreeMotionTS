/**** written by Thomas Schoenemann as an employee of Lund University, June 2010 ****/

#ifndef MOTION_CONVEXPROG_HH
#define MOTION_CONVEXPROG_HH

#include "tensor.hh"

double motion_estimation_quadprog(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                  int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                  uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);

#ifdef HAS_CPLEX
double motion_estimation_quadprog_bcd(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                      int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                      uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);
#endif


double motion_estimation_convprog_nesterov(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                           int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                           uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                           double exponent, bool use_cuda = false);

double motion_estimation_convprog_nesterov_smoothapprox(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                                        int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                                        uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                                        double epsilon, bool use_cuda = false);

double motion_estimation_convprog_standardrelax_nesterov_smoothapprox(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                                                      int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                                                      uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                                                      double epsilon);

double motion_estimation_goldluecke_cremers(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity);

double motion_estimation_smoothabs_nesterov(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                            double epsilon, bool use_cuda = false);


#endif
