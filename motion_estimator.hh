/*** first version written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** continued by Thomas Schoenemann as an employee of Lund University, Sweden, February 2010 ***/

#ifndef MOTION_ESTIMATOR_HH
#define MOTION_ESTIMATOR_HH

#include "colorimage.hh"
#include "vector.hh"
#include "matrix.hh"

class MotionEstimator {
public:

  MotionEstimator(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second,
                  DifferenceType data_norm, NormType reg_norm, double alpha, bool linearized_functional, bool spline_mode,
                  double epsilon = 0.001);


  void compute_flow(uint nWarpingIter = 25);

  void compute_flow_multi_scale(uint nWarpingIter = 25);

  const Math3D::NamedTensor<double>& flow() const;

  double energy() const;

protected:

  double residual_norm() const;

  double hyp_energy(const Math3D::Tensor<double>& flow_update) const;

  double addon_energy(const Math3D::Tensor<double>& flow_update) const;

  void compute_linflow_gradient(Math3D::Tensor<double>& gradient) const;

  void compute_incrementflow_gradient(Math3D::Tensor<double>& gradient) const;

  /*** methods for l2-l2 ***/

  /* linearized */

  void compute_flow_horn_schunck_sor();

  void compute_flow_horn_schunck_cg();

  void compute_flow_horn_schunck_pcg();

  //void compute_flow_horn_schunck_gradient_descent();

  void compute_flow_horn_schunck_nesterov();

  /* incremental */

  void compute_flow_increment_horn_schunck_sor();

  void compute_flow_increment_horn_schunck_cg();

  void compute_flow_increment_horn_schunck_nesterov();

  /*** methods for tv-l2 ***/

  /* linearized */

  void compute_flow_tv_l2_sor();

  void compute_flow_tv_l2_nesterov();

  /* incremental */

  void compute_flow_increment_tv_l2_sor();

  void compute_flow_increment_tv_l2_nesterov();

  /*** methods for tv-l1 ***/

  /* linearized */

  void compute_flow_tv_l1_sor();

  /* incremental */

  void compute_flow_increment_tv_l1_sor();

  /*** wholistic methods from nonlinear optimization ***/

  /* incremental */

  void compute_flow_lbfgs(int L=5, uint nIter = 25);

  void compute_incrementflow_gradient_descent();

  void compute_incrementflow_lbfgs(int L=5);

  /* linearized */

  void compute_linflow_gradient_descent();

  void compute_linflow_lbfgs(int L=5);

  Math3D::ColorImage<float> first_;
  Math3D::ColorImage<float> second_; 
  //Math3D::ColorImage<float> warped_second_;

  NamedStorage1D<Math3D::NamedTensor<float> > gradient_; //one gradient-tensor for each color channel
  Math3D::NamedTensor<float> t_derivative_;
  
  NamedStorage1D<Math2D::Matrix<double> > second_spline_coeffs_;

  Math3D::NamedTensor<double> flow_;
  Math3D::NamedTensor<double> flow_increment_;

  DifferenceType data_norm_;
  NormType reg_norm_;
  double alpha_;
  double epsilon_; // value used for the respective form of smoothing
  double scale_;

  bool linearized_functional_;
  bool spline_mode_;

  uint xDim_;
  uint yDim_;
  uint zDim_;
};



#endif
