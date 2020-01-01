/*** first version written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** continued by Thomas Schoenemann as an employee of Lund University, Sweden, 2010 and 2011 ***/

#include "motion_estimator.hh"
#include "gradient.hh"
#include "matrix.hh"
#include "matrix_inversion.hh"
#include "tensor_interpolation.hh"
#include "spline_interpolation.hh"
#include "storage_util.hh"
#include "sampling.hh"

//#define LIPSCHITZ_SMOOTHING

MotionEstimator::MotionEstimator(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second,
                                 DifferenceType data_norm, NormType reg_norm, double alpha, bool linearized_functional, 
                                 bool spline_mode, double epsilon) :
  first_(first), second_(second), flow_(MAKENAME(flow_)), flow_increment_(MAKENAME(flow_increment_)),
  data_norm_(data_norm), reg_norm_(reg_norm), alpha_(alpha), epsilon_(epsilon), scale_(1.0),
  linearized_functional_(linearized_functional), spline_mode_(spline_mode) {


  xDim_ = first.xDim();
  yDim_ = first.yDim();
  zDim_ = first.zDim();

  assert(second.xDim() == xDim_);
  assert(second.yDim() == yDim_);
  assert(second.zDim() == zDim_);

  flow_.resize(xDim_,yDim_,2, 0.0);
  flow_increment_.resize(xDim_,yDim_,2, 0.0);

  t_derivative_ = second_ - first_;

  second_spline_coeffs_.resize(zDim_);
  gradient_.resize(zDim_);

  for (uint z=0; z < zDim_; z++) {
    gradient_[z].set_name("gradient_("+toString(z)+")");
    gradient_[z].resize(xDim_,yDim_,2,0.0);
  }

  if (linearized_functional_) {
    if (!spline_mode_) {
      compute_channel_mean_gradients(first_,second_,gradient_);
    }
    else {

      for (uint z=0; z < zDim_; z++) {

        Math2D::Matrix<float> temp(xDim_,yDim_);
	
        for (uint y=0; y < yDim_; y++)
          for (uint x=0; x < xDim_; x++)
            temp(x,y) = first(x,y,z);

        Math2D::Matrix<double> coefficients(xDim_,yDim_);
        compute_2D_qspline_coefficients_overdetermined2(temp, coefficients);

        for (uint y=0; y < yDim_; y++)
          for (uint x=0; x < xDim_; x++)
            temp(x,y) = second(x,y,z);

        Math2D::Matrix<double> coefficients2(xDim_,yDim_);
        compute_2D_qspline_coefficients_overdetermined2(temp, coefficients2);

        coefficients *= 0.5;
        coefficients2 *= 0.5;
        coefficients += coefficients2;

        compute_2D_qspline_grad(coefficients, gradient_[z]);
      }
    }
  }
  else {
    if (!spline_mode_) {
      compute_channel_gradients(second_,gradient_);
    }
    else {

      for (uint z=0; z < zDim_; z++) {
        Math2D::Matrix<float> temp(xDim_,yDim_);
	
        for (uint y=0; y < yDim_; y++)
          for (uint x=0; x < xDim_; x++)
            temp(x,y) = second(x,y,z);

        compute_2D_qspline_coefficients_overdetermined2(temp, second_spline_coeffs_[z]);

        compute_2D_qspline_grad(second_spline_coeffs_[z], gradient_[z]);
      }
    }
  }
}

const Math3D::NamedTensor<double>& MotionEstimator::flow() const {
  return flow_;
}

void MotionEstimator::compute_flow(uint nWarpingIter) {

  std::cerr.precision(8);

  if (linearized_functional_) {

    std::cerr << "start energy: " << energy() << std::endl;

    if (data_norm_ == SquaredDiffs && reg_norm_ == L2) {

      compute_flow_horn_schunck_sor();
      //compute_flow_horn_schunck_cg();
      //compute_flow_horn_schunck_nesterov();
      //compute_linflow_gradient_descent();
      //compute_linflow_lbfgs();
    }
    else  if (data_norm_ == SquaredDiffs && reg_norm_ == L1) {
      
      compute_flow_tv_l2_sor();
      //compute_flow_tv_l2_nesterov();
      //compute_linflow_lbfgs();
    }
    else if (data_norm_ == AbsDiffs && (reg_norm_ == L1 || reg_norm_ == L0_5)) {
      compute_flow_tv_l1_sor();
      //compute_linflow_lbfgs();
    }
    else {
      compute_linflow_lbfgs();
    }

    std::cerr << "end energy: " << energy() << std::endl;
  }
  else {
    //warping / Taylor-expansions

    //TEMP
    //compute_flow_lbfgs(5,10*nWarpingIter);
    //return;
    //END_TEMP

    NamedStorage1D<Math3D::NamedTensor<float> > second_gradient(zDim_,MAKENAME(second_gradient));
    for (uint z=0; z < zDim_; z++) {
      second_gradient[z].set_name("second_gradient["+toString(z)+"]");
      second_gradient[z].resize(xDim_,yDim_,2,0.0);
    }
    
    compute_channel_gradients(second_,second_gradient);
    
    for (uint iter=0; iter < nWarpingIter; iter++) {
      
      std::cerr << "********** iter " << iter << std::endl;
      
      //compute gradients for the Taylor expansion
      for (uint y=0; y < yDim_; y++) {
        for (uint x=0; x < xDim_; x++) {
	  
          float tx = x + flow_(x,y,0);
          float ty = y + flow_(x,y,1);
	  
          for (uint z=0; z < zDim_; z++) {
	    
            if (tx < 0.0 || tx > xDim_-1.0 || ty < 0.0 || ty > yDim_-1.0) {
              gradient_[z](x,y,0) = 0.0;
              gradient_[z](x,y,1) = 0.0;
              t_derivative_(x,y,z) = 0.0;
            }
            else {
              if (!spline_mode_) {
                gradient_[z](x,y,0) = bilinear_interpolation(second_gradient[z],tx,ty,0);
                gradient_[z](x,y,1) = bilinear_interpolation(second_gradient[z],tx,ty,1);
		
                t_derivative_(x,y,z) = bilinear_interpolation(second_,tx,ty,z) - first_(x,y,z);
              }
              else {
		
                Math1D::Vector<double> cur_grad = interpolate_2D_qspline_grad(second_spline_coeffs_[z],tx,ty);
		
                gradient_[z](x,y,0) = cur_grad[0];
                gradient_[z](x,y,1) = cur_grad[1];
                t_derivative_(x,y,z) =  interpolate_2D_qspline(second_spline_coeffs_[z],tx,ty) - first_(x,y,z);
              }
            }
          }
        }
      }

      //compute a proposal flow increment
	
      if (data_norm_ == SquaredDiffs && reg_norm_ == L2) {
        //compute_flow_increment_horn_schunck_cg();
        compute_flow_increment_horn_schunck_sor();
        //compute_flow_increment_horn_schunck_nesterov();
	//compute_incrementflow_gradient_descent();
	//compute_incrementflow_lbfgs();
      }
      else if (data_norm_ == SquaredDiffs && reg_norm_ == L1) {
        compute_flow_increment_tv_l2_sor();
        //compute_flow_increment_tv_l2_nesterov();
	//compute_incrementflow_gradient_descent();
	//compute_incrementflow_lbfgs();
      }
      else if (data_norm_ == AbsDiffs && (reg_norm_ == L1 || reg_norm_ == L0_5)) {
	compute_flow_increment_tv_l1_sor();
      }
      else {
	compute_incrementflow_lbfgs();
      }

      double base_energy = energy();
      double gain = -100.0;
      
      uint inner_iter = 0;
      
      std::cerr << "previous energy: " << base_energy << std::endl;
      
      while (inner_iter < 10 && gain < 0.0) {
	
        inner_iter++;
	
        //check if the increment reduces the energy
        double hyp_new_energy = hyp_energy(flow_increment_);
	
        gain = base_energy - hyp_new_energy;
	
        std::cerr << "new energy: " << hyp_new_energy << std::endl;
	
        std::cerr << "gain: " << gain << std::endl;
	
	
        if (gain >= 0.0)
          flow_ += flow_increment_;
        else
          flow_increment_ *= 0.5;
      }
      
      if (gain < 0.0)
        break;
    }
  }

  if (linearized_functional_)
    std::cerr << "remaining residual norm: " << residual_norm() << std::endl;
}


void MotionEstimator::compute_flow_multi_scale(uint nWarpingIter) {

  if (!linearized_functional_ && xDim_ > 64) {

    Math3D::ColorImage<float> orig_first = first_;
    Math3D::ColorImage<float> orig_second = second_;

    scale_ = 64.0 / xDim_;

    xDim_ = round(scale_ * orig_first.xDim());
    yDim_ = round(scale_ * orig_first.yDim());

    flow_.resize(xDim_,yDim_,2);
    flow_.set_constant(0.0);

    while (scale_ < 1.0) {

      xDim_ = round(scale_ * orig_first.xDim());
      yDim_ = round(scale_ * orig_first.yDim());

      std::cerr << "**************** scale " << scale_ << " (" << xDim_ << "x" << yDim_ << ")" << std::endl;
      first_.resize(xDim_,yDim_,zDim_);
      second_.resize(xDim_,yDim_,zDim_);

      downsample_tensor(orig_first, first_);
      downsample_tensor(orig_second, second_);

      t_derivative_.resize(xDim_,yDim_,zDim_);
      for (uint z=0; z < zDim_; z++) {
	gradient_[z].resize(xDim_,yDim_,2);
	second_spline_coeffs_[z].resize(xDim_,yDim_);
      }
      
	
      if (!spline_mode_) {
	compute_channel_gradients(second_,gradient_);
      }
      else {
	
	for (uint z=0; z < zDim_; z++) {
	  Math2D::Matrix<float> temp(xDim_,yDim_);
	  
	  for (uint y=0; y < yDim_; y++)
	    for (uint x=0; x < xDim_; x++)
	      temp(x,y) = second_(x,y,z);
	  
	  compute_2D_qspline_coefficients_overdetermined2(temp, second_spline_coeffs_[z]);
	  
	  compute_2D_qspline_grad(second_spline_coeffs_[z], gradient_[z]);
	}
      }


      Math3D::Tensor<double> upsample_flow(xDim_,yDim_,2);
      upsample_tensor(flow_, upsample_flow);
      
      const double xfac = ((double) xDim_) / flow_.xDim();
      const double yfac = ((double) yDim_) / flow_.yDim();

      for (uint y=0; y < yDim_; y++) {
	for (uint x=0; x < xDim_; x++) {

	  upsample_flow(x,y,0) *= xfac;
	  upsample_flow(x,y,1) *= yfac;
	}
      }
	
      flow_ = upsample_flow;

      flow_increment_.resize(xDim_,yDim_,2);

      compute_flow(3);

      scale_ *= 1.05;
    }

    //restore everyting to original size
    first_ = orig_first;
    second_ = orig_second;
    xDim_ = first_.xDim();
    yDim_ = first_.yDim();


    t_derivative_.resize(xDim_,yDim_,zDim_);
    for (uint z=0; z < zDim_; z++) {
      gradient_[z].resize(xDim_,yDim_,zDim_);
      second_spline_coeffs_[z].resize(xDim_,yDim_);
    }
    
    if (!spline_mode_) {
      compute_channel_gradients(second_,gradient_);
    }
    else {
      
      for (uint z=0; z < zDim_; z++) {
	Math2D::Matrix<float> temp(xDim_,yDim_);
	
	for (uint y=0; y < yDim_; y++)
	  for (uint x=0; x < xDim_; x++)
	    temp(x,y) = second_(x,y,z);
	
	compute_2D_qspline_coefficients_overdetermined2(temp, second_spline_coeffs_[z]);
	
	compute_2D_qspline_grad(second_spline_coeffs_[z], gradient_[z]);
      }
    }

    if (xDim_ != flow_.xDim() || yDim_ != flow_.yDim()) {

      Math3D::Tensor<double> upsample_flow(xDim_,yDim_,2);
      upsample_tensor(flow_, upsample_flow);
      
      const double xfac = ((double) xDim_) / flow_.xDim();
      const double yfac = ((double) yDim_) / flow_.yDim();

      for (uint y=0; y < yDim_; y++) {
	for (uint x=0; x < xDim_; x++) {

	  upsample_flow(x,y,0) *= xfac;
	  upsample_flow(x,y,1) *= yfac;
	}
      }
	
      flow_ = upsample_flow;
      flow_increment_.resize(xDim_,yDim_,2);

      scale_ = 1.0;
    }

  }

  compute_flow(nWarpingIter);
}

double MotionEstimator::residual_norm() const {

  assert(data_norm_ == SquaredDiffs);

  double residual_sum = 0.0;
  
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
      
      double u = flow_(x,y,0);
      double v = flow_(x,y,1);
      
      double sum_u = 0.0;
      double sum_v = 0.0;

      if (reg_norm_ == L2) {
        if (x > 0) {
          sum_u += u - flow_(x-1,y,0);
          sum_v += v - flow_(x-1,y,1);
        }
        if ((x+1) < xDim_) {
          sum_u += u - flow_(x+1,y,0);
          sum_v += v - flow_(x+1,y,1);
        }
        if (y > 0) {
          sum_u += u - flow_(x,y-1,0);
          sum_v += v - flow_(x,y-1,1);
        }
        if ((y+1) < yDim_) {
          sum_u += u - flow_(x,y+1,0);
          sum_v += v - flow_(x,y+1,1);
        }
      }
      else {

        if (x != 0 || y != 0) {

          double hdiff = (x > 0) ? u - flow_(x-1,y,0) : 0.0;
          double vdiff = (y > 0) ? u - flow_(x,y-1,0) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_u += (0.5 * (hdiff+vdiff) ) / norm;
	    else
	      sum_u += 0.5 * (hdiff+vdiff) / epsilon_;
#else
	    sum_u += 0.5*(hdiff+vdiff) / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
	  }
	  else
	    sum_u += 0.25*(hdiff+vdiff) * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);
	  
	  
          hdiff = (x > 0) ? v - flow_(x-1,y,1) : 0.0;
          vdiff = (y > 0) ? v - flow_(x,y-1,1) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_v += (0.5 * (hdiff+vdiff) ) / norm;
	    else
	      sum_v += 0.5 * (hdiff+vdiff) / epsilon_;
#else
	    sum_v += 0.5*(hdiff+vdiff) / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
	  }
	  else
	    sum_v += 0.25*(hdiff+vdiff) * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);
        }

        if (x+1 < xDim_) {

          double hdiff = flow_(x+1,y,0) - u;
          double vdiff = (y > 0) ? flow_(x+1,y,0) - flow_(x+1,y-1,0) : 0.0;
	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            double norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_u -= (0.5 * hdiff ) / norm;
	    else
	      sum_u -=  0.5 * hdiff / epsilon_;
#else
	    sum_u -= 0.5*hdiff / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
	  }
	  else
	    sum_u += 0.25*hdiff * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);

          hdiff = flow_(x+1,y,1) - v;
          vdiff = (y > 0) ? flow_(x+1,y,1) - flow_(x+1,y-1,1) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_v -= (0.5 * hdiff ) / norm;
	    else
	      sum_v -= 0.5 * hdiff / epsilon_;
#else
	    sum_v -= 0.5*hdiff / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
	  }
	  else
	    sum_v += 0.25*hdiff * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);
        }

        if (y+1 < yDim_) {

          double hdiff = (x > 0) ? flow_(x-1,y+1,0) - flow_(x,y+1,0) : 0.0; 
          double vdiff = flow_(x,y+1,0) - u;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            double norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_u -= (0.5 * vdiff ) / norm;
	    else
	      sum_u -= 0.5 * vdiff / epsilon_;
#else
	    sum_u -= 0.5*vdiff / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
 	  }
	  else
	    sum_u += 0.25*vdiff * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);

          hdiff = (x > 0) ? flow_(x-1,y+1,1) - flow_(x,y+1,1) : 0.0; 
          vdiff = flow_(x,y+1,1) - v;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            norm = sqrt(vdiff*vdiff + hdiff*hdiff);
	    if (norm >= epsilon_)
	      sum_v -= (0.5 * vdiff ) / norm;
	    else
	      sum_v -= 0.5 * vdiff / epsilon_;
#else
	    sum_v -= 0.5*vdiff / sqrt(vdiff*vdiff + hdiff*hdiff + epsilon_);
#endif
	  }
	  else
	    sum_v += 0.25*vdiff * std::pow(vdiff*vdiff + hdiff*hdiff + epsilon_,-0.75);
        }
      }

      sum_u *= alpha_;
      sum_v *= alpha_;

      for (uint z=0; z < zDim_; z++) {
	
        double dx = gradient_[z](x,y,0);
        double dy = gradient_[z](x,y,1);
        double dt = t_derivative_(x,y,z);

	if (data_norm_ == SquaredDiffs) {
	  sum_u += dx*(dx*u + dy*v + dt);
	  sum_v += dy*(dx*u + dy*v + dt);
	}
	else {
	  double temp = dx*u + dy*v + dt;
	  temp = 2.0 * sqrt(temp*temp + epsilon_);
	  sum_u += dx / temp;
	  sum_v += dy / temp;
	}
      }
      
      residual_sum += sum_u*sum_u + sum_v*sum_v;
    }
  }

  return sqrt(residual_sum);
}

double MotionEstimator::energy() const {

  double data_energy = 0.0;
  double smooth_energy = 0.0;

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      const double u = flow_(x,y,0);
      const double v = flow_(x,y,1);
      
      double sum = 0.0;
      for (uint z=0; z < zDim_; z++) {

        double temp=0.0;

        if (linearized_functional_) {
          temp = gradient_[z](x,y,0)*u + gradient_[z](x,y,1)*v + t_derivative_(x,y,z);
        }
        else {
          if (!spline_mode_)
            temp = first_(x,y,z) - bilinear_interpolation(second_,x+u,y+v,z);
          else {
            temp = first_(x,y,z) - interpolate_2D_qspline(second_spline_coeffs_[z], x+u,y+v);
          }
        }

	if (data_norm_ == SquaredDiffs)
	  sum += temp*temp;
	else {
#ifdef LIPSCHITZ_SMOOTHING
          double norm = fabs(temp);
          if (norm > epsilon_)
            sum += norm - 0.5*epsilon_;
          else
            sum += norm*norm / (2.0*epsilon_);
#else
	  sum += sqrt(temp*temp+epsilon_);
#endif
	}
      }

      data_energy += sum;

      if (reg_norm_ == L2) {
	if (x > 0) {
	  const double diff_u = u - flow_(x-1,y,0);
	  const double diff_v = v - flow_(x-1,y,1);
	
	  smooth_energy += diff_u*diff_u + diff_v*diff_v;
	}
	if (y > 0) {  
	  const double diff_u = u - flow_(x,y-1,0);
	  const double diff_v = v - flow_(x,y-1,1);
	  
	  smooth_energy += diff_u*diff_u + diff_v*diff_v;
	}
      }
    }
  }

  if (reg_norm_ == L1 || reg_norm_ == L0_5) {

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double hdiff = (x > 0) ? flow_(x,y,i) - flow_(x-1,y,i) : 0.0;
          double vdiff = (y > 0) ? flow_(x,y,i) - flow_(x,y-1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            double norm = sqrt(hdiff*hdiff + vdiff*vdiff);
	    if (norm > epsilon_)
	      smooth_energy += norm - 0.5*epsilon_;
	    else
	      smooth_energy += norm*norm / (2.0*epsilon_);
#else
	    smooth_energy += sqrt(hdiff*hdiff + vdiff*vdiff + epsilon_);
#endif
	  }
	  else
	    smooth_energy += std::pow(hdiff*hdiff + vdiff*vdiff + epsilon_,0.25);
        }
      }
    }
  }

  return data_energy + alpha_*smooth_energy;
}

double MotionEstimator::addon_energy(const Math3D::Tensor<double>& flow_update) const {

  double data_energy = 0.0;
  double smooth_energy = 0.0;

  assert(!linearized_functional_);

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double u = flow_update(x,y,0);
      double v = flow_update(x,y,1);
      
      double sum = 0.0;
      for (uint z=0; z < zDim_; z++) {

        double temp=0.0;

	temp = gradient_[z](x,y,0)*u + gradient_[z](x,y,1)*v + t_derivative_(x,y,z);

	if (data_norm_ == SquaredDiffs)
	  sum += temp*temp;
	else {
#ifdef LIPSCHITZ_SMOOTHING
          double norm = fabs(temp);
          if (norm > epsilon_)
            sum += norm - 0.5*epsilon_;
          else
            sum += norm*norm / (2.0*epsilon_);
#else
	  sum += sqrt(temp*temp+epsilon_);
#endif
	}
      }

      data_energy += sum;

      if (reg_norm_ == L2) {
	if (x > 0) {
	  const double diff_u = flow_(x,y,0) + u - flow_(x-1,y,0) - flow_update(x-1,y,0);
	  const double diff_v = flow_(x,y,1) + v - flow_(x-1,y,1) - flow_update(x-1,y,1);
	
	  smooth_energy += diff_u*diff_u + diff_v*diff_v;
	}
	if (y > 0) {  
	  const double diff_u = flow_(x,y,0) + u - flow_(x,y-1,0) - flow_update(x,y-1,0);
	  const double diff_v = flow_(x,y,1) + v - flow_(x,y-1,1) - flow_update(x,y-1,1);
	  
	  smooth_energy += diff_u*diff_u + diff_v*diff_v;
	}
      }
    }
  }

  if (reg_norm_ == L1 || reg_norm_ == L0_5) {

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double hdiff = (x > 0) ? flow_(x,y,i) + flow_update(x,y,i) - flow_(x-1,y,i) - flow_update(x-1,y,i) : 0.0;
          double vdiff = (y > 0) ? flow_(x,y,i) + flow_update(x,y,i) - flow_(x,y-1,i) - flow_update(x,y-1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            double norm = sqrt(hdiff*hdiff + vdiff*vdiff);
	    if (norm > epsilon_)
	      smooth_energy += norm - 0.5*epsilon_;
	    else
	      smooth_energy += norm*norm / (2.0*epsilon_);
#else
	    smooth_energy += sqrt(hdiff*hdiff + vdiff*vdiff + epsilon_);
#endif
	  }
	  else
	    smooth_energy += std::pow(hdiff*hdiff + vdiff*vdiff + epsilon_,0.25);
        }
      }
    }

  }  

  return data_energy + alpha_*smooth_energy;  
}

double MotionEstimator::hyp_energy(const Math3D::Tensor<double>& flow_update) const {

  double data_energy = 0.0;
  double smooth_energy = 0.0;

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double u,v;

      if (linearized_functional_) {
	u = flow_update(x,y,0);
	v = flow_update(x,y,1);
      }
      else {
	u = flow_(x,y,0) + flow_update(x,y,0);
	v = flow_(x,y,1) + flow_update(x,y,1);
      }      

      double sum = 0.0;
      for (uint z=0; z < zDim_; z++) {
        double temp = 0.0;

        if (linearized_functional_)
          temp = gradient_[z](x,y,0)*u + gradient_[z](x,y,1)*v + t_derivative_(x,y,z);
        else {
          if (!spline_mode_)
            temp = first_(x,y,z) - bilinear_interpolation(second_,x+u,y+v,z);
          else {
            temp = first_(x,y,z) - interpolate_2D_qspline(second_spline_coeffs_[z], x+u,y+v);
          }
        }
	  
	if (data_norm_ == SquaredDiffs)
	  sum += temp*temp;
	else {
#ifdef LIPSCHITZ_SMOOTHING
          double norm = fabs(temp);
          if (norm > epsilon_)
            sum += norm - 0.5*epsilon_;
          else
            sum += norm*norm / (2.0*epsilon_);
#else
	  sum += sqrt(temp*temp + epsilon_);
#endif
	}
      }
      data_energy += sum;

      if (reg_norm_ == L2) {
	double diff_u,diff_v;

        if (x > 0) {
	  if (linearized_functional_) {
	    diff_u = u - flow_update(x-1,y,0);
	    diff_v = v - flow_update(x-1,y,1);
	  }
	  else {
	    diff_u = u - flow_(x-1,y,0) - flow_update(x-1,y,0);
	    diff_v = v - flow_(x-1,y,1) - flow_update(x-1,y,1);
	  }	  

          smooth_energy += diff_u*diff_u + diff_v*diff_v;
        }
        if (y > 0) {

	  if (linearized_functional_) {
	    diff_u = u - flow_update(x,y-1,0);
	    diff_v = v - flow_update(x,y-1,1);
	  }
	  else {
	    diff_u = u - flow_(x,y-1,0) - flow_update(x,y-1,0);
	    diff_v = v - flow_(x,y-1,1) - flow_update(x,y-1,1);
	  }	  

          smooth_energy += diff_u*diff_u + diff_v*diff_v;
        }
      }
    }
  }

  if (reg_norm_ == L1 || reg_norm_ == L0_5) {

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
	  double hdiff,vdiff;

	  if (linearized_functional_) {
	    hdiff = (x > 0) ? flow_update(x,y,i) - flow_update(x-1,y,i) : 0.0;
	    vdiff = (y > 0) ? flow_update(x,y,i) - flow_update(x,y-1,i) : 0.0;
	  }
	  else {
	    hdiff = (x > 0) ? flow_(x,y,i) + flow_update(x,y,i) - flow_(x-1,y,i) - flow_update(x-1,y,i) : 0.0;
	    vdiff = (y > 0) ? flow_(x,y,i) + flow_update(x,y,i) - flow_(x,y-1,i) - flow_update(x,y-1,i) : 0.0;
	  }

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
            double norm = sqrt(hdiff*hdiff + vdiff*vdiff);
	    if (norm > epsilon_)
	      smooth_energy += norm - 0.5*epsilon_;
	    else
	      smooth_energy += norm*norm / (2.0*epsilon_);
#else
	    smooth_energy += sqrt(hdiff*hdiff + vdiff*vdiff + epsilon_);
#endif
	  }
	  else
	    smooth_energy += std::pow(hdiff*hdiff + vdiff*vdiff + epsilon_,0.25);
        }
      }
    }
  }


  return data_energy + alpha_*smooth_energy;
}

void MotionEstimator::compute_flow_horn_schunck_sor() {

  assert(linearized_functional_);

  flow_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint iter = 1; iter <= 1000; iter++) {

    if (scale_ == 1.0)
      std::cerr << "iter: " << iter << ", energy: " << energy() << std::endl;
    double res_norm = residual_norm();
    if (res_norm < 1e-8)
      break;
    
    for (uint z=0; z < zDim_; z++) {
      for (uint y=0; y < yDim_; y++) {
        for (uint x=0; x < xDim_; x++) {

          //update u
          double sum_u = 0.0;
          double denom = 0.0;

          for (uint z=0; z < zDim_; z++) {
	    
            double dx = gradient_[z](x,y,0);
            double dy = gradient_[z](x,y,1);
            double dt = t_derivative_(x,y,z);
	    
            sum_u -= dx*(dy*flow_(x,y,1) + dt);
            denom += dx*dx;
          }
	  
          if (x > 0) {
            sum_u += alpha_*flow_(x-1,y,0);
            denom += alpha_;
          }
          if (x+1 < xDim_) {
            sum_u += alpha_*flow_(x+1,y,0);
            denom += alpha_;
          }
          if (y > 0) {
            sum_u += alpha_*flow_(x,y-1,0);
            denom += alpha_;
          }
          if (y+1 < yDim_) {
            sum_u += alpha_*flow_(x,y+1,0);
            denom += alpha_;
          }

          sum_u /= denom;
          flow_(x,y,0) = neg_omega * flow_(x,y,0) + omega *sum_u;

          //update v
          double sum_v = 0.0;
          denom = 0.0;

          for (uint z=0; z < zDim_; z++) {
	    
            double dx = gradient_[z](x,y,0);
            double dy = gradient_[z](x,y,1);
            double dt = t_derivative_(x,y,z);
	    
            sum_v -= dy*(dx*flow_(x,y,0)  + dt);
            denom += dy*dy;
          }

          if (x > 0) {
            sum_v += alpha_*flow_(x-1,y,1);
            denom += alpha_;
          }
          if (x+1 < xDim_) {
            sum_v += alpha_*flow_(x+1,y,1);
            denom += alpha_;
          }
          if (y > 0) {
            sum_v += alpha_*flow_(x,y-1,1);
            denom += alpha_;
          }
          if (y+1 < yDim_) {
            sum_v += alpha_*flow_(x,y+1,1);
            denom += alpha_;
          }

          sum_v /= denom;
          flow_(x,y,1) = neg_omega * flow_(x,y,1) + omega * sum_v;
        }
      }
    }
  }
}


void MotionEstimator::compute_flow_horn_schunck_cg() {

  flow_.set_constant(0.0);

  Math3D::NamedTensor<double> rhs(xDim_,yDim_,2,MAKENAME(rhs));
  Math3D::NamedTensor<double> dsquare(xDim_,yDim_,2,0.0,MAKENAME(rhs));
  Math2D::NamedMatrix<double> dcross(xDim_,yDim_,0.0,MAKENAME(dcross));

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double sum_u = 0.0;
      double sum_v = 0.0;
      for (uint z=0; z < zDim_; z++) {
        sum_u -= t_derivative_(x,y,z)*gradient_[z](x,y,0);
        sum_v -= t_derivative_(x,y,z)*gradient_[z](x,y,1);      
      }

      rhs(x,y,0) = sum_u;
      rhs(x,y,1) = sum_v;

      for (uint z=0; z < zDim_; z++) {
        double dx = gradient_[z](x,y,0);
        double dy = gradient_[z](x,y,1);
        dsquare(x,y,0) += dx*dx;
        dsquare(x,y,1) += dy*dy;
        dcross(x,y) += dx*dy;
      }
    }
  }

  Math3D::NamedTensor<double> residuum(xDim_,yDim_,2,MAKENAME(residuum));
  residuum = rhs;

  Math3D::NamedTensor<double> direction(MAKENAME(direction));
  direction = residuum;
  
  Math3D::NamedTensor<double> temp_dir(xDim_,yDim_,2,0.0,MAKENAME(temp_dir));

  double sum_u;
  double sum_v;
  
  double fac[2];
  fac[0] = alpha_;
  fac[1] = alpha_;

  double dcr[2];

  double res_sqr_norm = residuum.sqr_norm();

  for (uint n=1; n <= xDim_*yDim_*2; n++) {

    asm volatile ("movupd %[fac], %%xmm2" : : [fac] "m" (fac[0]) : "xmm2");

    prefetcht0(direction.direct_access());
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        //#ifndef USE_SSE
#if 1
        double u = direction(x,y,0);
        double v = direction(x,y,1);

        sum_u = 0.0;
        sum_v = 0.0;
	
        if (x > 0) {
          sum_u += u - direction(x-1,y,0);
          sum_v += v - direction(x-1,y,1);
        }
        if (x+1 < xDim_) {
          sum_u += u - direction(x+1,y,0);
          sum_v += v - direction(x+1,y,1);
        }
        if (y > 0) {
          sum_u += u - direction(x,y-1,0);
          sum_v += v - direction(x,y-1,1);
        }
        if (y+1 < yDim_) {
          sum_u += u - direction(x,y+1,0);
          sum_v += v - direction(x,y+1,1);
        }

        sum_u *= alpha_;
        sum_v *= alpha_;

        double dc = dcross(x,y);
        sum_u += dsquare(x,y,0)*u + dc*v;
        sum_v += dc*u + dsquare(x,y,1)*v;

        temp_dir(x,y,0) = sum_u;
        temp_dir(x,y,1) = sum_v;
#else
        //EXPERIMENTAL RESULT: the compiler produces the faster code

        uint offset = 2*(y*xDim_+x);
        double* dptr;
        double* dptr2;
        asm volatile ("xorps %%xmm7,%%xmm7" : : : "xmm7"); //xmm7 = sum_u, sum_v (set to 0,0 here)
        dptr = direction.direct_access() + offset;
        asm volatile ("movupd %[dptr], %%xmm6" : : [dptr] "m" (dptr[0]) : "xmm6"); //xmm6 = u,v
	
        if (x > 0) {
          asm volatile ("addpd %%xmm6, %%xmm7" : : : "xmm7");
          dptr2 = dptr - 2;
          asm volatile ("movupd %[dptr2], %%xmm5" : : [dptr2] "m" (dptr2[0]) : "xmm5");
          asm volatile ("subpd %%xmm5, %%xmm7" : : : "xmm7");
        }
	
        if (x+1 < xDim_) {
          asm volatile ("addpd %%xmm6, %%xmm7" : : : "xmm7");
          dptr2 = dptr + 2;
          asm volatile ("movupd %[dptr2], %%xmm5" : : [dptr2] "m" (dptr2[0]) : "xmm5");
          asm volatile ("subpd %%xmm5, %%xmm7" : : : "xmm7");
        }
	
        if (y > 0) {
          asm volatile ("addpd %%xmm6, %%xmm7" : : : "xmm7");
          dptr2 = dptr-2*xDim_;
          asm volatile ("movupd %[dptr2], %%xmm5" : : [dptr2] "m" (dptr2[0]) : "xmm5");
          asm volatile ("subpd %%xmm5, %%xmm7" : : : "xmm7");
        }

        if (y+1 < yDim_) {
          asm volatile ("addpd %%xmm6, %%xmm7" : : : "xmm7");
          dptr2 = dptr + 2*xDim_;
          asm volatile ("movupd %[dptr2], %%xmm5" : : [dptr2] "m" (dptr2[0]) : "xmm5");
          asm volatile ("subpd %%xmm5, %%xmm7" : : : "xmm7");
        }

        asm volatile ("mulpd %%xmm2, %%xmm7" : : : "xmm7"); //xmm2 contains {alpha_,alpha_}

        dptr = dsquare.direct_access() + offset;
        asm volatile ("movupd %[dptr], %%xmm5" : : [dptr] "m" (dptr[0]) : "xmm5");
        asm volatile ("mulpd %%xmm6, %%xmm5" : : : "xmm5");
        asm volatile ("addpd %%xmm5, %%xmm7" : : : "xmm7");
	
        dcr[0] = dcross(x,y);
        dcr[1] = dcr[0];

        asm volatile ("movupd %[dcr], %%xmm5" : : [dcr] "m" (dcr[0]) : "xmm5");
        asm volatile ("mulpd %%xmm6, %%xmm5" : : : "xmm5");
        //difficulty here: the two values have to be swapped
        asm volatile ("shufpd $1, %%xmm5,%%xmm5" : : : "xmm5");
        asm volatile ("addpd %%xmm5, %%xmm7" : : : "xmm7");

        dptr = temp_dir.direct_access() + offset;
        asm volatile ("movupd %%xmm7, %[dptr]" : [dptr] "=m" (dptr[0]) : :);
#endif
      }
    }

    double denom = 0.0;
    for (uint i=0; i < direction.size(); i++)
      denom += temp_dir.direct_access(i) * direction.direct_access(i);

    prefetcht0(direction.direct_access());
    
    const double alpha_cg =  res_sqr_norm / denom;
	
    //update solution
    for (uint i=0; i < direction.size(); i++)
      flow_.direct_access(i) += alpha_cg * direction.direct_access(i);
    
    //update residuum
    //     for (uint i=0; i < direction.size(); i++)
    //        residuum.direct_access(i) -= alpha_cg * temp_dir.direct_access(i);
    Makros::array_subtract_multiple(residuum.direct_access(), residuum.size(), 
                                    alpha_cg, temp_dir.direct_access());

    const double new_res_sqr_norm = residuum.sqr_norm();

    prefetcht0(direction.direct_access());

    if ((n%100) == 0) {
      std::cerr << "current norm: " << sqrt(new_res_sqr_norm) << std::endl;
    }

    if (new_res_sqr_norm < 1e-5) {
      std::cerr << "break after " << n << " iterations." << std::endl; 
      break;
    }

    const double beta_cg = new_res_sqr_norm / res_sqr_norm;
    direction *= beta_cg;
    direction += residuum;

    res_sqr_norm = new_res_sqr_norm;

    prefetcht0(dsquare.direct_access());
  }

}

void MotionEstimator::compute_flow_increment_horn_schunck_sor() {

  std::cerr << "SOR_inc" << std::endl;

  flow_increment_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint iter = 1; iter <= 500; iter++) {
    
    if (scale_ == 1.0)
      std::cerr << "iter: " << iter << ", energy: " << addon_energy(flow_increment_) << std::endl;

    for (uint z=0; z < zDim_; z++) {
      for (uint y=0; y < yDim_; y++) {
        for (uint x=0; x < xDim_; x++) {

          //update u
          double sum_u = 0.0;
          double denom = 0.0;

          for (uint z=0; z < zDim_; z++) {
	    
            double dx = gradient_[z](x,y,0);
            double dy = gradient_[z](x,y,1);
            double dt = t_derivative_(x,y,z);
	    
            sum_u -= dx*(dy*flow_increment_(x,y,1) + dt);
            denom += dx*dx;
          }
	  
          if (x > 0) {
            sum_u += alpha_*(flow_increment_(x-1,y,0) + flow_(x-1,y,0) - flow_(x,y,0));
            denom += alpha_;
          }
          if (x+1 < xDim_) {
            sum_u += alpha_*(flow_increment_(x+1,y,0) + flow_(x+1,y,0) - flow_(x,y,0));
            denom += alpha_;
          }
          if (y > 0) {
            sum_u += alpha_*(flow_increment_(x,y-1,0) + flow_(x,y-1,0) - flow_(x,y,0));
            denom += alpha_;
          }
          if (y+1 < yDim_) {
            sum_u += alpha_*(flow_increment_(x,y+1,0) + flow_(x,y+1,0) - flow_(x,y,0));
            denom += alpha_;
          }

          sum_u /= denom;
          flow_increment_(x,y,0) = neg_omega * flow_increment_(x,y,0) + omega *sum_u;

          //update v
          double sum_v = 0.0;
          denom = 0.0;

          for (uint z=0; z < zDim_; z++) {
	    
            double dx = gradient_[z](x,y,0);
            double dy = gradient_[z](x,y,1);
            double dt = t_derivative_(x,y,z);
	    
            sum_v -= dy*(dx*flow_increment_(x,y,0)  + dt);
            denom += dy*dy;
          }

          if (x > 0) {
            sum_v += alpha_*(flow_increment_(x-1,y,1) + flow_(x-1,y,1) - flow_(x,y,1));
            denom += alpha_;
          }
          if (x+1 < xDim_) {
            sum_v += alpha_*(flow_increment_(x+1,y,1) + flow_(x+1,y,1) - flow_(x,y,1));
            denom += alpha_;
          }
          if (y > 0) {
            sum_v += alpha_*(flow_increment_(x,y-1,1) + flow_(x,y-1,1) - flow_(x,y,1));
            denom += alpha_;
          }
          if (y+1 < yDim_) {
            sum_v += alpha_*(flow_increment_(x,y+1,1) + flow_(x,y+1,1) - flow_(x,y,1));
            denom += alpha_;
          }

          sum_v /= denom;
          flow_increment_(x,y,1) = neg_omega * flow_increment_(x,y,1) + omega * sum_v;
        }
      }
    }
  }
}
  


void MotionEstimator::compute_flow_increment_horn_schunck_cg() {

  flow_increment_.set_constant(0.0);

  Math3D::NamedTensor<double> rhs(xDim_,yDim_,2,MAKENAME(rhs));
  Math3D::NamedTensor<double> dsquare(xDim_,yDim_,2,0.0,MAKENAME(rhs));
  Math2D::NamedMatrix<double> dcross(xDim_,yDim_,0.0,MAKENAME(dcross));

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double sum_u = 0.0;
      double sum_v = 0.0;
      for (uint z=0; z < zDim_; z++) {
        sum_u -= t_derivative_(x,y,z)*gradient_[z](x,y,0);
        sum_v -= t_derivative_(x,y,z)*gradient_[z](x,y,1);      

        if (x > 0) {
          sum_u += alpha_ * (flow_(x-1,y,0) - flow_(x,y,0));
          sum_v += alpha_ * (flow_(x-1,y,1) - flow_(x,y,1));
        }
        if (x+1 < xDim_) {
          sum_u += alpha_ * (flow_(x+1,y,0) - flow_(x,y,0));
          sum_v += alpha_ * (flow_(x+1,y,1) - flow_(x,y,1));
        }
        if (y > 0) {
          sum_u += alpha_ * (flow_(x,y-1,0) - flow_(x,y,0));
          sum_v += alpha_ * (flow_(x,y-1,1) - flow_(x,y,1));
        }
        if (y+1 < yDim_) {
          sum_u += alpha_ * (flow_(x,y+1,0) - flow_(x,y,0));
          sum_v += alpha_ * (flow_(x,y+1,1) - flow_(x,y,1));
        }
      }

      rhs(x,y,0) = sum_u;
      rhs(x,y,1) = sum_v;

      for (uint z=0; z < zDim_; z++) {
        double dx = gradient_[z](x,y,0);
        double dy = gradient_[z](x,y,1);
        dsquare(x,y,0) += dx*dx;
        dsquare(x,y,1) += dy*dy;
        dcross(x,y) += dx*dy;
      }
    }
  }

  Math3D::NamedTensor<double> residuum(xDim_,yDim_,2,MAKENAME(residuum));
  residuum = rhs;

  Math3D::NamedTensor<double> direction(MAKENAME(direction));
  direction = residuum;
  
  Math3D::NamedTensor<double> temp_dir(xDim_,yDim_,2,0.0,MAKENAME(temp_dir));

  double sum_u = 0.0;
  double sum_v = 0.0;

  double res_sqr_norm = residuum.sqr_norm();

  for (uint n=1; n <= xDim_*yDim_*2; n++) {

    prefetcht0(direction.direct_access());
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        double u = direction(x,y,0);
        double v = direction(x,y,1);

        sum_u = 0.0;
        sum_v = 0.0;
	
        if (x > 0) {
          sum_u += u - direction(x-1,y,0);
          sum_v += v - direction(x-1,y,1);
        }
        if (x+1 < xDim_) {
          sum_u += u - direction(x+1,y,0);
          sum_v += v - direction(x+1,y,1);
        }
        if (y > 0) {
          sum_u += u - direction(x,y-1,0);
          sum_v += v - direction(x,y-1,1);
        }
        if (y+1 < yDim_) {
          sum_u += u - direction(x,y+1,0);
          sum_v += v - direction(x,y+1,1);
        }

        sum_u *= alpha_;
        sum_v *= alpha_;

        double dc = dcross(x,y);
        sum_u += dsquare(x,y,0)*u + dc*v;
        sum_v += dc*u + dsquare(x,y,1)*v;

        temp_dir(x,y,0) = sum_u;
        temp_dir(x,y,1) = sum_v;

      }
    }  

    double denom = 0.0;
    for (uint i=0; i < direction.size(); i++)
      denom += temp_dir.direct_access(i) * direction.direct_access(i);

    prefetcht0(direction.direct_access());
    
    const double alpha_cg =  res_sqr_norm / denom;
	
    //update solution
    for (uint i=0; i < direction.size(); i++)
      flow_increment_.direct_access(i) += alpha_cg * direction.direct_access(i);
    
    //update residuum
    //     for (uint i=0; i < direction.size(); i++)
    //        residuum.direct_access(i) -= alpha_cg * temp_dir.direct_access(i);
    Makros::array_subtract_multiple(residuum.direct_access(), residuum.size(), 
                                    alpha_cg, temp_dir.direct_access());

    const double new_res_sqr_norm = residuum.sqr_norm();

    prefetcht0(direction.direct_access());

    if ((n%100) == 0) {
      std::cerr << "current norm: " << sqrt(new_res_sqr_norm) << std::endl;
    }

    if (new_res_sqr_norm < 1e-2) {
      std::cerr << "break after " << n << " iterations." << std::endl; 
      break;
    }

    const double beta_cg = new_res_sqr_norm / res_sqr_norm;
    direction *= beta_cg;
    direction += residuum;

    res_sqr_norm = new_res_sqr_norm;

    prefetcht0(dsquare.direct_access());

  }

}


void MotionEstimator::compute_flow_horn_schunck_pcg() {

  flow_.set_constant(0.0);

  Math3D::NamedTensor<double> rhs(xDim_,yDim_,2,MAKENAME(rhs));
  Math3D::NamedTensor<double> dsquare(xDim_,yDim_,2,0.0,MAKENAME(rhs));
  Math2D::NamedMatrix<double> dcross(xDim_,yDim_,0.0,MAKENAME(dcross));

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double sum_u = 0.0;
      double sum_v = 0.0;
      for (uint z=0; z < zDim_; z++) {
        sum_u -= t_derivative_(x,y,z)*gradient_[z](x,y,0);
        sum_v -= t_derivative_(x,y,z)*gradient_[z](x,y,1);      
      }

      rhs(x,y,0) = sum_u;
      rhs(x,y,1) = sum_v;

      for (uint z=0; z < zDim_; z++) {
        double dx = gradient_[z](x,y,0);
        double dy = gradient_[z](x,y,1);
        dsquare(x,y,0) += dx*dx;
        dsquare(x,y,1) += dy*dy;
        dcross(x,y) += dx*dy;
      }
    }
  }

  Math3D::NamedTensor<double> preconditioner(xDim_,yDim_,3,MAKENAME(preconditioner));
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      uint nNeighbors = 0;
      if (x > 0)
        nNeighbors++;
      if ((x+1) < xDim_)
        nNeighbors++;
      if (y > 0)
        nNeighbors++;
      if ((y+1) < yDim_)
        nNeighbors++;

      double sum_u = dsquare(x,y,0) + alpha_*nNeighbors;
      double sum_v = dsquare(x,y,1) + alpha_*nNeighbors;

      Math2D::Matrix<double> M(2,2);
      Math2D::Matrix<double> M_inv(2,2);

      M(0,0) = sum_u;
      M(1,1) = sum_v;
      M(0,1) = dcross(x,y);
      M(1,0) = M(0,1);

      invert_matrix(M,M_inv);
      preconditioner(x,y,0) = M_inv(0,0);
      preconditioner(x,y,1) = M_inv(1,1);
      preconditioner(x,y,2) = M_inv(0,1);

      //       preconditioner(x,y,0) = 1.0 / sum_u;
      //       preconditioner(x,y,1) = 1.0 / sum_v;
    }
  }

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double rx = rhs(x,y,0);
      double ry = rhs(x,y,1);
      rhs(x,y,0) = rx*preconditioner(x,y,0) + ry*preconditioner(x,y,2);
      rhs(x,y,1) = rx*preconditioner(x,y,2) + ry*preconditioner(x,y,1);

      //       rhs(x,y,0) *= preconditioner(x,y,0);
      //       rhs(x,y,1) *= preconditioner(x,y,1);
    }
  }

  Math3D::NamedTensor<double> residuum(xDim_,yDim_,2,MAKENAME(residuum));
  residuum = rhs;

  Math3D::NamedTensor<double> direction(MAKENAME(direction));
  direction = residuum;
  
  Math3D::NamedTensor<double> temp_dir(xDim_,yDim_,2,0.0,MAKENAME(temp_dir));

  double sum_u;
  double sum_v;
  
  double fac[2];
  fac[0] = alpha_;
  fac[1] = alpha_;

  double dcr[2];

  double res_sqr_norm = residuum.sqr_norm();

  for (uint n=1; n <= xDim_*yDim_*2; n++) {

    prefetcht0(direction.direct_access());
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        double u = direction(x,y,0);
        double v = direction(x,y,1);

        sum_u = 0.0;
        sum_v = 0.0;
	
        if (x > 0) {
          sum_u += u - direction(x-1,y,0);
          sum_v += v - direction(x-1,y,1);
        }
        if (x+1 < xDim_) {
          sum_u += u - direction(x+1,y,0);
          sum_v += v - direction(x+1,y,1);
        }
        if (y > 0) {
          sum_u += u - direction(x,y-1,0);
          sum_v += v - direction(x,y-1,1);
        }
        if (y+1 < yDim_) {
          sum_u += u - direction(x,y+1,0);
          sum_v += v - direction(x,y+1,1);
        }

        sum_u *= alpha_;
        sum_v *= alpha_;

        double dc = dcross(x,y);
        sum_u += dsquare(x,y,0)*u + dc*v;
        sum_v += dc*u + dsquare(x,y,1)*v;

        temp_dir(x,y,0) = sum_u;
        temp_dir(x,y,1) = sum_v;
      }
    }


    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {
	
        double tx = temp_dir(x,y,0);
        double ty = temp_dir(x,y,1);
        temp_dir(x,y,0) = tx*preconditioner(x,y,0) + ty*preconditioner(x,y,2);
        temp_dir(x,y,1) = tx*preconditioner(x,y,2) + ty*preconditioner(x,y,1);
      }
    }

    double denom = 0.0;
    for (uint i=0; i < direction.size(); i++)
      denom += temp_dir.direct_access(i) * direction.direct_access(i);

    prefetcht0(direction.direct_access());
    
    const double alpha_cg =  res_sqr_norm / denom;
	
    //update solution
    for (uint i=0; i < direction.size(); i++)
      flow_.direct_access(i) += alpha_cg * direction.direct_access(i);
    
    //update residuum
    //     for (uint i=0; i < direction.size(); i++)
    //        residuum.direct_access(i) -= alpha_cg * temp_dir.direct_access(i);
    Makros::array_subtract_multiple(residuum.direct_access(), residuum.size(), 
                                    alpha_cg, temp_dir.direct_access());

    const double new_res_sqr_norm = residuum.sqr_norm();

    prefetcht0(direction.direct_access());

    if ((n%100) == 0) {
      std::cerr << "current norm: " << sqrt(new_res_sqr_norm) << std::endl;
    }

    if (new_res_sqr_norm < 1e-9) {
      std::cerr << "break after " << n << " preconditioned iterations." << std::endl; 
      break;
    }

    const double beta_cg = new_res_sqr_norm / res_sqr_norm;
    direction *= beta_cg;
    direction += residuum;

    res_sqr_norm = new_res_sqr_norm;

    prefetcht0(dsquare.direct_access());
  }

}


void MotionEstimator::compute_linflow_gradient(Math3D::Tensor<double>& gradient) const {

  assert(linearized_functional_);

  gradient.set_constant(0.0);
  
  //compute gradient
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
       
      const double u = flow_(x,y,0);
      const double v = flow_(x,y,1);
      
      for (uint z=0; z < zDim_; z++) {
	const double dx = gradient_[z](x,y,0);
	const double dy = gradient_[z](x,y,1);
	const double dt = t_derivative_(x,y,z);
	
	const double diff = dx*u + dy*v + dt;

	if (data_norm_ == SquaredDiffs) {
	  gradient(x,y,0) += 2.0*dx*diff;
	  gradient(x,y,1) += 2.0*dy*diff; 
	}
	else {
#ifdef LIPSCHITZ_SMOOTHING
          double norm = fabs(diff);
          if (norm > epsilon_) {
	    //sum += norm - 0.5*epsilon_;
	    gradient(x,y,0) += dx * ( (diff < 0.0) ? -1.0 : 1.0  );
	    gradient(x,y,1) += dy * ( (diff < 0.0) ? -1.0 : 1.0  );
	  }
          else {
            //sum += norm*norm / (2.0*epsilon_);
	    gradient(x,y,0) += dx * diff / epsilon_;
	    gradient(x,y,1) += dy * diff / epsilon_;
	  }
#else
	  const double norm = sqrt(diff*diff+epsilon_);
	  gradient(x,y,0) += dx*diff / norm;
	  gradient(x,y,1) += dy*diff / norm;
#endif
	}
      }

      if (reg_norm_ == L2) {
      
	if (x > 0) {
	  const double diff_u = (u - flow_(x-1,y,0));
	  const double diff_v = (v - flow_(x-1,y,1));
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v; 
	}	
	if (x+1 < xDim_) {
	  const double diff_u = (u - flow_(x+1,y,0));
	  const double diff_v = (v - flow_(x+1,y,1));
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v;
	}
	if (y > 0) {
	  const double diff_u = (u - flow_(x,y-1,0));
	  const double diff_v = (v - flow_(x,y-1,1));
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v;
	}
	if (y+1 < yDim_) {
	  const double diff_u = (u - flow_(x,y+1,0));
	  const double diff_v = (v - flow_(x,y+1,1));
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v;
	}
      }
    }
  }

  if (reg_norm_ == L1 || reg_norm_ == L0_5) {

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {

	  const double hdiff = (x > 0) ? flow_(x,y,i) - flow_(x-1,y,i) : 0.0;
	  const double vdiff = (y > 0) ? flow_(x,y,i) - flow_(x,y-1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    const double norm = sqrt(hdiff*hdiff + vdiff*vdiff);
	    if (norm > epsilon_) {
	      //smooth_energy += norm - 0.5*epsilon_;
	      gradient(x,y,i) += alpha_ * (hdiff+vdiff) / norm;
	      if (x > 0)
		gradient(x-1,y,i) -= alpha_ * hdiff / norm;
	      if (y > 0)
		gradient(x,y-1,i) -= alpha_ * vdiff / norm;
	    }
	    else {
	      //smooth_energy += norm*norm / (2.0*epsilon_);
	      gradient(x,y,i) += alpha_ * (hdiff+vdiff) / epsilon_;
	      if (x > 0)
		gradient(x-1,y,i) -= alpha_ * hdiff / epsilon_;
	      if (y > 0)
		gradient(x,y-1,i) -= alpha_ * vdiff / epsilon_;
	    }
#else
	    const double norm = sqrt(hdiff*hdiff + vdiff*vdiff + epsilon_);
	    
	    gradient(x,y,i) += alpha_ * (hdiff+vdiff) / norm;
	    if (x > 0)
	      gradient(x-1,y,i) -= alpha_ * hdiff / norm;
	    if (y > 0)
	      gradient(x,y-1,i) -= alpha_ * vdiff / norm;
#endif
	  }
	  else {
	    //L0.5
	    const double temp = 0.5 * alpha_ * std::pow(hdiff*hdiff + vdiff*vdiff + epsilon_,-0.75); //a factor of 0.5 cancels with the inner differentiation
	    gradient(x,y,i) += temp * (hdiff+vdiff);
	    if (x > 0)
	      gradient(x-1,y,i) -= temp * hdiff;
	    if (y > 0)
	      gradient(x,y-1,i) -= temp * vdiff;
	  }
	}
      }
    }
  }

}


void MotionEstimator::compute_incrementflow_gradient(Math3D::Tensor<double>& gradient) const {

  assert(!linearized_functional_);
  gradient.set_constant(0.0);


  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      double u = flow_increment_(x,y,0);
      double v = flow_increment_(x,y,1);
      
      for (uint z=0; z < zDim_; z++) {

	double temp = gradient_[z](x,y,0)*u + gradient_[z](x,y,1)*v + t_derivative_(x,y,z);

	if (data_norm_ == SquaredDiffs) {
	  gradient(x,y,0) += 2.0*gradient_[z](x,y,0)*temp;
	  gradient(x,y,1) += 2.0*gradient_[z](x,y,1)*temp;
	}
	else {
#ifdef LIPSCHITZ_SMOOTHING
          double norm = fabs(temp);
          if (norm > epsilon_) {
	    //sum += norm - 0.5*epsilon_;
	    gradient(x,y,0) += alpha_ * gradient_[z](x,y,0) * ( (temp < 0.0) ? -1.0 : 1.0  );
	    gradient(x,y,1) += alpha_ * gradient_[z](x,y,1) * ( (temp < 0.0) ? -1.0 : 1.0  );
	  }
          else {
            //sum += norm*norm / (2.0*epsilon_);
	    gradient(x,y,0) += alpha_ * gradient_[z](x,y,0) * temp / epsilon_;
	    gradient(x,y,1) += alpha_ * gradient_[z](x,y,1) * temp / epsilon_;
	  }
#else
	  const double norm = sqrt(temp*temp+epsilon_);
	  gradient(x,y,0) += gradient_[z](x,y,0) * temp / norm;
	  gradient(x,y,1) += gradient_[z](x,y,1) * temp / norm;
#endif
	}
      }

      if (reg_norm_ == L2) {
	if (x > 0) {
	  const double diff_u = flow_(x,y,0) + u - flow_(x-1,y,0) - flow_increment_(x-1,y,0);
	  const double diff_v = flow_(x,y,1) + v - flow_(x-1,y,1) - flow_increment_(x-1,y,1);
	
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v;
	}
	if (y > 0) {  
	  const double diff_u = flow_(x,y,0) + u - flow_(x,y-1,0) - flow_increment_(x,y-1,0);
	  const double diff_v = flow_(x,y,1) + v - flow_(x,y-1,1) - flow_increment_(x,y-1,1);
	  
	  gradient(x,y,0) += 2.0 * alpha_ * diff_u;
	  gradient(x,y,1) += 2.0 * alpha_ * diff_v;
	}
      }
    }
  }

  if (reg_norm_ == L1 || reg_norm_ == L0_5) {
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

	for (uint i=0; i < 2; i++) {
	  double hdiff = (x > 0) ? flow_(x,y,i) + flow_increment_(x,y,i) - flow_(x-1,y,i) - flow_increment_(x-1,y,i) : 0.0;
	  double vdiff = (y > 0) ? flow_(x,y,i) + flow_increment_(x,y,i) - flow_(x,y-1,i) - flow_increment_(x,y-1,i) : 0.0;
	  
	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    const double norm = sqrt(hdiff*hdiff + vdiff*vdiff);
	    if (norm > epsilon_) {
	      //smooth_energy += norm - 0.5*epsilon_;
	      gradient(x,y,i) += alpha_ * (hdiff+vdiff) / norm;
	      if (x > 0)
		gradient(x-1,y,i) -= alpha_ * hdiff / norm;
	      if (y > 0)
		gradient(x,y-1,i) -= alpha_ * vdiff / norm;
	    }
	    else {
	      //smooth_energy += norm*norm / (2.0*epsilon_);
	      gradient(x,y,i) += alpha_ * (hdiff+vdiff) / epsilon_;
	      if (x > 0)
		gradient(x-1,y,i) -= alpha_ * hdiff / epsilon_;
	      if (y > 0)
		gradient(x,y-1,i) -= alpha_ * vdiff / epsilon_;
	    }
#else
	    const double norm = sqrt(hdiff*hdiff+vdiff*vdiff+epsilon_);
	  
	    gradient(x,y,i) += alpha_ * (hdiff+vdiff) / norm; 
	    if (x > 0)
	      gradient(x-1,y,i) -= alpha_ * hdiff / norm; 
	    if (y > 0)
	      gradient(x,y-1,i) -= alpha_ * vdiff / norm;
#endif	  
	  }
	  else {
	    //L0.5
	    const double temp = 0.5 * alpha_ * std::pow(hdiff*hdiff+vdiff*vdiff+epsilon_,-0.75);
	    gradient(x,y,i) += temp * (hdiff+vdiff); 
	    if (x > 0)
	      gradient(x-1,y,i) -= temp * hdiff; 
	    if (y > 0)
	      gradient(x,y-1,i) -= temp * vdiff;
	  }
	}
      }
    }
  }
}


void MotionEstimator::compute_linflow_gradient_descent() {

  assert(linearized_functional_);

  flow_.set_constant(0.0);

  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));

  double step_size = 4.0e-4;  

  double cur_energy = energy();

  for (uint iter=1; iter < 1000; iter++) {

    //std::cerr << "iter: " << iter << ", energy: " << cur_energy() << std::endl;
    
    compute_linflow_gradient(gradient);

    if ( (iter % 50) == 0) {
      std::cerr << "gradient norm: " << gradient.norm() << std::endl;
      std::cerr << "energy: " << energy() << std::endl;
    }

    //go in the direction of the negative gradient
    gradient *= (-step_size);
    flow_ += gradient;

    double new_energy = energy();
    if (new_energy > cur_energy)
      step_size *= 0.1;

    cur_energy = new_energy;
  }

}


void MotionEstimator::compute_incrementflow_gradient_descent() {

  assert(!linearized_functional_);

  flow_increment_.set_constant(0.0);

  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));

  double step_size = 4.0e-4;  

  double cur_energy = addon_energy(flow_increment_);

  for (uint iter=1; iter < 1000; iter++) {

    if (scale_ == 1.0)
      std::cerr << "iter: " << iter << ", energy: " << addon_energy(flow_increment_) << std::endl;
    
    compute_incrementflow_gradient(gradient);

    if ( (iter % 50) == 0) {
      std::cerr << "gradient norm: " << gradient.norm() << std::endl;
      std::cerr << "energy: " << addon_energy(flow_increment_) << std::endl;
      std::cerr << "step size: " << step_size << std::endl;
    }

    //go in the direction of the negative gradient
    gradient *= (-step_size);
    flow_increment_ += gradient;

    double new_energy = addon_energy(flow_increment_);
    if (new_energy > cur_energy)
      step_size *= 0.1;

    cur_energy = new_energy;
  }    
}


void MotionEstimator::compute_linflow_lbfgs(int L) {

  assert(linearized_functional_);

  flow_.set_constant(0.0);

  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));

  Storage1D<Math3D::Tensor<double> > grad_diff(L);
  Storage1D<Math3D::Tensor<double> > step(L);

  Math1D::Vector<double> rho(L);

  for (int l=0; l < L; l++) {

    grad_diff[l].resize(xDim_,yDim_,2);
    step[l].resize(xDim_,yDim_,2);
  } 

  Math3D::Tensor<double> search_direction(xDim_,yDim_,2);

  double cur_energy = energy();

  for (uint iter=1; iter < 2000; iter++) {

    if (scale_ == 1.0)
      std::cerr << "iter: " << iter << ", energy: " << energy() << std::endl;
    
    compute_linflow_gradient(gradient);

    double sqr_grad_norm = gradient.sqr_norm();
    std::cerr << "sqr grad norm: " << sqr_grad_norm << std::endl;
    if (sqr_grad_norm < 0.01)
      break; //problem solved

    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter-1) % L;
      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];
      const Math3D::Tensor<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k=0; k < flow_.size(); k++) {
	
	//cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
	cur_grad_diff.direct_access(k) += gradient.direct_access(k);
	cur_rho += cur_grad_diff.direct_access(k) * cur_step.direct_access(k);
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      assert(cur_curv > 0.0); //since the function is strictly convex (otherwise we would need to enforce Wolfe part 2)

      rho[cur_l] = 1.0 / cur_rho;
    }

    //a) determine the search direction via L-BFGS
    search_direction = gradient;
    
    if (iter > 1) {

      Math1D::Vector<double> alpha(L);
      
      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter-1; prev_iter >= std::max<int>(1,iter-L); prev_iter--) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double cur_alpha = 0.0; 
	for (uint k=0; k < search_direction.size(); k++) {
	  cur_alpha += search_direction.direct_access(k) * cur_step.direct_access(k);
	}
	cur_alpha *= rho[prev_l];
	alpha[prev_l] = cur_alpha;
	
	for (uint k=0; k < search_direction.size(); k++) {
	  search_direction.direct_access(k) -= cur_alpha * cur_grad_diff.direct_access(k);
	}
      }
      
      //we use a scaled identity as base matrix (q=r=search_direction)
      // uint last_l = (iter-1) % L;

      // const Math3D::Tensor<double>& cur_step = step[last_l];
      // const Math3D::Tensor<double>& cur_grad_diff = grad_diff[last_l];

      // double scale = 0.0;
      // for (uint k=0; k < search_direction.size(); k++) 
      // 	scale += cur_step.direct_access(k) * cur_grad_diff.direct_access(k);
      // scale /= cur_grad_diff.sqr_norm();

      // if (scale <= 0.0) {
      // 	TODO("handle negative scale estimates - should not happen as our objective is strongly convex");
      // }

      // search_direction *= scale;

      search_direction *= cur_curv;
      
      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = std::max<int>(1,iter-L); prev_iter < iter; prev_iter++) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double beta = 0.0; 
	for (uint k=0; k < search_direction.size(); k++) {
	  beta += search_direction.direct_access(k) * cur_grad_diff.direct_access(k);
	}
	beta *= rho[prev_l];

	const double gamma = alpha[prev_l] - beta;
	
	for (uint k=0; k < search_direction.size(); k++) {
	  search_direction.direct_access(k) += cur_step.direct_access(k) * gamma;
	}
      }
      
    }
    negate(search_direction);

    //b) line search along the current search direction
    double best_alpha = 1.0; 
    Math3D::Tensor<double> hyp_flow(xDim_,yDim_,2);
    for (uint k=0; k < hyp_flow.size(); k++)
      hyp_flow.direct_access(k) = flow_.direct_access(k) + best_alpha * search_direction.direct_access(k);
    double best_energy = hyp_energy(hyp_flow);

    double alpha = best_alpha;

    //std::cerr << "alpha: " << alpha << ", hyp energy: " << best_energy << std::endl;

    double cutoff_offset = 0.0;
    for (uint k=0; k < flow_.size(); k++)
      cutoff_offset += gradient.direct_access(k) * search_direction.direct_access(k);
    cutoff_offset *= 0.05; //0.001;

    if (cutoff_offset >= 0.0) {
      INTERNAL_ERROR << "not a descent direction" << std::endl;
    }


    bool wolfe_satisfied = false;

    while (alpha > 1e-30) {

      alpha *= 0.75;
      for (uint k=0; k < hyp_flow.size(); k++)
	hyp_flow.direct_access(k) = flow_.direct_access(k) + alpha * search_direction.direct_access(k);
      double cur_hyp_energy = hyp_energy(hyp_flow);

      // std::cerr << "alpha: " << alpha << ", hyp energy: " << hyp_energy << ", threshold: " 
      // 		<< (cur_energy + alpha*cutoff_offset) << ", satisfied: " 
      // 		<< (hyp_energy <= cur_energy + alpha*cutoff_offset) << std::endl;

      //Wolfe conditions, part 1:
      if (cur_hyp_energy <= cur_energy + alpha*cutoff_offset) {
      	wolfe_satisfied = true; 

	//NOTE: because the function is strongly convex, 
	// the curvature condition must be satisfied, so we need not check part 2 of the Wolfe conditions.
	// Also, backtracking line search is not suited to find a point that satisfies part 2. We would need
	//   Algorithm 3.5 from [Nocedal & Wright]

      	if (cur_hyp_energy < best_energy) {
      	  best_energy = cur_hyp_energy;
      	  best_alpha = alpha;
      	}
      	else 
      	  break;
      }
      else if (wolfe_satisfied) 
       	break;
    }


    //c) update denoised_image and the step vectors
    if (best_energy <= cur_energy) {

      uint cur_l = (iter % L);

      Math3D::Tensor<double>& cur_step = step[cur_l];
      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];

      for (uint k=0; k < flow_.size(); k++) {
	double step = best_alpha * search_direction.direct_access(k);
	cur_step.direct_access(k) = step;
	flow_.direct_access(k) += step;

	//prepare for the next iteration
	cur_grad_diff.direct_access(k) = -gradient.direct_access(k);
      }

      cur_energy = best_energy;
    }
    else {
      INTERNAL_ERROR << " failed to get descent, sqr grad norm: " << sqr_grad_norm << std::endl;
      exit(1);
    }

  }

}

void MotionEstimator::compute_incrementflow_lbfgs(int L) {

  assert(!linearized_functional_);

  flow_increment_.set_constant(0.0);

  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));

  Storage1D<Math3D::Tensor<double> > grad_diff(L);
  Storage1D<Math3D::Tensor<double> > step(L);

  Math1D::Vector<double> rho(L);

  for (int l=0; l < L; l++) {

    grad_diff[l].resize(xDim_,yDim_,2);
    step[l].resize(xDim_,yDim_,2);
  } 

  Math3D::Tensor<double> search_direction(xDim_,yDim_,2);

  double cur_energy = addon_energy(flow_increment_);

  int start_iter = 1;

  for (uint iter=1; iter < 1000; iter++) {

    if (scale_ == 1.0)
      std::cerr << "iter: " << iter << ", energy: " << cur_energy << std::endl;
    assert(cur_energy == addon_energy(flow_increment_));

    compute_incrementflow_gradient(gradient);

    double sqr_grad_norm = gradient.sqr_norm();
    if (scale_ == 1.0)
      std::cerr << "sqr grad norm: " << sqr_grad_norm << std::endl;
    if (sqr_grad_norm < 0.01)
      break; //problem solved

    double cur_curv = 0.0;

    if (iter > start_iter) {
      //update grad_diff and rho
      uint cur_l = (iter-1) % L;
      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];
      const Math3D::Tensor<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k=0; k < flow_.size(); k++) {
	
	//cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
	cur_grad_diff.direct_access(k) += gradient.direct_access(k);
	cur_rho += cur_grad_diff.direct_access(k) * cur_step.direct_access(k);
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      assert(cur_curv > 0.0); //since the function is strictly convex (otherwise we would need to enforce Wolfe part 2)

      rho[cur_l] = 1.0 / cur_rho;
    }

    //a) determine the search direction via L-BFGS
    search_direction = gradient;
    
    if (iter > start_iter) {

      Math1D::Vector<double> alpha(L);
      
      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter-1; prev_iter >= std::max<int>(start_iter,iter-L); prev_iter--) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double cur_alpha = 0.0; 
	for (uint k=0; k < search_direction.size(); k++) {
	  cur_alpha += search_direction.direct_access(k) * cur_step.direct_access(k);
	}
	cur_alpha *= rho[prev_l];
	alpha[prev_l] = cur_alpha;
	
	for (uint k=0; k < search_direction.size(); k++) {
	  search_direction.direct_access(k) -= cur_alpha * cur_grad_diff.direct_access(k);
	}
      }
      
      //we use a scaled identity as base matrix (q=r=search_direction)
      // uint last_l = (iter-1) % L;

      // const Math3D::Tensor<double>& cur_step = step[last_l];
      // const Math3D::Tensor<double>& cur_grad_diff = grad_diff[last_l];

      // double scale = 0.0;
      // for (uint k=0; k < search_direction.size(); k++) 
      // 	scale += cur_step.direct_access(k) * cur_grad_diff.direct_access(k);
      // scale /= cur_grad_diff.sqr_norm();

      // if (scale <= 0.0) {
      // 	TODO("handle negative scale estimates - should not happen as our objective is strongly convex");
      // }

      // search_direction *= scale;

      search_direction *= cur_curv;
      
      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = std::max<int>(1,iter-L); prev_iter < iter; prev_iter++) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double beta = 0.0; 
	for (uint k=0; k < search_direction.size(); k++) {
	  beta += search_direction.direct_access(k) * cur_grad_diff.direct_access(k);
	}
	beta *= rho[prev_l];

	const double gamma = alpha[prev_l] - beta;
	
	for (uint k=0; k < search_direction.size(); k++) {
	  search_direction.direct_access(k) += cur_step.direct_access(k) * gamma;
	}
      }
      
    }
    negate(search_direction);

    //b) line search along the current search direction
    double best_alpha = 1.0; 
    Math3D::Tensor<double> hyp_flow_increment(xDim_,yDim_,2);
    for (uint k=0; k < hyp_flow_increment.size(); k++)
      hyp_flow_increment.direct_access(k) = flow_increment_.direct_access(k) + best_alpha * search_direction.direct_access(k);
    double best_energy = addon_energy(hyp_flow_increment);

    double alpha = best_alpha;

    //std::cerr << "alpha: " << alpha << ", hyp energy: " << best_energy << std::endl;

    double cutoff_offset = 0.0;
    for (uint k=0; k < flow_increment_.size(); k++)
      cutoff_offset += gradient.direct_access(k) * search_direction.direct_access(k);
    cutoff_offset *= 0.05; //0.001;

    if (cutoff_offset >= 0.0) {
      INTERNAL_ERROR << "not a descent direction" << std::endl;
    }


    bool wolfe_satisfied = false;

    while (alpha > 1e-20) {

      alpha *= 0.75;
      for (uint k=0; k < hyp_flow_increment.size(); k++)
	hyp_flow_increment.direct_access(k) = flow_increment_.direct_access(k) + alpha * search_direction.direct_access(k);
      double cur_hyp_energy = addon_energy(hyp_flow_increment);

      // std::cerr << "alpha: " << alpha << ", hyp energy: " << cur_hyp_energy << ", threshold: " 
      // 		<< (cur_energy + alpha*cutoff_offset) << ", satisfied: " 
      // 		<< (cur_hyp_energy <= cur_energy + alpha*cutoff_offset) << std::endl;

      //Wolfe conditions, part 1:
      if (cur_hyp_energy <= cur_energy + alpha*cutoff_offset) {
      	wolfe_satisfied = true; 

	//NOTE: because the function is strongly convex, 
	// the curvature condition must be satisfied, so we need not check part 2 of the Wolfe conditions.
	// Also, backtracking line search is not suited to find a point that satisfies part 2. We would need
	//   Algorithm 3.5 from [Nocedal & Wright]

      	if (cur_hyp_energy < best_energy) {
      	  best_energy = cur_hyp_energy;
      	  best_alpha = alpha;
      	}
      	else 
      	  break;
      }
      else if (wolfe_satisfied) 
       	break;
    }
    

    //if (best_alpha < 1e-6)
    //  start_iter = iter;

    //std::cerr << "best_alpha: " << best_alpha << std::endl;
    
    //c) update denoised_image and the step vectors
    if (best_energy <= cur_energy) {

      uint cur_l = (iter % L);

      Math3D::Tensor<double>& cur_step = step[cur_l];
      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];

      for (uint k=0; k < flow_increment_.size(); k++) {
	double step = best_alpha * search_direction.direct_access(k);
	cur_step.direct_access(k) = step;
	flow_increment_.direct_access(k) += step;

	//prepare for the next iteration
	cur_grad_diff.direct_access(k) = -gradient.direct_access(k);
      }

      cur_energy = best_energy;
    }
    else {
      INTERNAL_ERROR << " failed to get descent, sqr grad norm: " << sqr_grad_norm << ", best energy: " << best_energy << std::endl;
      exit(1);
    }

  }

}


void MotionEstimator::compute_flow_horn_schunck_nesterov() {
  
  flow_.set_constant(0.0);
  
  Math3D::Tensor<double> aux_flow = flow_;
  
  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));
  
  double prev_t = 1.0;

  for (uint iter = 1; iter <= 1000; iter++) {

    if (scale_ == 1.0)
      std::cerr << "current energy: " << energy() << std::endl;

    //1.) compute gradient
    gradient.set_constant(0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        const double u = aux_flow(x,y,0);
        const double v = aux_flow(x,y,1);
       
        for (uint z=0; z < zDim_; z++) {
          const double dx = gradient_[z](x,y,0);
          const double dy = gradient_[z](x,y,1);
          const double dt = t_derivative_(x,y,z);
	 
          gradient(x,y,0) += dx*(dx*u + dy*v + dt);
          gradient(x,y,1) += dy*(dx*u + dy*v + dt);
        }
       
        if (x > 0) {
          gradient(x,y,0) += alpha_ * (u - aux_flow(x-1,y,0));
          gradient(x,y,1) += alpha_ * (v - aux_flow(x-1,y,1));
        }	
        if (x+1 < xDim_) {
          gradient(x,y,0) += alpha_ * (u - aux_flow(x+1,y,0));
          gradient(x,y,1) += alpha_ * (v - aux_flow(x+1,y,1));
        }
        if (y > 0) {
          gradient(x,y,0) += alpha_ * (u - aux_flow(x,y-1,0));
          gradient(x,y,1) += alpha_ * (v - aux_flow(x,y-1,1));
        }
        if (y+1 < yDim_) {
          gradient(x,y,0) += alpha_ * (u - aux_flow(x,y+1,0));
          gradient(x,y,1) += alpha_ * (v - aux_flow(x,y+1,1));
        }
      }
    }

    double stepsize = 2e-4;

    for (uint i=0; i < flow_.size(); i++)
      aux_flow.direct_access(i) -= stepsize * gradient.direct_access(i);

    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;
    //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
    
    for (uint i=0; i < aux_flow.size(); i++) {
      
      const double old_aux = aux_flow.direct_access(i);
      aux_flow.direct_access(i) = old_aux + nesterov_fac*(old_aux - flow_.direct_access(i)) ;
      flow_.direct_access(i) = old_aux;
    }
    
    prev_t = new_t; 
  }
}


void MotionEstimator::compute_flow_increment_horn_schunck_nesterov() {

  flow_increment_.set_constant(0.0);

  Math3D::Tensor<double> aux_flow_increment = flow_increment_;
  
  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));
  
  double prev_t = 1.0;
  
  for (uint iter = 1; iter <= 1000; iter++) {

    //1.) compute gradient
    gradient.set_constant(0.0);
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {
	
        const double u = aux_flow_increment(x,y,0);
        const double v = aux_flow_increment(x,y,1);
	
        for (uint z=0; z < zDim_; z++) {
          const double dx = gradient_[z](x,y,0);
          const double dy = gradient_[z](x,y,1);
          const double dt = t_derivative_(x,y,z);
	  
          gradient(x,y,0) += dx*(dx*u + dy*v + dt);
          gradient(x,y,1) += dy*(dx*u + dy*v + dt);
        }
	
        if (x > 0) {
          gradient(x,y,0) += alpha_ * (u + flow_(x,y,0) - aux_flow_increment(x-1,y,0) - flow_(x-1,y,0));
          gradient(x,y,1) += alpha_ * (v + flow_(x,y,1) - aux_flow_increment(x-1,y,1) - flow_(x-1,y,1));
        }	
        if (x+1 < xDim_) {
          gradient(x,y,0) += alpha_ * (u + flow_(x,y,0) - aux_flow_increment(x+1,y,0) - flow_(x+1,y,0));
          gradient(x,y,1) += alpha_ * (v + flow_(x,y,1) - aux_flow_increment(x+1,y,1) - flow_(x+1,y,1));
        }
        if (y > 0) {
          gradient(x,y,0) += alpha_ * (u + flow_(x,y,0) - aux_flow_increment(x,y-1,0) - flow_(x,y-1,0));
          gradient(x,y,1) += alpha_ * (v + flow_(x,y,1) - aux_flow_increment(x,y-1,1) - flow_(x,y-1,1));
        }
        if (y+1 < yDim_) {
          gradient(x,y,0) += alpha_ * (u + flow_(x,y,0) - aux_flow_increment(x,y+1,0) - flow_(x,y+1,0));
          gradient(x,y,1) += alpha_ * (v + flow_(x,y,1) - aux_flow_increment(x,y+1,1) - flow_(x,y+1,1));
        }
      }
    }

    double stepsize = 2e-4;

    for (uint i=0; i < flow_increment_.size(); i++)
      aux_flow_increment.direct_access(i) -= stepsize * gradient.direct_access(i);

    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;
    //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
    
    for (uint i=0; i < aux_flow_increment.size(); i++) {
      
      const double old_aux = aux_flow_increment.direct_access(i);
      aux_flow_increment.direct_access(i) = old_aux + nesterov_fac*(old_aux - flow_increment_.direct_access(i)) ;
      flow_increment_.direct_access(i) = old_aux;
    }
    
    prev_t = new_t; 
  }
}


void MotionEstimator::compute_flow_tv_l2_sor() {

  std::cerr << "tv-l2 (iterated sor)" << std::endl;

  flow_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint outer_iter = 1; outer_iter <= 400; outer_iter++) {

    Math3D::Tensor<double> hdiff(xDim_-1,yDim_,2,0.0);
    Math3D::Tensor<double> vdiff(xDim_,yDim_-1,2,0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_-1; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x+1,y,i) - flow_(x,y,i);
          double add = (y > 0) ? flow_(x+1,y,i) - flow_(x+1,y-1,i) : 0.0;

#ifdef LIPSCHITZ_SMOOTHING
          double norm = sqrt(temp*temp + add*add);
          if (norm >= epsilon_)
            hdiff(x,y,i) = (0.5 * alpha_) / norm;
          else
            hdiff(x,y,i) = 0.5 * alpha_ / epsilon_;
#else
          //the factor 0.5 is used to cancel out the 2.0 neglected in the data term
          hdiff (x,y,i) = (0.5 * alpha_) / sqrt(temp*temp + add*add + epsilon_);
#endif
        }
      }
    }

    for (uint y=0; y < yDim_-1; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x,y+1,i) - flow_(x,y,i);
          double add = (x > 0) ? flow_(x,y+1,i) - flow_(x-1,y+1,i) : 0.0;

#ifdef LIPSCHITZ_SMOOTHING
          double norm = sqrt(temp*temp + add*add);
          if (norm >= epsilon_)
            vdiff(x,y,i) = (0.5 * alpha_) / norm;
          else
            vdiff(x,y,i) = 0.5 * alpha_ / epsilon_;
#else
          //the factor 0.5 is used to cancel out the 2.0 neglected in the data term
          vdiff(x,y,i) = (0.5 * alpha_) / sqrt(temp*temp + add*add + epsilon_);
#endif
        }
      }
    }

    for (uint inner_iter = 1; inner_iter <= 5; inner_iter++) {

      for (uint z=0; z < zDim_; z++) {
        for (uint y=0; y < yDim_; y++) {
          for (uint x=0; x < xDim_; x++) {
	    
            //update u
            double sum_u = 0.0;
            double denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	    
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_u -= dx*(dy*flow_(x,y,1) + dt);
              denom += dx*dx;
            }
	  
            if (x > 0) {
              sum_u += hdiff(x-1,y,0)*flow_(x-1,y,0);
              denom += hdiff(x-1,y,0);
            }
            if (x+1 < xDim_) {
              sum_u += hdiff(x,y,0)*flow_(x+1,y,0);
              denom += hdiff(x,y,0);
            }
            if (y > 0) {
              sum_u += vdiff(x,y-1,0)*flow_(x,y-1,0);
              denom += vdiff(x,y-1,0);
            }
            if (y+1 < yDim_) {
              sum_u += vdiff(x,y,0)*flow_(x,y+1,0);
              denom += vdiff(x,y,0);
            }
	    
            sum_u /= denom;
            flow_(x,y,0) = neg_omega * flow_(x,y,0) + omega *sum_u;

            //update v
            double sum_v = 0.0;
            denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	      
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_v -= dy*(dx*flow_(x,y,0)  + dt);
              denom += dy*dy;
            }
	    
            if (x > 0) {
              sum_v += hdiff(x-1,y,1)*flow_(x-1,y,1);
              denom += hdiff(x-1,y,1);
            }
            if (x+1 < xDim_) {
              sum_v += hdiff(x,y,1)*flow_(x+1,y,1);
              denom += hdiff(x,y,1);
            }
            if (y > 0) {
              sum_v += vdiff(x,y-1,1)*flow_(x,y-1,1);
              denom += vdiff(x,y-1,1);
            }
            if (y+1 < yDim_) {
              sum_v += vdiff(x,y,1)*flow_(x,y+1,1);
              denom += vdiff(x,y,1);
            }

            sum_v /= denom;
            flow_(x,y,1) = neg_omega * flow_(x,y,1) + omega * sum_v;
          }
        }
      }
    }

    if (scale_ == 1.0)
      std::cerr << "energy: " << energy() << std::endl;
  }
}

void MotionEstimator::compute_flow_tv_l1_sor() {

  if (reg_norm_ == L1)
    std::cerr << "tv-l1 (iterated sor)" << std::endl;
  else
    std::cerr << "sqrt(tv)-l1 (iterated sor)" << std::endl;

  flow_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint outer_iter = 1; outer_iter <= 200; outer_iter++) {

    Math3D::Tensor<double> hdiff(xDim_-1,yDim_,2,0.0);
    Math3D::Tensor<double> vdiff(xDim_,yDim_-1,2,0.0);

    Math3D::Tensor<double> data_diff(xDim_,yDim_,zDim_,0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_-1; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x+1,y,i) - flow_(x,y,i);
          double add = (y > 0) ? flow_(x+1,y,i) - flow_(x+1,y-1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = sqrt(temp*temp + add*add);
	    if (norm >= epsilon_)
	      hdiff(x,y,i) = alpha_ / norm;
	    else
	      hdiff(x,y,i) = alpha_ / epsilon_;
#else
	    hdiff(x,y,i) = alpha_ / sqrt(temp*temp + add*add + epsilon_);
#endif
	  }
	  else
	    hdiff(x,y,i) = 0.5 * alpha_ * std::pow(temp*temp + add*add + epsilon_,-0.75); //factor of 0.5 cancels with the inner differentiation
        }
      }
    }

    for (uint y=0; y < yDim_-1; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x,y+1,i) - flow_(x,y,i);
          double add = (x > 0) ? flow_(x,y+1,i) - flow_(x-1,y+1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = sqrt(temp*temp + add*add);
	    if (norm >= epsilon_)
	      vdiff(x,y,i) = alpha_ / norm;
	    else
	      vdiff(x,y,i) = alpha_ / epsilon_;
#else
	    vdiff(x,y,i) = alpha_ / sqrt(temp*temp + add*add + epsilon_);
#endif
	  }
	  else
	    vdiff(x,y,i) = 0.5 * alpha_ * std::pow(temp*temp + add*add + epsilon_,-0.75);
        }
      }


      for (uint y=0; y < yDim_; y++) {
	for (uint x=0; x < xDim_; x++) {
	  
	  for (uint z=0; z < zDim_; z++) {
	    double dx = gradient_[z](x,y,0);
	    double dy = gradient_[z](x,y,1);
	    double dt = t_derivative_(x,y,z);
	  
	    double temp = dx*flow_(x,y,0) + dy*flow_(x,y,1) + dt;
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = fabs(temp);
	    if (norm >= epsilon_)
	      data_diff(x,y,z) = 1.0 / norm; 
	    else
	      data_diff(x,y,z) = 1.0 / epsilon_;
#else
	    data_diff(x,y,z) = 1.0 / sqrt(temp*temp + epsilon_);
#endif
	  }
	}
      }
    }


    // if (outer_iter == 1) {

    //   //initialize the problems with the absolutes by the solution of the one with the squares
    //   //experimentally, this did not perform well
    //   hdiff.set_constant(1.0);
    //   vdiff.set_constant(1.0);
    //   data_diff.set_constant(1.0);
    // }

    for (uint inner_iter = 1; inner_iter <= 5; inner_iter++) {

      for (uint z=0; z < zDim_; z++) {
        for (uint y=0; y < yDim_; y++) {
          for (uint x=0; x < xDim_; x++) {
	    
	    const double cur_data_diff = data_diff(x,y,z);

            //update u
            double sum_u = 0.0;
            double denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	    
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_u -= cur_data_diff*dx*(dy*flow_(x,y,1) + dt);
              denom += cur_data_diff*dx*dx;
            }
	  
            if (x > 0) {
              sum_u += hdiff(x-1,y,0)*flow_(x-1,y,0);
              denom += hdiff(x-1,y,0);
            }
            if (x+1 < xDim_) {
              sum_u += hdiff(x,y,0)*flow_(x+1,y,0);
              denom += hdiff(x,y,0);
            }
            if (y > 0) {
              sum_u += vdiff(x,y-1,0)*flow_(x,y-1,0);
              denom += vdiff(x,y-1,0);
            }
            if (y+1 < yDim_) {
              sum_u += vdiff(x,y,0)*flow_(x,y+1,0);
              denom += vdiff(x,y,0);
            }
	    
            sum_u /= denom;
            flow_(x,y,0) = neg_omega * flow_(x,y,0) + omega *sum_u;

            //update v
            double sum_v = 0.0;
            denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	      
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_v -= cur_data_diff*dy*(dx*flow_(x,y,0)  + dt);
              denom += cur_data_diff*dy*dy;
            }
	    
            if (x > 0) {
              sum_v += hdiff(x-1,y,1)*flow_(x-1,y,1);
              denom += hdiff(x-1,y,1);
            }
            if (x+1 < xDim_) {
              sum_v += hdiff(x,y,1)*flow_(x+1,y,1);
              denom += hdiff(x,y,1);
            }
            if (y > 0) {
              sum_v += vdiff(x,y-1,1)*flow_(x,y-1,1);
              denom += vdiff(x,y-1,1);
            }
            if (y+1 < yDim_) {
              sum_v += vdiff(x,y,1)*flow_(x,y+1,1);
              denom += vdiff(x,y,1);
            }

            sum_v /= denom;
            flow_(x,y,1) = neg_omega * flow_(x,y,1) + omega * sum_v;
          }
        }
      }
    }

    if (scale_ == 1.0)
      std::cerr << "energy: " << energy() << std::endl;
  }
}

void MotionEstimator::compute_flow_tv_l2_nesterov() {

  std::cerr << "tv-l2 (Nesterov's optimal method)" << std::endl;

  flow_.set_constant(0.0);

#ifdef LIPSCHITZ_SMOOTHING

  Math3D::Tensor<double> aux_flow = flow_;
  
  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));
  
  double prev_t = 1.0;

  for (uint iter = 1; iter <= 1000; iter++) {

    std::cerr << "current energy: " << energy() << std::endl;

    //1.) compute gradient
    gradient.set_constant(0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        const double u = aux_flow(x,y,0);
        const double v = aux_flow(x,y,1);
       
        for (uint z=0; z < zDim_; z++) {
          const double dx = gradient_[z](x,y,0);
          const double dy = gradient_[z](x,y,1);
          const double dt = t_derivative_(x,y,z);
	 
          gradient(x,y,0) += dx*(dx*u + dy*v + dt);
          gradient(x,y,1) += dy*(dx*u + dy*v + dt);
        }

        if (x > 0 || y > 0) {

          for (uint i=0; i < 2; i++) {
            double hdiff = (x > 0) ? aux_flow(x,y,i) - aux_flow(x-1,y,i) : 0.0; 
            double vdiff = (y > 0) ? aux_flow(x,y,i) - aux_flow(x,y-1,i) : 0.0; 

            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);

            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (hdiff+vdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (hdiff+vdiff) / norm;
            }
          }
        }
        if (x + 1 < xDim_) {

          for (uint i=0; i < 2; i++) {
            double hdiff = aux_flow(x,y,i) - aux_flow(x+1,y,i); 
            double vdiff = (y > 0) ? aux_flow(x+1,y,i) - aux_flow(x+1,y-1,i) : 0.0; 
	  
            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);
	    
            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (hdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (hdiff) / norm;
            }
          }
        }
        if (y + 1 < yDim_) {

          for (uint i=0; i < 2; i++) {
            double hdiff = (x > 0) ? aux_flow(x,y+1,i) - aux_flow(x-1,y+1,i) : 0.0;
            double vdiff = aux_flow(x,y,i) - aux_flow(x,y+1,i);
	    
            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);
	    
            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (vdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (vdiff) / norm;
            }
          }
        }
      }
    }

    double stepsize = 0.5e-5;

    for (uint i=0; i < flow_.size(); i++)
      aux_flow.direct_access(i) -= stepsize * gradient.direct_access(i);
    
    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;
    //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
	
    for (uint i=0; i < aux_flow.size(); i++) {
      
      const double old_aux = aux_flow.direct_access(i);
      aux_flow.direct_access(i) = old_aux + nesterov_fac*(old_aux - flow_.direct_access(i)) ;
      flow_.direct_access(i) = old_aux;
    }
    
    prev_t = new_t; 
  }
#else
  INTERNAL_ERROR << " Nesterov's method is not applicable for this kind of smoothing. Exiting." << std::endl;
  exit(1);
#endif
}

void MotionEstimator::compute_flow_increment_tv_l2_sor() {

  std::cerr << "tv-l2 increment (iterated sor)" << std::endl;

  flow_increment_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint outer_iter = 1; outer_iter <= 240; outer_iter++) {

    Math3D::Tensor<double> hdiff(xDim_-1,yDim_,2,0.0);
    Math3D::Tensor<double> vdiff(xDim_,yDim_-1,2,0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_-1; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x+1,y,i) + flow_increment_(x+1,y,i)  - flow_(x,y,i) - flow_increment_(x,y,i);
          double add = (y > 0) ? flow_(x+1,y,i) + flow_increment_(x+1,y,i) - flow_(x+1,y-1,i) - flow_increment_(x+1,y-1,i) : 0.0;

#ifdef LIPSCHITZ_SMOOTHING
          double norm = sqrt(temp*temp + add*add);
          if (norm >= epsilon_)
            hdiff(x,y,i) = (0.5 * alpha_) / norm;
          else
            hdiff(x,y,i) = 0.5*alpha_ / epsilon_;
#else
          //the factor 0.5 is used to cancel out the 2.0 neglected in the data term
          hdiff(x,y,i) = (0.5 * alpha_) / sqrt(temp*temp + add*add + epsilon_);
#endif
        }
      }
    }

    for (uint y=0; y < yDim_-1; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x,y+1,i) + flow_increment_(x,y+1,i) - flow_(x,y,i) - flow_increment_(x,y,i);
          double add = (x > 0) ? flow_(x,y+1,i) + flow_increment_(x,y+1,i) - flow_(x-1,y+1,i) - flow_increment_(x-1,y+1,i) : 0.0;

#ifdef LIPSCHITZ_SMOOTHING
          double norm = sqrt(temp*temp + add*add);
          if (norm >= epsilon_)
            vdiff(x,y,i) = (0.5 * alpha_) / norm;
          else
            vdiff(x,y,i) = 0.5*alpha_ / epsilon_;
#else
          //the factor 0.5 is used to cancel out the 2.0 neglected in the data term
          vdiff(x,y,i) = (0.5 * alpha_) / sqrt(temp*temp + add*add + epsilon_);
#endif
        }
      }
    }


    for (uint inner_iter = 1; inner_iter <= 1; inner_iter++) {

      for (uint z=0; z < zDim_; z++) {
        for (uint y=0; y < yDim_; y++) {
          for (uint x=0; x < xDim_; x++) {
	    
            //update u
            double sum_u = 0.0;
            double denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	    
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_u -= dx*(dy*flow_increment_(x,y,1) + dt);
              denom += dx*dx;
            }
	  
            if (x > 0) {
              sum_u += hdiff(x-1,y,0)*(flow_increment_(x-1,y,0) + flow_(x-1,y,0) - flow_(x,y,0));
              denom += hdiff(x-1,y,0);
            }
            if (x+1 < xDim_) {
              sum_u += hdiff(x,y,0)*(flow_increment_(x+1,y,0) + flow_(x+1,y,0) - flow_(x,y,0));
              denom += hdiff(x,y,0);
            }
            if (y > 0) {
              sum_u += vdiff(x,y-1,0)*(flow_increment_(x,y-1,0) + flow_(x,y-1,0) - flow_(x,y,0));
              denom += vdiff(x,y-1,0);
            }
            if (y+1 < yDim_) {
              sum_u += vdiff(x,y,0)*(flow_increment_(x,y+1,0) + flow_(x,y+1,0) - flow_(x,y,0));
              denom += vdiff(x,y,0);
            }
	    
            sum_u /= denom;
            flow_increment_(x,y,0) = neg_omega * flow_increment_(x,y,0) + omega *sum_u;

            //update v
            double sum_v = 0.0;
            denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	      
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_v -= dy*(dx*flow_increment_(x,y,0)  + dt);
              denom += dy*dy;
            }
	    
            if (x > 0) {
              sum_v += hdiff(x-1,y,1)*(flow_increment_(x-1,y,1) + flow_(x-1,y,1) - flow_(x,y,1));
              denom += hdiff(x-1,y,1);
            }
            if (x+1 < xDim_) {
              sum_v += hdiff(x,y,1)*(flow_increment_(x+1,y,1) + flow_(x+1,y,1) - flow_(x,y,1));
              denom += hdiff(x,y,1);
            }
            if (y > 0) {
              sum_v += vdiff(x,y-1,1)*(flow_increment_(x,y-1,1) + flow_(x,y-1,1) - flow_(x,y,1));
              denom += vdiff(x,y-1,1);
            }
            if (y+1 < yDim_) {
              sum_v += vdiff(x,y,1)*(flow_increment_(x,y+1,1) + flow_(x,y+1,1) - flow_(x,y,1));
              denom += vdiff(x,y,1);
            }

            sum_v /= denom;
            flow_increment_(x,y,1) = neg_omega * flow_increment_(x,y,1) + omega * sum_v;
          }
        }
      }
    }

    if (scale_ == 1.0)
      std::cerr << "addon energy: " << addon_energy(flow_increment_) << std::endl;
  }
}

void MotionEstimator::compute_flow_increment_tv_l1_sor() {

  std::cerr << "tv-l1 increment (iterated sor)" << std::endl;

  flow_increment_.set_constant(0.0);

  const double omega = 1.9;
  const double neg_omega = 1.0f - omega;

  for (uint outer_iter = 1; outer_iter <= 240; outer_iter++) {

    Math3D::Tensor<double> hdiff(xDim_-1,yDim_,2,0.0);
    Math3D::Tensor<double> vdiff(xDim_,yDim_-1,2,0.0);
    Math3D::Tensor<double> data_diff(xDim_,yDim_,2,0.0);


    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_-1; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x+1,y,i) + flow_increment_(x+1,y,i)  - flow_(x,y,i) - flow_increment_(x,y,i);
          double add = (y > 0) ? flow_(x+1,y,i) + flow_increment_(x+1,y,i) - flow_(x+1,y-1,i) - flow_increment_(x+1,y-1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = sqrt(temp*temp + add*add);
	    if (norm >= epsilon_)
	      hdiff(x,y,i) = alpha_ / norm;
	    else
	      hdiff(x,y,i) = alpha_ / epsilon_;
#else
	    hdiff(x,y,i) = alpha_ / sqrt(temp*temp + add*add + epsilon_);
#endif
	  }
	  else
	    hdiff(x,y,i) = 0.5 * alpha_ * std::pow(temp*temp + add*add + epsilon_,-0.75); //factor of 0.5 cancels with the inner differentiation
        }
      }
    }

    for (uint y=0; y < yDim_-1; y++) {
      for (uint x=0; x < xDim_; x++) {

        for (uint i=0; i < 2; i++) {
          double temp = flow_(x,y+1,i) + flow_increment_(x,y+1,i) - flow_(x,y,i) - flow_increment_(x,y,i);
          double add = (x > 0) ? flow_(x,y+1,i) + flow_increment_(x,y+1,i) - flow_(x-1,y+1,i) - flow_increment_(x-1,y+1,i) : 0.0;

	  if (reg_norm_ == L1) {
#ifdef LIPSCHITZ_SMOOTHING
	    double norm = sqrt(temp*temp + add*add);
	    if (norm >= epsilon_)
	      vdiff(x,y,i) = alpha_ / norm;
	    else
	      vdiff(x,y,i) = alpha_ / epsilon_;
#else
	    vdiff(x,y,i) = alpha_ / sqrt(temp*temp + add*add + epsilon_);
#endif
	  }
	  else
	    vdiff(x,y,i) = 0.5 * alpha_ * std::pow(temp*temp + add*add + epsilon_,-0.75);
        }
      }
    }

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {
	for (uint z=0; z < zDim_; z++) {

	  double dx = gradient_[z](x,y,0);
	  double dy = gradient_[z](x,y,1);
	  double dt = t_derivative_(x,y,z);
	  
	  double temp = dx*flow_increment_(x,y,0) + dy*flow_increment_(x,y,1) + dt;
#ifdef LIPSCHITZ_SMOOTHING
	  double norm = fabs(temp);
	  if (norm >= epsilon_)
	    data_diff(x,y,z) = 1.0 / norm; 
	  else
	    data_diff(x,y,z) = 1.0 / epsilon_;
#else
	  data_diff(x,y,z) = 1.0 / sqrt(temp*temp + epsilon_);
#endif
	}
      }
    }


    // if (outer_iter == 1) {

    //   //initialize the problems with the absolutes by the solution of the one with the squares
    //   //experimentally, this did not perform well
    //   hdiff.set_constant(1.0);
    //   vdiff.set_constant(1.0);
    //   data_diff.set_constant(1.0);
    // }


    for (uint inner_iter = 1; inner_iter <= 1; inner_iter++) {

      for (uint z=0; z < zDim_; z++) {
        for (uint y=0; y < yDim_; y++) {
          for (uint x=0; x < xDim_; x++) {
	
	    const double cur_data_diff = data_diff(x,y,z);
    
            //update u
            double sum_u = 0.0;
            double denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	    
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_u -= cur_data_diff*dx*(dy*flow_increment_(x,y,1) + dt);
              denom += cur_data_diff*dx*dx;
            }
	  
            if (x > 0) {
              sum_u += hdiff(x-1,y,0)*(flow_increment_(x-1,y,0) + flow_(x-1,y,0) - flow_(x,y,0));
              denom += hdiff(x-1,y,0);
            }
            if (x+1 < xDim_) {
              sum_u += hdiff(x,y,0)*(flow_increment_(x+1,y,0) + flow_(x+1,y,0) - flow_(x,y,0));
              denom += hdiff(x,y,0);
            }
            if (y > 0) {
              sum_u += vdiff(x,y-1,0)*(flow_increment_(x,y-1,0) + flow_(x,y-1,0) - flow_(x,y,0));
              denom += vdiff(x,y-1,0);
            }
            if (y+1 < yDim_) {
              sum_u += vdiff(x,y,0)*(flow_increment_(x,y+1,0) + flow_(x,y+1,0) - flow_(x,y,0));
              denom += vdiff(x,y,0);
            }
	    
            sum_u /= denom;
            flow_increment_(x,y,0) = neg_omega * flow_increment_(x,y,0) + omega *sum_u;

            //update v
            double sum_v = 0.0;
            denom = 0.0;
	    
            for (uint z=0; z < zDim_; z++) {
	      
              double dx = gradient_[z](x,y,0);
              double dy = gradient_[z](x,y,1);
              double dt = t_derivative_(x,y,z);
	      
              sum_v -= cur_data_diff*dy*(dx*flow_increment_(x,y,0)  + dt);
              denom += cur_data_diff*dy*dy;
            }
	    
            if (x > 0) {
              sum_v += hdiff(x-1,y,1)*(flow_increment_(x-1,y,1) + flow_(x-1,y,1) - flow_(x,y,1));
              denom += hdiff(x-1,y,1);
            }
            if (x+1 < xDim_) {
              sum_v += hdiff(x,y,1)*(flow_increment_(x+1,y,1) + flow_(x+1,y,1) - flow_(x,y,1));
              denom += hdiff(x,y,1);
            }
            if (y > 0) {
              sum_v += vdiff(x,y-1,1)*(flow_increment_(x,y-1,1) + flow_(x,y-1,1) - flow_(x,y,1));
              denom += vdiff(x,y-1,1);
            }
            if (y+1 < yDim_) {
              sum_v += vdiff(x,y,1)*(flow_increment_(x,y+1,1) + flow_(x,y+1,1) - flow_(x,y,1));
              denom += vdiff(x,y,1);
            }

            sum_v /= denom;
            flow_increment_(x,y,1) = neg_omega * flow_increment_(x,y,1) + omega * sum_v;
          }
        }
      }
    }

    if (scale_ == 1.0)
      std::cerr << "addon energy: " << addon_energy(flow_increment_) << std::endl;
  }
}



void MotionEstimator::compute_flow_increment_tv_l2_nesterov() {

  std::cerr << "tv-l2 increment (Nesterov's optimal method)" << std::endl;

  flow_increment_.set_constant(0.0);

#ifdef LIPSCHITZ_SMOOTHING

  Math3D::Tensor<double> aux_flow_increment = flow_increment_;
  
  Math3D::NamedTensor<double> gradient(xDim_,yDim_,2,MAKENAME(gradient));
  
  double prev_t = 1.0;

  for (uint iter = 1; iter <= 250; iter++) {

    //1.) compute gradient
    gradient.set_constant(0.0);

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

        const double u = aux_flow_increment(x,y,0);
        const double v = aux_flow_increment(x,y,1);
       
        for (uint z=0; z < zDim_; z++) {
          const double dx = gradient_[z](x,y,0);
          const double dy = gradient_[z](x,y,1);
          const double dt = t_derivative_(x,y,z);
	 
          gradient(x,y,0) += dx*(dx*u + dy*v + dt);
          gradient(x,y,1) += dy*(dx*u + dy*v + dt);
        }

        if (x > 0 || y > 0) {

          for (uint i=0; i < 2; i++) {
            double hdiff = (x > 0) ? flow_(x,y,i) + aux_flow_increment(x,y,i) - flow_(x-1,y,i) - aux_flow_increment(x-1,y,i) : 0.0; 
            double vdiff = (y > 0) ? flow_(x,y,i) + aux_flow_increment(x,y,i) - flow_(x,y-1,i) - aux_flow_increment(x,y-1,i) : 0.0; 

            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);

            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (hdiff+vdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (hdiff+vdiff) / norm;
            }
          }
        }
        if (x + 1 < xDim_) {

          for (uint i=0; i < 2; i++) {
            double hdiff = flow_(x,y,i) + aux_flow_increment(x,y,i) - flow_(x+1,y,i) - aux_flow_increment(x+1,y,i); 
            double vdiff = (y > 0) ? flow_(x+1,y,i) + aux_flow_increment(x+1,y,i) - flow_(x+1,y-1,i) - aux_flow_increment(x+1,y-1,i) : 0.0; 
	  
            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);
	    
            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (hdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (hdiff) / norm;
            }
          }
        }
        if (y + 1 < yDim_) {

          for (uint i=0; i < 2; i++) {
            double hdiff = (x > 0) ? flow_(x,y+1,i) + aux_flow_increment(x,y+1,i) - flow_(x-1,y+1,i) - aux_flow_increment(x-1,y+1,i) : 0.0;
            double vdiff = flow_(x,y,i) + aux_flow_increment(x,y,i) - flow_(x,y+1,i) - aux_flow_increment(x,y+1,i);
	    
            double norm = sqrt(hdiff*hdiff+vdiff*vdiff);
	    
            if (norm <= epsilon_) {
              gradient(x,y,i) += 0.5 * alpha_ * norm * (vdiff) / epsilon_;
            }
            else {
              gradient(x,y,i) += 0.5 * alpha_ * (vdiff) / norm;
            }
          }
        }
      }
    }

    double stepsize = 0.5e-5;

    for (uint i=0; i < flow_.size(); i++)
      aux_flow_increment.direct_access(i) -= stepsize * gradient.direct_access(i);
    
    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;
    //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
	
    for (uint i=0; i < aux_flow_increment.size(); i++) {
      
      const double old_aux = aux_flow_increment.direct_access(i);
      aux_flow_increment.direct_access(i) = old_aux + nesterov_fac*(old_aux - flow_increment_.direct_access(i)) ;
      flow_increment_.direct_access(i) = old_aux;
    }
    
    prev_t = new_t; 
  }
#else
  INTERNAL_ERROR << " Nesterov's method is not applicable for this kind of smoothing. Exiting." << std::endl;
  exit(1);
#endif
}

void MotionEstimator::compute_flow_lbfgs(int L, uint nIter) {

  if (!spline_mode_) {
    INTERNAL_ERROR << " this method requires a smooth representation of the images" << std::endl;
    exit(1);
  }

  assert( !linearized_functional_);

  Storage1D<Math3D::Tensor<double> > step(L);
  Storage1D<Math3D::Tensor<double> > grad_diff(L);
  
  Math1D::Vector<double> rho(L);

  for (int l=0; l < L; l++) {

    grad_diff[l].resize(xDim_,yDim_,2);
    step[l].resize(xDim_,yDim_,2);
  } 

  flow_.set_constant(0.0);

  Math3D::Tensor<double> cur_grad(xDim_,yDim_,2);

  double cur_energy = energy();

  int start_iter = 1;

  for (int iter = 1; iter <= int(nIter); iter++) {

    std::cerr << "##### iteration " << iter << ", energy: " << cur_energy << std::endl;

    // a) compute current gradient

    cur_grad.set_constant(0.0); //needed???

    // a1) data term
    
    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

	double u = flow_(x,y,0);
	double v = flow_(x,y,1);
	
	for (uint z=0; z < zDim_; z++) {
	
	  double diff = first_(x,y,z) - interpolate_2D_qspline(second_spline_coeffs_[z], x+u,y+v);
	  Math1D::Vector<double> spline_grad =  interpolate_2D_qspline_grad(second_spline_coeffs_[z], x+u,y+v);
	  
	  if (data_norm_ == SquaredDiffs) {
	    cur_grad(x,y,0) -= 2.0 * diff * spline_grad[0];
	    cur_grad(x,y,1) -= 2.0 * diff * spline_grad[1];
	  }
	  else {
#ifdef LIPSCHITZ_SMOOTHING
	    TODO("Lipschitz smoothing");
#else
	    cur_grad(x,y,0) -= diff * spline_grad[0] / sqrt(diff*diff + epsilon_);
	    cur_grad(x,y,1) -= diff * spline_grad[1] / sqrt(diff*diff + epsilon_);
#endif
	  }
	}
      }
    }

    // a2) regularity term

    for (uint y=0; y < yDim_; y++) {
      for (uint x=0; x < xDim_; x++) {

	double u = flow_(x,y,0);
	double v = flow_(x,y,1);

	if (reg_norm_ == L2) {

	  if (x > 0) {
	    
	    double diff_u = u - flow_(x-1,y,0);
	    double diff_v = v - flow_(x-1,y,1);
	    
	    cur_grad(x,y,0) += alpha_ * 2.0 *  diff_u;
	    cur_grad(x,y,1) += alpha_ * 2.0 *  diff_v;
	  }
	  if (y > 0) {
	    
	    double diff_u = u - flow_(x,y-1,0);
	    double diff_v = v - flow_(x,y-1,1);
	    
	    cur_grad(x,y,0) += alpha_ * 2.0 *  diff_u;
	    cur_grad(x,y,1) += alpha_ * 2.0 *  diff_v;
	  }
	  
	  if (x+1 < xDim_) {
	    double diff_u = u - flow_(x+1,y,0);
	    double diff_v = v - flow_(x+1,y,1);

	    cur_grad(x,y,0) += alpha_ * 2.0 *  diff_u;
	    cur_grad(x,y,1) += alpha_ * 2.0 *  diff_v;
	  }
	  if (y+1 < yDim_) {
	    double diff_u = u - flow_(x,y+1,0);
	    double diff_v = v - flow_(x,y+1,1);

	    cur_grad(x,y,0) += alpha_ * 2.0 *  diff_u;
	    cur_grad(x,y,1) += alpha_ * 2.0 *  diff_v;
	  }

	}
	else {

	  for (uint i=0; i < 2; i++) {

	    double hdiff = (x > 0) ? u - flow_(x-1,y,i) : 0.0;
	    double vdiff = (y > 0) ? u - flow_(x,y-1,i) : 0.0;

#ifdef LIPSCHITZ_SMOOTHING
	    TODO("Lipschitz smoothing");
#else
	    double norm = sqrt(hdiff*hdiff + vdiff*vdiff + epsilon_);
	    cur_grad(x,y,i) += alpha_ * (hdiff + vdiff) / norm;
	    if (x > 0)
	      cur_grad(x-1,y,i) -= alpha_ * hdiff / norm;
	    if (y > 0)
	      cur_grad(x,y-1,i) -= alpha_ * vdiff / norm;
#endif
	  }

	}
      }
    }

    double sqr_grad_norm = cur_grad.sqr_norm();
    std::cerr << "sqr grad norm: " << sqr_grad_norm << std::endl;
    if (sqr_grad_norm < 0.01)
      break; //problem solved


    //b) compute search direction

    
    double cur_curv = 0.0;

    if (iter > 1) {
      //update grad_diff and rho
      uint cur_l = (iter-1) % L;

      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];
      const Math3D::Tensor<double>& cur_step = step[cur_l];

      double cur_rho = 0.0;

      for (uint k=0; k < cur_step.size(); k++) {
	
	//cur_grad_diff was set to minus the previous gradient at the end of the previous iteration
	cur_grad_diff.direct_access(k) += cur_grad.direct_access(k);
	cur_rho += cur_grad_diff.direct_access(k) * cur_step.direct_access(k);
      }

      cur_curv = cur_rho / cur_grad_diff.sqr_norm();

      if (cur_curv <= 0.0) {

	//this can happen if the function is not (strictly) convex, since we do not enforce Wolfe part 2
	//  (would need Algorithm 3.5 from [Nocedal & Wright] to enforce that, backtracking line search is NOT enough)
	// Our solution is to simply restart L-BFGS

	std::cerr << "RESTART" << std::endl;

	start_iter = iter;
      }

      rho[cur_l] = 1.0 / cur_rho;
    }


    // b) compute search direction

    flow_increment_ = cur_grad;

    if (iter > start_iter) {

      const int cur_first_iter = std::max<int>(start_iter,iter-L);

      Math1D::Vector<double> alpha(L);
      
      //first loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = iter-1; prev_iter >= cur_first_iter; prev_iter--) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double cur_alpha = 0.0; 
	for (uint k=0; k < cur_step.size(); k++) {
	  cur_alpha += flow_increment_.direct_access(k) * cur_step.direct_access(k);
	}
	cur_alpha *= rho[prev_l];
	alpha[prev_l] = cur_alpha;
	
	for (uint k=0; k < cur_step.size(); k++) {
	  flow_increment_.direct_access(k) -= cur_alpha * cur_grad_diff.direct_access(k);
	}
      }
      
      //we use a scaled identity as base matrix (q=r=flow_increment_)
      flow_increment_ *= cur_curv;
      
      //second loop in Algorithm 7.4 from [Nocedal & Wright]
      for (int prev_iter = cur_first_iter; prev_iter < iter; prev_iter++) {
	
	uint prev_l = prev_iter % L;
	
	const Math3D::Tensor<double>& cur_step = step[prev_l];
	const Math3D::Tensor<double>& cur_grad_diff = grad_diff[prev_l];
	
	double beta = 0.0; 
	for (uint k=0; k < cur_step.size(); k++) {
	  beta += flow_increment_.direct_access(k) * cur_grad_diff.direct_access(k);
	}
	beta *= rho[prev_l];

	const double gamma = alpha[prev_l] - beta;
	
	for (uint k=0; k < cur_step.size(); k++) {
	  flow_increment_.direct_access(k) += cur_step.direct_access(k) * gamma;
	}
      }
    }
    else {
      flow_increment_ *= 10.0 / sqrt(flow_increment_.sqr_norm());
    }

    negate(flow_increment_);

    //DEBUG
    double search_dir_test = 0.0;
    for (uint k=0; k < cur_grad.size(); k++) 
      search_dir_test += cur_grad.direct_access(k) * flow_increment_.direct_access(k);

    std::cerr << "search dir test: " << search_dir_test << ", steepest descent would give " 
	      << (-cur_grad.sqr_norm()) << std::endl;
    assert(search_dir_test < 0.0);
    //END_DEBUG

    // c) line search

    uint inner_iter = 0;
      
    double gain;
    double hyp_new_energy; 

    uint nMaxIter = (start_iter == iter) ? 25 : 10;

    while (inner_iter < nMaxIter) {
      
      inner_iter++;
      
      //check if the increment reduces the energy
      hyp_new_energy = hyp_energy(flow_increment_);
      
      gain = cur_energy - hyp_new_energy;
      
      std::cerr << "new energy: " << hyp_new_energy << std::endl;
      
      std::cerr << "gain: " << gain << std::endl;

      std::cerr << " increment norm: " << sqrt(flow_increment_.sqr_norm()) << std::endl;
      
      if (gain > 0.0)
	break;
      else
	flow_increment_ *= 0.5;
    }
      

    // d) update
    if (gain > 0.0) {
      uint cur_l = (iter % L);

      Math3D::Tensor<double>& cur_step = step[cur_l];
      Math3D::Tensor<double>& cur_grad_diff = grad_diff[cur_l];
      
      for (uint k=0; k < cur_step.size(); k++) {
	double step = flow_increment_.direct_access(k);
	cur_step.direct_access(k) = step;
	flow_.direct_access(k) += step;
	
	//prepare for the next iteration
	cur_grad_diff.direct_access(k) = -cur_grad.direct_access(k);
      }
      
      cur_energy = hyp_new_energy;
    }
    else {
      std::cerr << "WARNING: failed to get descent => RESTART" << std::endl;
      if (start_iter == iter)
	break;
      start_iter = iter+1;
    }
    

  }
}
