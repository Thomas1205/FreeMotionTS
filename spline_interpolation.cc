/**** written by Thomas Schoenemann as an employee of Lund University, Sweden, 2011 ****/

#include "spline_interpolation.hh"

#include <cmath>

double quadratic_spline_1D(double x) {

  x = fabs(x); //the function is symmetric

  if (x >= 1.5)
    return 0.0;

  if (x <= 0.5) {
    x = 0.5 - x;
    return -1.0*x*x + x + 0.5;
  }
  else {
    x -= 1.0;
    x = 0.5 - x;
    return 0.5*x*x; 
  }
}

double quadratic_spline_prime_1D(double x) {

  double abs_x = fabs(x);

  if (abs_x >= 1.5)
    return 0.0;
  
  if (abs_x <= 0.5) {

    //we can multiply by sign(x) since the function is symmetric (and hence the derivative anti-symmetric)
    return sign(x) * (2.0*(0.5-abs_x) - 1.0);
  }
  else {

    return sign(x) * -1.0 * (0.5 - (abs_x - 1.0));
  }
}

double cubic_spline_1D(double x) {

  x = fabs(x); //the function is symmetric
  if (x >= 2.0)
    return 0.0;
  else {

    x *= 0.5;
    double dx = x*x;
    
    return 0.5 * (1.0 + dx * (2.0*x - 3.0));
  }
}

double cubic_spline_prime_1D(double x) {

  double abs_x = fabs(x);

  //we can multiply by sign(x) since the function is symmetric (and hence the derivative anti-symmetric)

  if (abs_x >= 2.0)
    return 0.0;
  else
    return sign(x) * (0.375 * abs_x*abs_x - 0.75*abs_x);
}

void compute_qspline_coefficients(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficient) {

  uint size = input.size();
  
  coefficient.resize(input.size());
  for (uint k=0; k < size; k++) 
    coefficient[k] = input[k];

  //we use Gauss-Seidel
  for (uint iter = 1; iter <= 1000; iter++) {

    for (uint k=0; k < size; k++) {

      double sum = input[k];

      if (k > 0) {
        sum -= 0.125*coefficient[k-1];
      }
      if (k+1 < size) {
        sum -= 0.125*coefficient[k+1];
      }
      coefficient[k] = sum / 0.75;
    }
  }
}

void compute_qspline_coefficients_overdetermined(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficient) {

  //the determined spline function should roughly match the given input values at the given points.
  // moreover, its derivatives should roughly match the induced discrete derivatives of the input vector

  uint size = input.size();
  
  coefficient.resize(input.size());
  coefficient.set_constant(0.0);

  //we use CG

  double spline_prime = quadratic_spline_prime_1D(1.0);

  uint nRows = size + size-2;

  Math1D::Vector<double> ideal_values(nRows);
  for (uint i=0; i < size; i++)
    ideal_values[i] = input[i];
  for (uint i=0; i < size-2; i++)
    ideal_values[size+i] = 0.5*(input[i+2] - input[i]);

  Math1D::Vector<double> ax(nRows);

  Math1D::Vector<double> AATp(size);

  Math1D::Vector<double> residuum(size,0.0);
  
  //init: residuum = A^T * ideal_values
  for (uint i=0; i < size; i++) {

    residuum[i] += 0.75*ideal_values[i];
    
    if (i > 0)
      residuum[i-1] += 0.125*ideal_values[i];
    if (i+1 < size)
      residuum[i+1] += 0.125*ideal_values[i];
  }
  for (uint i=0; i < size-2; i++) {
    
    residuum[i] -= spline_prime * ideal_values[size+i];
    residuum[i+2] += spline_prime * ideal_values[size+i];
  }


  Math1D::Vector<double> dir = residuum;

  for (uint iter=1; iter <= 25; iter++) {

    //compute product A^T*A * dir

    double alpha_denom = 0.0;

    //1. A*dir
    for (uint i=0; i < size; i++) {
      
      double sum = 0.75 * dir[i];
      if (i > 0)
        sum += 0.125 * dir[i-1];
      if (i+1 < size)
        sum += 0.125 * dir[i+1];

      ax[i] = sum;
    }
    for (uint i=0; i < size-2; i++) {

      ax[size+i] = -spline_prime * dir[i] + spline_prime * dir[i+2];
    }
    
    //2. A^T * ax    
    AATp.set_constant(0.0);

    for (uint i=0; i < size; i++) {

      AATp[i] += 0.75*ax[i];
      
      if (i > 0)
        AATp[i-1] += 0.125*ax[i];
      if (i+1 < size)
        AATp[i+1] += 0.125*ax[i];
    }
    for (uint i=0; i < size-2; i++) {

      AATp[i] -= spline_prime * ax[size+i];
      AATp[i+2] += spline_prime * ax[size+i];
    }

    for (uint i=0; i < size; i++) {
      alpha_denom += AATp[i] * dir[i];
    }

    double old_res_norm = residuum.sqr_norm();

    std::cerr << "residual norm: " << old_res_norm << std::endl;

    double alpha = old_res_norm / alpha_denom;
    assert(!isnan(alpha));

    for (uint i=0; i < size; i++) {
      coefficient[i] += alpha*dir[i];
      residuum[i] -= alpha * AATp[i];
    }
    
    double beta = residuum.sqr_norm() / old_res_norm;

    for (uint i=0; i < size; i++) {
      dir[i] = beta*dir[i] + residuum[i];
    }
  }

}

void compute_qspline_coefficients_overdetermined2(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficient) {

  //the determined spline function should roughly match the given input values at the given points.
  // moreover, its derivatives should roughly match the induced discrete derivatives of the input vector

  uint size = input.size();
  
  coefficient.resize(input.size());
  coefficient.set_constant(0.0);

  //we use CG

  double spline_prime_cur = quadratic_spline_prime_1D(0.5);
  double spline_prime_next = quadratic_spline_prime_1D(-0.5);

  uint nRows = size + size-1;

  Math1D::Vector<double> ideal_values(nRows);
  for (uint i=0; i < size; i++)
    ideal_values[i] = input[i];
  for (uint i=0; i < size-1; i++)
    ideal_values[size+i] = (input[i+1] - input[i]);

  Math1D::Vector<double> ax(nRows);

  Math1D::Vector<double> AATp(size);

  Math1D::Vector<double> residuum(size,0.0);
  
  //init: residuum = A^T * ideal_values
  for (uint i=0; i < size; i++) {

    residuum[i] += 0.75*ideal_values[i];
    
    if (i > 0)
      residuum[i-1] += 0.125*ideal_values[i];
    if (i+1 < size)
      residuum[i+1] += 0.125*ideal_values[i];
  }
  for (uint i=0; i < size-1; i++) {
    
    residuum[i] += spline_prime_cur * ideal_values[size+i];
    residuum[i+1] += spline_prime_next * ideal_values[size+i];
  }


  Math1D::Vector<double> dir = residuum;

  for (uint iter=1; iter <= 25; iter++) {

    //compute product A^T*A * dir

    double alpha_denom = 0.0;

    //1. A*dir
    for (uint i=0; i < size; i++) {
      
      double sum = 0.75 * dir[i];
      if (i > 0)
        sum += 0.125 * dir[i-1];
      if (i+1 < size)
        sum += 0.125 * dir[i+1];

      ax[i] = sum;
    }
    for (uint i=0; i < size-1; i++) {

      ax[size+i] = spline_prime_cur * dir[i] + spline_prime_next * dir[i+1];
    }
    
    //2. A^T * ax    
    AATp.set_constant(0.0);

    for (uint i=0; i < size; i++) {

      AATp[i] += 0.75*ax[i];
      
      if (i > 0)
        AATp[i-1] += 0.125*ax[i];
      if (i+1 < size)
        AATp[i+1] += 0.125*ax[i];
    }
    for (uint i=0; i < size-2; i++) {

      AATp[i] += spline_prime_cur * ax[size+i];
      AATp[i+1] += spline_prime_next * ax[size+i];
    }

    for (uint i=0; i < size; i++) {
      alpha_denom += AATp[i] * dir[i];
    }

    double old_res_norm = residuum.sqr_norm();

    std::cerr << "residual norm: " << old_res_norm << std::endl;

    double alpha = old_res_norm / alpha_denom;
    assert(!isnan(alpha));

    for (uint i=0; i < size; i++) {
      coefficient[i] += alpha*dir[i];
      residuum[i] -= alpha * AATp[i];
    }
    
    double beta = residuum.sqr_norm() / old_res_norm;

    for (uint i=0; i < size; i++) {
      dir[i] = beta*dir[i] + residuum[i];
    }
  }
}



void compute_cubic_spline_coefficients(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficient) {

  uint size = input.size();
  
  coefficient.resize(input.size());

  //!!!!! it seems Gauss-Seidel does not work here (not diagonally dominant)
  
  //we use CG
  coefficient.set_constant(0.0);

  Math1D::Vector<double> residual(size);
  for (uint k=0; k < size; k++)
    residual[k] = input[k];

  Math1D::Vector<double> direction = residual;

  Math1D::Vector<double> ap(size,0.0);

  double prev_norm = residual.sqr_norm();

  for (uint iter=1; iter <= 10000; iter++) {

    //multiply the direction by the matrix
    for (uint k=0; k < size; k++) {
      
      double sum = 0.5*direction[k];

      if (k > 0)
        sum += 0.25*direction[k-1];
      if (k+1 < size)
        sum += 0.25*direction[k+1];
      
      ap[k] = sum;
    }

    double alpha = 0.0;
    for (uint k=0; k < size; k++) 
      alpha += direction[k] * ap[k];

    alpha /= prev_norm;

    double new_norm = 0.0;

    for (uint k=0; k < size; k++) {
      coefficient[k] += alpha*direction[k];
      residual[k] -= alpha*ap[k];

      new_norm += residual[k]*residual[k];
    }

    double beta = new_norm / prev_norm;

    for (uint k=0; k < size; k++) 
      direction[k] = residual[k] + beta*direction[k];

    prev_norm = new_norm;
  }

}			   

void compute_cubic_spline_coefficients_overdetermined2(const Math1D::Vector<float>& input, Math1D::Vector<double>& coefficient) {



  //the determined spline function should roughly match the given input values at the given points.
  // moreover, its derivatives should roughly match the induced discrete derivatives of the input vector

  uint size = input.size();
  
  coefficient.resize(input.size());
  coefficient.set_constant(0.0);

  //we use CG
  double spline_prime_prev = cubic_spline_prime_1D(1.5);
  double spline_prime_cur = cubic_spline_prime_1D(0.5);
  double spline_prime_next = cubic_spline_prime_1D(-0.5);
  double spline_prime_next_plus = cubic_spline_prime_1D(-1.5);

  uint nRows = size + size-1;

  Math1D::Vector<double> ideal_values(nRows);
  for (uint i=0; i < size; i++)
    ideal_values[i] = input[i];
  for (uint i=0; i < size-1; i++)
    ideal_values[size+i] = (input[i+1] - input[i]);

  Math1D::Vector<double> ax(nRows);

  Math1D::Vector<double> AATp(size);

  Math1D::Vector<double> residuum(size,0.0);
  
  //init: residuum = A^T * ideal_values
  for (uint i=0; i < size; i++) {

    residuum[i] += 0.5*ideal_values[i];
    
    if (i > 0)
      residuum[i-1] += 0.25*ideal_values[i];
    if (i+1 < size)
      residuum[i+1] += 0.25*ideal_values[i];
  }
  for (uint i=0; i < size-1; i++) {
    
    residuum[i] += spline_prime_cur * ideal_values[size+i];
    residuum[i+1] += spline_prime_next * ideal_values[size+i];
    if (i > 0)
      residuum[i-1] += spline_prime_prev * ideal_values[size+i];
    if (i+2 < size)
      residuum[i+2] += spline_prime_next_plus * ideal_values[size+i];
  }


  Math1D::Vector<double> dir = residuum;

  for (uint iter=1; iter <= 25; iter++) {

    //compute product A^T*A * dir

    double alpha_denom = 0.0;

    //1. A*dir
    for (uint i=0; i < size; i++) {
      
      double sum = 0.5 * dir[i];
      if (i > 0)
        sum += 0.25 * dir[i-1];
      if (i+1 < size)
        sum += 0.25 * dir[i+1];

      ax[i] = sum;
    }
    for (uint i=0; i < size-1; i++) {

      ax[size+i] = spline_prime_cur * dir[i] + spline_prime_next * dir[i+1];
      if (i > 0)
        ax[size+i] += spline_prime_prev * dir[i-1];
      if (i+2 < size)
        ax[size+i] += spline_prime_next_plus * dir[i+2];
    }
    
    //2. A^T * ax    
    AATp.set_constant(0.0);

    for (uint i=0; i < size; i++) {

      AATp[i] += 0.5*ax[i];
      
      if (i > 0)
        AATp[i-1] += 0.25*ax[i];
      if (i+1 < size)
        AATp[i+1] += 0.25*ax[i];
    }
    for (uint i=0; i < size-2; i++) {

      AATp[i] += spline_prime_cur * ax[size+i];
      AATp[i+1] += spline_prime_next * ax[size+i];
      if (i > 0)
        AATp[i-1] += spline_prime_prev * ax[size+i];
      if (i+2 < size)
        AATp[i+2] += spline_prime_next_plus * ax[size+i];
    }

    for (uint i=0; i < size; i++) {
      alpha_denom += AATp[i] * dir[i];
    }

    double old_res_norm = residuum.sqr_norm();

    std::cerr << "residual norm: " << old_res_norm << std::endl;

    double alpha = old_res_norm / alpha_denom;
    assert(!isnan(alpha));

    for (uint i=0; i < size; i++) {
      coefficient[i] += alpha*dir[i];
      residuum[i] -= alpha * AATp[i];
    }
    
    double beta = residuum.sqr_norm() / old_res_norm;

    for (uint i=0; i < size; i++) {
      dir[i] = beta*dir[i] + residuum[i];
    }
  }
}


double interpolate_qspline(const Math1D::Vector<double>& coefficients, double x) {

  int last = coefficients.size()-1;

  //true interpolation
  if (x < 0.0)
    x = 0.0;
  if (x > last)
    x = last;

  double sum = 0.0;

  int k = (int) floor(x);
  for (int kk = k-2; kk <= k+2; kk++) {

    if (kk >= 0 && kk < last)
      sum += coefficients[kk] * quadratic_spline_1D( x  - kk );
  }

  return sum;
}


double interpolate_cubic_spline(const Math1D::Vector<double>& coefficients, double x) {

  int last = coefficients.size()-1;

  //true interpolation
  if (x < 0.0)
    x = 0.0;
  if (x > last)
    x = last;

  double sum = 0.0;

  int k = (int) floor(x);
  for (int kk = k-2; kk <= k+2; kk++) {

    if (kk >= 0 && kk < last)
      sum += coefficients[kk] * cubic_spline_1D( x  - kk );
  }

  return sum;
}



/************** 2-D functionality ***************/

double quadratic_spline_xprime_2D(double x, double y) {

  return quadratic_spline_prime_1D(x) * quadratic_spline_1D(y);
}

double quadratic_spline_yprime_2D(double x, double y) {

  return quadratic_spline_1D(x) * quadratic_spline_prime_1D(y);
}


double interpolate_2D_qspline(const Math2D::Matrix<double>& coefficients, double x, double y) {

  int xlast = coefficients.xDim()-1;
  int ylast = coefficients.yDim()-1;

  //true interpolation
  if (x < 0.0)
    x = 0.0;
  if (x > xlast)
    x = xlast;

  if (y < 0.0)
    y = 0.0;
  if (y > ylast)
    y = ylast;

  double sum = 0.0;

  int kx = (int) floor(x);
  int ky = (int) floor(y);

  for (int kkx = kx-2; kkx <= kx+2; kkx++) {
    for (int kky = ky-2; kky <= ky+2; kky++) {

      if (kkx >= 0 && kkx < xlast &&
          kky >= 0 && kky < ylast)
        sum += coefficients(kkx,kky) * cubic_spline_1D( x  - kkx ) * cubic_spline_1D( y  - kky );
    }
  }

  return sum;
}

Math1D::Vector<double> interpolate_2D_qspline_grad(const Math2D::Matrix<double>& coefficients, double x, double y) {
  
  Math1D::Vector<double> grad(2,0.0);

  int xlast = coefficients.xDim()-1;
  int ylast = coefficients.yDim()-1;

  //true interpolation
  if (x < 0.0)
    x = 0.0;
  if (x > xlast)
    x = xlast;

  if (y < 0.0)
    y = 0.0;
  if (y > ylast)
    y = ylast;

  int kx = (int) floor(x);
  int ky = (int) floor(y);

  for (int kkx = kx-2; kkx <= kx+2; kkx++) {
    for (int kky = ky-2; kky <= ky+2; kky++) {

      if (kkx >= 0 && kkx < xlast &&
          kky >= 0 && kky < ylast) {
        grad[0] += coefficients(kkx,kky) * quadratic_spline_xprime_2D(x - kkx, y - kky);
        grad[1] += coefficients(kkx,kky) * quadratic_spline_yprime_2D(x - kkx, y - kky);
        //sum += coefficients(kkx,kky) * cubic_spline_1D( x  - kkx ) * cubic_spline_1D( y  - kky );
      }
    }
  }

  return grad;
}

void compute_2D_qspline_coefficients(const Math2D::Matrix<float>& input, Math2D::Matrix<double>& coefficient) {

  //we use CG
  uint xDim = input.xDim();
  uint yDim = input.yDim();

  coefficient.resize(xDim,yDim);
  coefficient.set_constant(0.0);

  Math2D::Matrix<double> residuum(xDim,yDim);
  for (uint i=0; i < input.size(); i++)
    residuum.direct_access(i) = input.direct_access(i);

  Math2D::Matrix<double> dir = residuum;

  Math2D::Matrix<double> ax(xDim,yDim);

  for (uint iter = 1; iter <= 100; iter++) {

    //compute A*dir
    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            sum += quadratic_spline_1D(y-yy) * quadratic_spline_1D(x-xx) * dir(xx,yy);

        ax(x,y) = sum;
      }
    }
      
    double alpha = 0.0;
    double alpha_denom = 0.0;
    for (uint i=0; i < input.size(); i++) {
      alpha += dir.direct_access(i) * ax.direct_access(i);
      alpha_denom += residuum.direct_access(i) * residuum.direct_access(i);
    }
    
    alpha /= alpha_denom;

    double new_norm = 0.0;

    for (uint i=0; i < input.size(); i++) {

      coefficient.direct_access(i) += alpha * dir.direct_access(i);
      residuum.direct_access(i) -= alpha * ax.direct_access(i);

      new_norm += residuum.direct_access(i) * residuum.direct_access(i);
    }

    std::cerr << "new residuual norm: " << new_norm << std::endl;

    double beta = new_norm / alpha_denom;

    for (uint i=0; i < input.size(); i++) {

      dir.direct_access(i) = residuum.direct_access(i) + beta * dir.direct_access(i);
    }
  }

}


void compute_2D_qspline_coefficients_overdetermined2(const Math2D::Matrix<float>& input, Math2D::Matrix<double>& coefficient) {

  
  uint xDim = input.xDim();
  uint yDim = input.yDim();

  coefficient.resize(xDim,yDim);
  coefficient.set_constant(0.0);

  Math2D::Matrix<double> ideal_xgrad(xDim-1,yDim);
  for (uint y=0; y < yDim; y++)
    for (uint x=0; x < xDim-1; x++)
      ideal_xgrad(x,y) = input(x+1,y) - input(x,y);

  Math2D::Matrix<double> ideal_ygrad(xDim,yDim-1);
  for (uint y=0; y < yDim-1; y++)
    for (uint x=0; x < xDim; x++)
      ideal_ygrad(x,y) = input(x,y+1) - input(x,y);

  Math2D::Matrix<double> inter_image(xDim,yDim,0.0);
  Math2D::Matrix<double> inter_xgrad(xDim-1,yDim,0.0);
  Math2D::Matrix<double> inter_ygrad(xDim,yDim-1,0.0);

#if 0
  //we use CG (but it seems the matrix is not full rank)

  Math2D::Matrix<double> residuum(xDim,yDim,0.0);

  //init residuum
  for (int y=0; y < (int) yDim; y++) {
    for (int x=0; x < (int) xDim; x++) {
      
      for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
        for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
          residuum(xx,yy) += quadratic_spline_1D(y-yy) * quadratic_spline_1D(x-xx) * input(x,y);
    }
  }

  for (int y=0; y < (int) yDim; y++) {
    for (int x=0; x < (int) xDim-1; x++) {

      for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
        for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+2); xx++)
          residuum(xx,yy) += quadratic_spline_xprime_2D(x + 0.5 - xx, y - yy) * ideal_xgrad(x,y);
    }
  }    

  for (int y=0; y < (int) yDim-1; y++) {
    for (int x=0; x < (int) xDim; x++) {

      for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+2); yy++)
        for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
          residuum(xx,yy) += quadratic_spline_yprime_2D(x  - xx, y + 0.5 - yy) * ideal_ygrad(x,y);
    }
  }

  Math2D::Matrix<double> dir = residuum;

  Math2D::Matrix<double> ax(xDim,yDim);

  for (uint iter = 1; iter <= 100; iter++) {

    //compute A^T*A*dir

    //a) compute A*dir
    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            sum += quadratic_spline_1D(y-yy) * quadratic_spline_1D(x-xx) * dir(xx,yy);

        inter_image(x,y) = sum;
      }
    }

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim-1; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+2); xx++)
            sum += quadratic_spline_xprime_2D(x + 0.5 - xx, y - yy) * dir(xx,yy) ;

        inter_xgrad(x,y) = sum;
      }
    }    
    
    for (int y=0; y < (int) yDim-1; y++) {
      for (int x=0; x < (int) xDim; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+2); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            sum += quadratic_spline_yprime_2D(x  - xx, y + 0.5 - yy) * dir(xx,yy) ;

        inter_ygrad(x,y) = sum;
      }
    }

    //b) compute A^T*inter*
    ax.set_constant(0.0);

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            ax(xx,yy) += quadratic_spline_1D(y-yy) * quadratic_spline_1D(x-xx) * inter_image(x,y);
      }
    }

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim-1; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+2); xx++)
            ax(xx,yy) += quadratic_spline_xprime_2D(x + 0.5 - xx, y - yy) * inter_xgrad(x,y);
      }
    }    

    for (int y=0; y < (int) yDim-1; y++) {
      for (int x=0; x < (int) xDim; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+2); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            ax(xx,yy) += quadratic_spline_yprime_2D(x  - xx, y + 0.5 - yy) * inter_ygrad(x,y);
      }
    }
		    
    double alpha = 0.0;
    double alpha_denom = 0.0;

    for (uint i=0; i < input.size(); i++) {
      alpha += dir.direct_access(i) * ax.direct_access(i);
      alpha_denom += residuum.direct_access(i) * residuum.direct_access(i);
    }
    
    alpha /= alpha_denom;

    double new_norm = 0.0;

    for (uint i=0; i < input.size(); i++) {

      coefficient.direct_access(i) += alpha * dir.direct_access(i);
      residuum.direct_access(i) -= alpha * ax.direct_access(i);

      new_norm += residuum.direct_access(i) * residuum.direct_access(i);
    }

    std::cerr << "new residuual norm: " << new_norm << ", previous: " << alpha_denom << std::endl;

    double beta = new_norm / alpha_denom;

    for (uint i=0; i < input.size(); i++) {

      dir.direct_access(i) = residuum.direct_access(i) + beta * dir.direct_access(i);
    }
  }
#else
  //we use Nesterov's optimal method

  Math2D::Matrix<double> aux_coeff = coefficient;

  Math2D::Matrix<double> grad(xDim,yDim,0.0);

  double prev_t = 1.0;

  for (uint iter = 1; iter <= 100; iter++) {

    std::cerr << "iter " << iter << std::endl;

    //compute A^T*(A*aux_coeff - b) where b denotes the right-hand-side

    //a) compute A*dir
    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++) {
          double qy = quadratic_spline_1D(y-yy);
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            sum += qy * quadratic_spline_1D(x-xx) * aux_coeff(xx,yy);
        }

        inter_image(x,y) = sum;
      }
    }

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim-1; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++) 
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+2); xx++)
            sum += quadratic_spline_xprime_2D(x + 0.5 - xx, y - yy) * aux_coeff(xx,yy);

        inter_xgrad(x,y) = sum;
      }
    }    
    
    for (int y=0; y < (int) yDim-1; y++) {
      for (int x=0; x < (int) xDim; x++) {

        double sum = 0.0;

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+2); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            sum += quadratic_spline_yprime_2D(x  - xx, y + 0.5 - yy) * aux_coeff(xx,yy);

        inter_ygrad(x,y) = sum;
      }
    }

    for (uint i=0; i < input.size(); i++)
      inter_image.direct_access(i) -= input.direct_access(i);

    inter_xgrad -= ideal_xgrad;
    inter_ygrad -= ideal_ygrad;
    

    //b) compute A^T*inter*
    grad.set_constant(0.0);

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++) {
          double qy = quadratic_spline_1D(y-yy);
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            grad(xx,yy) += qy * quadratic_spline_1D(x-xx) * inter_image(x,y);
        }
      }
    }

    for (int y=0; y < (int) yDim; y++) {
      for (int x=0; x < (int) xDim-1; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+1); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+2); xx++)
            grad(xx,yy) += quadratic_spline_xprime_2D(x + 0.5 - xx, y - yy) * inter_xgrad(x,y);
      }
    }    

    for (int y=0; y < (int) yDim-1; y++) {
      for (int x=0; x < (int) xDim; x++) {

        for (int yy = std::max(0,y-1); yy <= std::min<int>(yDim-1,y+2); yy++)
          for (int xx = std::max(0,x-1); xx <= std::min<int>(xDim-1,x+1); xx++)
            grad(xx,yy) += quadratic_spline_yprime_2D(x  - xx, y + 0.5 - yy) * inter_ygrad(x,y);
      }
    }

    double alpha = 0.1;

    for (uint i=0; i < aux_coeff.size(); i++)
      aux_coeff.direct_access(i) -= alpha * grad.direct_access(i);

    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;
    //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
    
    for (uint i=0; i < aux_coeff.size(); i++) {
      
      const double old_aux = aux_coeff.direct_access(i);
      aux_coeff.direct_access(i) = old_aux + nesterov_fac*(old_aux - coefficient.direct_access(i)) ;
      coefficient.direct_access(i) = old_aux;
    }
    
    prev_t = new_t;
  }

#endif
}
