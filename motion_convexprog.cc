/**** written by Thomas Schoenemann as an employee of Lund University, June 2010 ****/


#include "motion_convexprog.hh"
#include "matrix.hh"
//#include "matrix_inversion.hh"
#include "tensor_interpolation.hh"
#include "motion_moves.hh"
#include "projection.hh"

#ifdef HAS_CPLEX
#include <ilcplex/cplex.h>
#endif

#ifdef USE_CUDA
#warning using cuda
#include "motion_convprog.cuh"
#endif

//#define USE_XPRESS

#ifdef USE_XPRESS
#include "xprs.h" 

void XPRS_CC optimizermsg(XPRSprob prob, void* data, const char *sMsg,int nLen,int nMsgLvl) {
  //discard all
}
#endif

//#define USE_EXPLICIT_GRADIENT

#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif


double motion_estimation_quadprog(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                  int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                  uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1);
  const uint nLabels = nHorLabels * nVertLabels;

  Math3D::NamedTensor<double> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          int u = ((int) lx) + min_x_disp;
          int v = ((int) ly) + min_y_disp;
	
          int tx = ((int) x) + u;
          int ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          double disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - second(tx,ty,z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  Math3D::NamedTensor<double> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

  /*** initialization ***/
#if 0
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      double min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

  double energy;

  //NOTE: by construction hmarginal(.,.,nHorLabels-1) would always be 1
  Math3D::NamedTensor<double> hmarginal(xDim,yDim,nHorLabels-1, 0.0, MAKENAME(hmarginal));
  //NOTE: by construction vmarginal(.,.,nVertLabels-1) would always be 1
  Math3D::NamedTensor<double> vmarginal(xDim,yDim,nVertLabels-1, 0.0, MAKENAME(vmarginal));  
  Math3D::NamedTensor<double> grad(xDim,yDim,nLabels,MAKENAME(grad));

  for (uint iter = 1; iter <= 100000; iter++) {

    std::cerr << "*********** iteration " << iter << std::endl;

    /** 1.) gradient computation **/

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        //update h-marginals
        for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
          double sum = 0.0;
          for (uint lv = 0; lv < nVertLabels; lv++)
            sum += var(x,y,lv*nHorLabels + lh);
	  
          if (lh > 0)
            sum += hmarginal(x,y,lh-1);
	  
          assert(sum >= 0.0);
          assert(sum <= 1.001);
	  
          hmarginal(x,y,lh) = sum;
        }
	
        //update v-marginals
        for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
          double sum  = 0.0;
          for (uint lh = 0; lh < nHorLabels; lh++) 
            sum += var(x,y,lv*nHorLabels + lh);
	  
          if (lv > 0)
            sum += vmarginal(x,y,lv-1);

          assert(sum >= 0.0);
          assert(sum <= 1.001);
	  
          vmarginal(x,y,lv) = sum;
        }

      }
    }

    energy = 0.0;

    grad = label_cost;

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        double hgrad = 0.0;

        for (int h=nHorLabels-2; h >= 0; h--) {

          const double cur_marginal = hmarginal(x,y,h);

          double diff = 0.0;
          if (x > 0) {
            const double temp = cur_marginal - hmarginal(x-1,y,h);
            energy += temp*temp;
            diff += temp;
          }
          if (x+1 < xDim) {
            const double temp = cur_marginal - hmarginal(x+1,y,h);
            energy += temp*temp;
            diff += temp;	    
          }
          if (y > 0) {
            const double temp = cur_marginal - hmarginal(x,y-1,h);
            energy += temp*temp;
            diff += temp;
          }
          if (y+1 < yDim) {
            const double temp = cur_marginal - hmarginal(x,y+1,h);
            energy += temp*temp;
            diff += temp;
          }

          diff *= 2.0 * lambda;

          hgrad += diff;

          for (uint v=0; v < nVertLabels; v++)
            grad(x,y, v*nHorLabels + h) += hgrad;
        }

        double vgrad = 0.0;

        for (int v=nVertLabels-2; v>= 0; v--) {

          const double cur_marginal = vmarginal(x,y,v);

          double diff = 0.0;
          if (x > 0) {
            const double temp = cur_marginal - vmarginal(x-1,y,v);
            energy += temp*temp;
            diff += temp;
          }
          if (x+1 < xDim) {
            const double temp = cur_marginal - vmarginal(x+1,y,v);
            energy += temp*temp;
            diff += temp;
          }
          if (y > 0) {
            const double temp = cur_marginal - vmarginal(x,y-1,v);
            energy += temp*temp;
            diff += temp;
          }
          if (y+1 < yDim) {
            const double temp = cur_marginal - vmarginal(x,y+1,v);
            energy += temp*temp;
            diff += temp;
          }

          diff *= 2.0 * lambda;
          vgrad += diff;

          for (uint h=0; h < nHorLabels; h++)
            grad(x,y, v*nHorLabels + h) += vgrad;
        }

      }
    }

    energy *= lambda;

    //std::cerr << "intermediate energy: " << energy << std::endl;

    for (uint i=0; i < var.size(); i++)
      energy += var.direct_access(i) * label_cost.direct_access(i);

    std::cerr << "energy: " << energy << std::endl;

    /** 2.) perform a step of gradient descent **/
    double alpha = 0.00025 / lambda; 
    //double alpha = 0.1 / iter; 

    grad *= (-1.0) * alpha;
    var += grad;

    /** 3.) reprojection [Michelot 1986] **/
    
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
    
        uint nNonZeros = nLabels;
	
        while (nNonZeros > 0) {

          // 	  std::cerr << "nNonZeros: " << nNonZeros << std::endl;

          // 	  std::cerr << "AAA var: ";
          // 	  double vs = 0.0;
          // 	  for (uint l=0; l < nLabels; l++) {
          // 	    std::cerr << var(x,y,l) << ", ";
          // 	    vs += var(x,y,l);
          // 	  }
          // 	  std::cerr << std::endl;
          // 	  std::cerr << "vs(A): " << vs << std::endl;
	
          //a) project onto the plane
          double mean_dev = - 1.0;
          for (uint l=0; l < nLabels; l++)
            mean_dev += var(x,y,l);
	  
          mean_dev /= nNonZeros;
	  
          //b) subtract mean
          bool all_pos = true;

          for (uint l=0; l < nLabels; l++) {

            if (nNonZeros == nLabels || var(x,y,l) != 0.0) {
              var(x,y,l) -= mean_dev;
	      
              if (var(x,y,l) < 0.0)
                all_pos = false;
            }
          }

          // 	  std::cerr << "BBB var: ";
          // 	  vs = 0.0;
          // 	  for (uint l=0; l < nLabels; l++) {
          // 	    std::cerr << var(x,y,l) << ", ";
          // 	    vs += var(x,y,l);
          // 	  }
          // 	  std::cerr << std::endl;
          // 	  std::cerr << "vs(B): " << vs << std::endl;

          if (all_pos)
            break;

          nNonZeros = nLabels;
          for (uint l=0; l < nLabels; l++) {

            if (var(x,y,l) < 1e-8) {
              var(x,y,l) = 0.0;
              nNonZeros--;
            }
          }
        }

        //DEBUG
        // double sum = 0.0;
        // for (uint l=0; l < nLabels; l++) {
        //   sum += var(x,y,l);
        //   assert(var(x,y,l) >= 0.0);
        // }

        // //std::cerr << "sum: " << sum << std::endl;
        // assert(sum > 0.99);
        // assert(sum < 1.01);
        //END_DEBUG

      }
    }
  }  

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      double max_var = -1.0;
      double sum = 0.0;

      int arg_max_x = MAX_UINT;
      int arg_max_y = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          double val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) + min_x_disp;
            arg_max_y = ((int) ly) + min_y_disp;
          }
        }
      }

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

  return energy;
}



#ifdef HAS_CPLEX
double motion_estimation_quadprog_bcd(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                      int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                      uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1);
  const uint nLabels = nHorLabels * nVertLabels;

  Math3D::NamedTensor<double> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          int u = ((int) lx) + min_x_disp;
          int v = ((int) ly) + min_y_disp;
	
          int tx = ((int) x) + u;
          int ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          double disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - second(tx,ty,z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }


  Math2D::NamedMatrix<double> Q(nLabels,nLabels,0.0,MAKENAME(Q));
  Math2D::NamedMatrix<int> Q_ind(nLabels,nLabels,0,MAKENAME(Q_ind));
  for (uint x=0; x < nLabels; x++)
    for (uint y=0; y < nLabels; y++)
      Q_ind(x,y) = x;

  for (uint h_level = 0; h_level < nHorLabels; h_level++) {

    for (uint h1 = 0; h1 <= h_level; h1++) {
      for (uint v1 = 0; v1 < nVertLabels; v1++) {	

        uint i1 = v1*nHorLabels + h1;

        for (uint h2 = 0; h2 <= h_level; h2++) {
          for (uint v2 = 0; v2 < nVertLabels; v2++) {

            uint i2 = v2*nHorLabels + h2;
	    
            Q(i1,i2) += lambda;
          }
        }
      }
    }
  }

#if 1
  for (uint v_level = 0; v_level < nVertLabels; v_level++) {

    for (uint v1 = 0; v1 <= v_level; v1++) {
      for (uint h1 = 0; h1 < nHorLabels; h1++) {
	
        uint i1 = v1*nHorLabels + h1;
	
        for (uint v2 = 0; v2 <= v_level; v2++) {
          for (uint h2 = 0; h2 < nHorLabels; h2++) {

            uint i2 = v2*nHorLabels + h2;
	    
            Q(i1,i2) += lambda;
          }
        }
      }
    }
  }  
#endif

  Q *= 2.0; //since CPLEX has a factor of 0.5 for the quadratic term

  //std::cerr << "Q: " << Q << std::endl;

  Math3D::NamedTensor<double> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

  /*** initialization ***/
#if 0
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      double min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

  double energy;

  Math3D::NamedTensor<double> hmarginal(xDim,yDim,nHorLabels, 0.0, MAKENAME(hmarginal));
  Math3D::NamedTensor<double> vmarginal(xDim,yDim,nVertLabels, 0.0, MAKENAME(vmarginal));  

  int* qstart = new int[nLabels+1];
  int* qcnt = new int[nLabels];
  
  for (uint l=0; l < nLabels; l++) {
    qstart[l] = l*nLabels;
    qcnt[l] = nLabels;
  }
  qstart[nLabels] = nLabels*nLabels;

  int row_start[2];
  row_start[0] = 0;
  row_start[1] = nLabels;
  
  int* col_idx = new int[nLabels];
  double* value = new double[nLabels];
  
  for (uint l=0; l < nLabels; l++) {
    col_idx[l] = l;
    value[l] = 1.0;
  }

  double* sol = new double[nLabels];

  Math1D::Vector<double> var_lb(nLabels,0.0);
  //Math1D::Vector<double> var_ub(nLabels,1.0);
  Math1D::Vector<double> var_ub(nLabels,1e20);

  int status;

  CPXENVptr  env = NULL;
  env = CPXopenCPLEX (&status);

  status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_OFF);
  //status = CPXsetintparam (env, CPX_PARAM_QPMETHOD, CPX_ALG_DUAL);
  status = CPXsetintparam (env, CPX_PARAM_QPMETHOD, CPX_ALG_PRIMAL);

  if ( status ) {
    fprintf (stderr,
             "Failure to turn on screen indicator, error %d.\n", status);
    exit(1);
  }

  //status = CPXsetdblparam(env, CPX_PARAM_BAREPCOMP, 1e-5);

  CPXLPptr      lp = NULL;

  lp = CPXcreateprob (env, &status, "motion-qp");

  Math1D::NamedVector<double> lin_cost(nLabels,0.0,MAKENAME(lin_cost));
  
  status = CPXnewcols (env, lp, nLabels, lin_cost.direct_access(), var_lb.direct_access(), 
                       var_ub.direct_access(), NULL, NULL);

  double rhs = 1.0;
  char row_sense = 'E';

  status = CPXaddrows(env, lp, 0, 1, nLabels, &rhs, &row_sense, 
                      row_start, col_idx, value, NULL, NULL);
  
  status = CPXcopyquad( env, lp, qstart , qcnt, Q_ind.direct_access() , Q.direct_access() );

#ifdef USE_XPRESS

  int nReturn;
  XPRSprob xp_prob;
  
  nReturn=XPRSinit("/opt/xpressmp/");
  
  if (nReturn != 0) {
    
    char msg[512];
    XPRSgetlicerrmsg(msg,512);
    
    std::cerr << "error message: " << msg << std::endl;
  }
  
  int* start = new int[nLabels+1];
  int* row = new int[nLabels+1];
  
  for (uint v=0; v <= nLabels; v++) {
    start[v] = v;
    row[v] = 0;
  }
  
  uint nQEntries = 0;
  int* qfirst = new int[Q.size()];
  int* qsecond = new int[Q.size()];
  double* qval = new double[Q.size()];
  
  for (uint l1=0; l1 < nLabels; l1++) {
    for (uint l2=0; l2 <= l1; l2++) {
	  
      qfirst[nQEntries] = l1;
      qsecond[nQEntries] = l2;
      qval[nQEntries] = /*0.5 * */ Q(l1,l2);
      
      nQEntries++;
    }
  }

#endif  

  for (uint iter = 1; iter <= 1000; iter++) {

    std::cerr << "*********** iteration " << iter << std::endl;

    if (iter == 1) {
      /** 1.) compute marginals **/

      for (uint y=0; y < yDim; y++) {
	
        for (uint x=0; x < xDim; x++) {
	  
          //update h-marginals
          for (uint lh = 0; lh < nHorLabels; lh++) {
	    
            double sum = 0.0;
            for (uint lv = 0; lv < nVertLabels; lv++)
              sum += var(x,y,lv*nHorLabels + lh);
	    
            if (lh > 0)
              sum += hmarginal(x,y,lh-1);
	  
            assert(sum >= 0.0);
            assert(sum <= 1.001);
	    
            if (iter > 1)
              assert(fabs(sum - hmarginal(x,y,lh)) < 1e-3);
	    
            hmarginal(x,y,lh) = sum;
          }
	
#if 1
          //update v-marginals
          for (uint lv = 0; lv < nVertLabels; lv++) {
	  
            double sum  = 0.0;
            for (uint lh = 0; lh < nHorLabels; lh++) 
              sum += var(x,y,lv*nHorLabels + lh);
	  
            if (lv > 0)
              sum += vmarginal(x,y,lv-1);
	    
            assert(sum >= 0.0);
            assert(sum <= 1.001);
	  
            if (iter > 1)
              assert(fabs(sum - vmarginal(x,y,lv)) < 1e-3);
	    
            vmarginal(x,y,lv) = sum;
          }
#endif
        }
      }
    }

    energy = 0.0;

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        for (uint h=0; h < nHorLabels; h++) {

          const double cur_marginal = hmarginal(x,y,h);

          if (x > 0) {
            double temp = cur_marginal - hmarginal(x-1,y,h);
            energy += temp*temp;
          }
          if (x+1 < xDim) {
            double temp = cur_marginal - hmarginal(x+1,y,h);
            energy += temp*temp;
          }
          if (y > 0) {
            double temp = cur_marginal - hmarginal(x,y-1,h);
            energy += temp*temp;
          }
          if (y+1 < yDim) {
            double temp = cur_marginal - hmarginal(x,y+1,h);
            energy += temp*temp;
          }
        }

#if 1
        for (uint v=0; v < nVertLabels; v++) {

          const double cur_marginal = vmarginal(x,y,v);

          if (x > 0) {
            double temp = cur_marginal - vmarginal(x-1,y,v);
            energy += temp*temp;
          }
          if (x+1 < xDim) {
            double temp = cur_marginal - vmarginal(x+1,y,v);
            energy += temp*temp;
          }
          if (y > 0) {
            double temp = cur_marginal - vmarginal(x,y-1,v);
            energy += temp*temp;
          }
          if (y+1 < yDim) {
            double temp = cur_marginal - vmarginal(x,y+1,v);
            energy += temp*temp;
          }
        }
#endif

      }
    }

    energy *= lambda;

    //std::cerr << "intermediate energy: " << energy << std::endl;

    for (uint i=0; i < var.size(); i++)
      energy += var.direct_access(i) * label_cost.direct_access(i);

    std::cerr << "energy: " << energy << std::endl;

    /** 2.) perform a step of block coordinate gradient descent **/

    const uint nPixels = xDim*yDim;

    // for (uint y=0; y < yDim; y++) {
    //   for (uint x=0; x < xDim; x++) {

    for (uint p=0; p < nPixels; p++) {
      
      uint x,y;

      if ((iter % 2) == 1) {
        x = p % xDim;
        y = p / xDim;
      }
      else {
        x = p / yDim;
        y = p % yDim;
      }
    
      double nNeighbors = 0.0;
      if (x > 0)
        nNeighbors++;
      if (x+1 < xDim)
        nNeighbors++;
      
      if (y > 0)
        nNeighbors++;
      if (y+1 < yDim)
        nNeighbors++;
      
      for (uint l=0; l < nLabels; l++) {
        lin_cost[l] = label_cost(x,y,l) / nNeighbors;
      }
      
      for (uint h=0; h < nHorLabels; h++) {

        double val = 0.0;
	
        if (x > 0)
          val += 2.0 * hmarginal(x-1,y,h);
        if (x+1 < xDim)
          val += 2.0 * hmarginal(x+1,y,h);
        if (y > 0)
          val += 2.0 * hmarginal(x,y-1,h);
        if (y+1 < yDim)
          val += 2.0 * hmarginal(x,y+1,h);
	
        for (uint hh=0; hh <= h; hh++)
          for (uint v=0; v < nVertLabels; v++)
            lin_cost[v*nHorLabels+hh] -= lambda * val / nNeighbors;
      }
      
#if 1
      for (uint v=0; v < nVertLabels; v++) {
	
        double val = 0.0;
	
        if (x > 0)
          val += 2.0 * vmarginal(x-1,y,v);
        if (x+1 < xDim)
          val += 2.0 * vmarginal(x+1,y,v);
        if (y > 0)
          val += 2.0 * vmarginal(x,y-1,v);
        if (y+1 < yDim)
          val += 2.0 * vmarginal(x,y+1,v);	  
	
        for (uint vv=0; vv <= v; vv++) {
          for (uint h=0; h < nHorLabels; h++)
            lin_cost[vv*nHorLabels + h] -= lambda * val / nNeighbors;
        }
      }
#endif


#ifdef USE_XPRESS

      nReturn=XPRScreateprob(&xp_prob);
      
      nReturn=XPRSsetcbmessage(xp_prob,optimizermsg,NULL);

      XPRSsetdblcontrol(xp_prob,XPRS_BARGAPSTOP,1e-5);
      //XPRSsetintcontrol(xp_prob,XPRS_PRESOLVE,0);
      XPRSsetintcontrol(xp_prob,XPRS_IFCHECKCONVEXITY,0);      

      nReturn = XPRSloadqp(xp_prob, "motion-qp", nLabels, 1, &row_sense, &rhs, NULL,
                           lin_cost.direct_access(), start, NULL, row, value, 
                           var_lb.direct_access(), var_ub.direct_access(), 
                           nQEntries, qfirst, qsecond, qval );

      for (uint v=0; v < nLabels; v++) {
      	int idx = v;
      	XPRSchgobj(xp_prob, 1,  &idx, lin_cost.direct_access() + v);      
      }

      XPRSminim(xp_prob,"d");

      XPRSgetlpsol(xp_prob, sol, NULL, NULL, NULL);
	
      nReturn=XPRSdestroyprob(xp_prob);
#else
      //CPXENVptr     env = NULL;
      //CPXLPptr      lp = NULL;
      //int status = 0;
      
      /* Initialize the CPLEX environment */
      
      //env = CPXopenCPLEX (&status);
	
      /* If an error occurs, the status value indicates the reason for
         failure.  A call to CPXgeterrorstring will produce the text of
         the error message.  Note that CPXopenCPLEX produces no output,
         so the only way to see the cause of the error is to use
         CPXgeterrorstring.  For other CPLEX routines, the errors will
         be seen if the CPX_PARAM_SCRIND indicator is set to CPX_ON.  */
	
      if ( env == NULL ) {
        char  errmsg[1024];
        fprintf (stderr, "Could not open CPLEX environment.\n");
        CPXgeterrorstring (env, status, errmsg);
        fprintf (stderr, "%s", errmsg);
        exit(1);
      }
      
      //set problem data      
      //lp = CPXcreateprob (env, &status, "motion-qp");
      
      // set linear cost
      // status = CPXnewcols (env, lp, nLabels, lin_cost.direct_access(), var_lb.direct_access(), 
      // 			   var_ub.direct_access(), NULL, NULL);

      for (uint v=0; v < nLabels; v++)
        CPXchgcoef(env, lp,  -1, v, lin_cost[v]);
      
      //set constraint
      //double rhs = 1.0;
      //char row_sense = 'E';
      
      // status = CPXaddrows(env, lp, 0, 1, nLabels, &rhs, &row_sense, 
      // 			  row_start, col_idx, value, NULL, NULL);
      

      if (status)
        exit(1);
      
      //set quadratic cost
      //status = CPXcopyquad( env, lp, qstart , qcnt, Q_ind.direct_access() , Q.direct_access() );

      if (status)
        exit(1);
      
      //optimize
      status = CPXqpopt(env, lp);

      int solstat = CPXgetstat (env, lp);

      if (solstat == 6)
        std::cerr << "WARNING: numerical instability" << std::endl;
      else if (solstat != 1)
        std::cerr << "solution status " << solstat << std::endl;
      
      assert(solstat == 1 || solstat == 6);
      
      if (status)
        exit(1);
      
      double objval;
      status = CPXgetobjval (env, lp, &objval);
      assert(status == 0);
      
      status = CPXgetx (env, lp, sol, 0, nLabels-1);

      if (status)
        exit(1);
#endif      

      //double sum_sol = 0.0;
      
      for (uint l=0; l < nLabels; l++) {
        var(x,y,l) = sol[l];
        //sum_sol += sol[l];
      }

      
      // double real_energy = 0.0;
      // for (uint l=0; l < nLabels; l++) {
      //   real_energy += sol[l]*lin_cost[l];	  	  
      // }
      // std::cerr << "0. real lin energy: " << real_energy << std::endl;
      
      // for (uint l1=0; l1 < nLabels; l1++) {
      //   for (uint l2=0; l2 < nLabels; l2++) {
      //     real_energy += 0.5*sol[l1]*sol[l2]*Q(l1,l2);
      //   }
      // }
      
      // std::cerr << "1. objective value: " << objval << std::endl;
      // std::cerr << "2. real energy: " << real_energy << std::endl;
      
      //assert(sum_sol >= 0.99 && sum_sol < 1.01);
      // update marginals
      double hsum = 0.0;
      for (uint h=0; h < nHorLabels; h++) {
	
        for (uint v=0; v < nVertLabels; v++)
          hsum += var(x,y,v*nHorLabels+h);
	
        hmarginal(x,y,h) = hsum;
      } 
      
#if 1
      double vsum = 0.0;
      for (uint v=0; v < nVertLabels; v++) {
	
        for (uint h=0; h < nHorLabels; h++)
          vsum += var(x,y,v*nHorLabels+h);
	
        vmarginal(x,y,v) = vsum;
      } 
#endif
      
      //CPXfreeprob (env, &lp);
      //CPXcloseCPLEX(&env);
    }
  }

  CPXfreeprob (env, &lp);
  CPXcloseCPLEX(&env);
  
  //extract flow field

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      double max_var = -1.0;
      double sum = 0.0;

      int arg_max_x = MAX_UINT;
      int arg_max_y = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          double val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) + min_x_disp;
            arg_max_y = ((int) ly) + min_y_disp;
          }
        }
      }

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

#ifdef USE_XPRESS

  nReturn=XPRSfree();

  delete[] qfirst;
  delete[] qsecond;
  delete[] qval;
  delete[] start;
  delete[] row;
  
#endif

  delete[] qstart;
  delete[] qcnt;
  delete[] col_idx;
  delete[] value;
  delete[] sol;

  return energy;
}
#endif //HAS_CPLEX


double motion_estimation_convprog_nesterov(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                           int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                           uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                           double exponent) {

  Real exponent_m1 = exponent - ((Real) 1.0);

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint nVars = xDim*yDim*nLabels;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  
  float inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= inv_spacing;
  const Real diag_lambda = lambda / sqrt(2.0);

  const Real inv_sqrt2 = 1.0 / sqrt(2.0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

#if 0
  /***** variable exclusion stage *****/
  Math2D::Matrix<Real>  smoothWorstCost(nHorLabels, nVertLabels,0.0);

  for (uint h=0; h < nHorLabels; h++) {
    for (uint v=0; v < nVertLabels; v++) {
      
      Real dist = std::max(h, nHorLabels-1-h) + std::max(v, nVertLabels-1-v);

      if (neighborhood == 4)
        dist *= 4.0 * lambda;
      else {
        assert(neighborhood == 8);

        dist *= 4.0 * (1.0 + sqrt(0.5)) * lambda;
      }

      smoothWorstCost(h,v) = dist;
    }
  }

  uint nVarExcluded = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real best_label_cost = 1e300;
      uint arg_min = MAX_UINT;

      for (uint l=0; l < nLabels; l++) {

        Real hyp_cost = label_cost(x,y,l);
        if (hyp_cost < best_label_cost) {
          best_label_cost = hyp_cost;
          arg_min = l;
        }
      }

      uint lx_best = arg_min % nHorLabels;
      uint ly_best = arg_min / nHorLabels;

      best_label_cost += smoothWorstCost(lx_best,ly_best);

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          Real hyp_cost = label_cost(x,y,ly*nHorLabels+lx);
	
          if (hyp_cost > best_label_cost) { // + smoothWorstCost(lx,ly)) {

            label_cost(x,y,ly*nHorLabels+lx) += 100.0;
            nVarExcluded++;
          }
        }
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars." << std::endl;
#endif

  Math3D::NamedTensor<Real> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

  /*** initialization ***/
#if 1
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      Real min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

  double energy;

  if (true) {

    const float* const_label_cost = label_cost.direct_access();

    Math3D::NamedTensor<Real> aux_var(xDim,yDim,nLabels, MAKENAME(aux_var));

    //NOTE: by construction hmarginal(.,.,nHorLabels-1) would always be 1
    Math3D::NamedTensor<Real> hmarginal(xDim,yDim,nHorLabels-1, 0.0, MAKENAME(hmarginal));
    //NOTE: by construction vmarginal(.,.,nVertLabels-1) would always be 1
    Math3D::NamedTensor<Real> vmarginal(xDim,yDim,nVertLabels-1, 0.0, MAKENAME(vmarginal));  

#ifdef USE_EXPLICIT_GRADIENT
    Math3D::NamedTensor<Real> grad(xDim,yDim,nLabels,MAKENAME(grad));
#endif

    uint nOuterIter = 1;
    double org_exponent = exponent;
    double increment = 0.05; //0.1;

#if 0
    while (exponent < 2.0) {

      exponent += increment;
      nOuterIter++;
    }
#endif
    exponent += increment;

    Real alpha = 0.000125 / lambda;  
    
    for (uint outer_iter = 1; outer_iter <= nOuterIter; outer_iter++) {

      exponent -= increment;
      exponent_m1 = exponent - ((Real) 1.0);

      std::cerr << "################ exponent " << exponent << std::endl;
      
      aux_var = var;
      
      Real prev_t = 1.0;
            
      Real last_energy = 1e50;
      Real best_energy = last_energy;
      
      uint iter_since_restart = 0;

      uint restart_threshold = (outer_iter == 1) ? 10 : 5;
      
      uint nInnerIter = (outer_iter == nOuterIter) ? 45000 : 50 + (outer_iter-1)*5;

      Real save_energy = 1e50;
      const Real cutoff = 1e-6;

      uint iter = 1;
      for (; iter <= nInnerIter; iter++) { 

        std::cerr << "*********** iteration " << iter << std::endl;

        const Real* const_var = var.direct_access();

        /** 1.) energy computation **/
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum_var = 0.0;

            const uint hmbase = (y*xDim+x)*(nHorLabels-1);
            const uint vmbase = (y*xDim+x)*(nVertLabels-1);
            const uint vbase = (y*xDim+x)*nLabels;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum_var += var(x,y,lv*nHorLabels + lh);
                sum_var += const_var[vbase + lv*nHorLabels + lh];
              }
	  
              //hmarginal(x,y,lh) = sum_var;
              hmarginal.direct_access(hmbase + lh) = sum_var;
            }
	
            sum_var = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum_var += var(x,y,label_offs + lh);
                sum_var += const_var[vbase + label_offs + lh];
              }

              //vmarginal(x,y,lv) = sum_var;
              vmarginal.direct_access(vmbase+lv) = sum_var;
            }
          }
        }
      
        energy = (Real) 0.0;
      
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real curvar_marginal = hmarginal(x,y,h);

              if (x > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x-1,y,h);

                energy += std::pow(fabs(temp2),exponent);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x,y-1,h);

                energy += std::pow(fabs(temp2),exponent);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x-1,y-1,h);

                  energy += inv_sqrt2*std::pow(fabs(temp2),exponent);
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x+1,y-1,h);

                  energy += inv_sqrt2*std::pow(fabs(temp2),exponent);
                }
              }
            }

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real curvar_marginal = vmarginal(x,y,v);

              if (x > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x-1,y,v);

                energy += std::pow(fabs(temp2),exponent);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x,y-1,v);

                energy += std::pow(fabs(temp2),exponent);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x-1,y-1,v);

                  energy += inv_sqrt2*std::pow(fabs(temp2),exponent);
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x+1,y-1,v);

                  energy += inv_sqrt2*std::pow(fabs(temp2),exponent);
                }
              }
            }
          }
        }

        energy *= lambda;

        //std::cerr << "intermediate energy: " << energy << std::endl;
  
        for (uint i=0; i < nVars; i++) {
          //energy += const_var[i] * label_cost.direct_access(i);
          energy += const_var[i] * const_label_cost[i];
        }

        std::cerr.precision(10);
        std::cerr << "energy: " << energy << std::endl;

        if ((iter % 15) == 0) {

          if (iter >= 45 && fabs(save_energy - energy) < cutoff) {

            std::cerr << "OSCILLATION OR (SLOW) CONVERGENCE DETECTED -> CUTOFF)" << std::endl;
            break;
          }

          save_energy = energy;
        }

        if  ((energy > 1.15*best_energy || (energy > last_energy && iter_since_restart >= restart_threshold)  ) 
             ) {

          //std::cerr << "BREAK because of energy increase" << std::endl;
          //break;
          iter_since_restart = 1;

          aux_var = var; 
          prev_t = 1.0;
          if (energy > last_energy) {
            alpha *= 0.65;
          }

          std::cerr << "RESTART because of energy increase, new alpha " << alpha << std::endl;
        }
        else
          iter_since_restart++;

        /** 2.) gradient computation **/
        const Real* const_aux_var = aux_var.direct_access();

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum = 0.0;

            const uint hmbase = (y*xDim+x)*(nHorLabels-1);
            const uint vmbase = (y*xDim+x)*(nVertLabels-1);
            const uint vbase = (y*xDim+x)*nLabels;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum += aux_var(x,y,lv*nHorLabels + lh);
                sum += const_aux_var[vbase + lv*nHorLabels + lh];
              }
	  
              //hmarginal(x,y,lh) = sum;
              hmarginal.direct_access(hmbase + lh) = sum;
            }
	
            sum = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum += aux_var(x,y,label_offs + lh);
                sum += const_aux_var[vbase + label_offs + lh];
              }
	  
              //vmarginal(x,y,lv) = sum;
              vmarginal.direct_access(vmbase + lv) = sum;
            }
          }
        }

#ifdef USE_EXPLICIT_GRADIENT
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          grad.direct_access(v) = const_label_cost[v];
#else
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          aux_var.direct_access(v) -= alpha * const_label_cost[v];
#endif

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real hgrad = (Real) 0.0;

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real cur_marginal = hmarginal(x,y,h);

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - hmarginal(x-1,y,h);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - hmarginal(x+1,y,h);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (y > 0) {
                const Real temp = cur_marginal - hmarginal(x,y-1,h);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - hmarginal(x,y+1,h);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }

              diff *= exponent * lambda;

              hgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x-1,y-1,h);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x+1,y-1,h);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x > 0 && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x-1,y+1,h);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x+1 < xDim && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x+1,y+1,h);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }

                hgrad += diff * exponent * diag_lambda;
              }

              for (uint v=0; v < nVertLabels; v++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, v*nHorLabels + h) += hgrad;
#else
                aux_var(x,y, v*nHorLabels + h) -= alpha* hgrad;
#endif
              }
            }

            Real vgrad = 0.0;

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real cur_marginal = vmarginal(x,y,v);

              const uint label_offs = v*nHorLabels;

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - vmarginal(x-1,y,v);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - vmarginal(x+1,y,v);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (y > 0) {
                const Real temp = cur_marginal - vmarginal(x,y-1,v);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - vmarginal(x,y+1,v);

                diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
              }

              diff *= exponent * lambda;
              vgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x-1,y-1,v);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x+1,y-1,v);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x > 0 && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x-1,y+1,v);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
                if (x+1 < xDim && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x+1,y+1,v);

                  diff += sign(temp)*std::pow(fabs(temp),exponent_m1);
                }
	  
                vgrad += diff * exponent * diag_lambda;
              }

              for (uint h=0; h < nHorLabels; h++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, label_offs + h) += vgrad;
#else
                aux_var(x,y, label_offs + h) -= alpha * vgrad;
#endif
              }
            }
          }
        }

        last_energy = energy;
        best_energy = std::min(best_energy,last_energy);

        /** 3.) perform a step of gradient descent **/

#ifdef USE_EXPLICIT_GRADIENT
        grad *= alpha;
        aux_var -= grad;
#endif

        /** 4.) reprojection [Michelot 1986] **/
    
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            projection_on_simplex(aux_var.direct_access() + (y*xDim+x)*nLabels, nLabels);
          }
        }

        const Real new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
        const Real nesterov_fac = (prev_t - 1) / new_t;
        //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
	  
        for (uint i=0; i < aux_var.size(); i++) {
      
          const Real old_aux = aux_var.direct_access(i);
          aux_var.direct_access(i) = old_aux + nesterov_fac*(old_aux - var.direct_access(i)) ;
          var.direct_access(i) = old_aux;
        }

        prev_t = new_t;
      }  
    }
  } //end of outer loop

  Math2D::Matrix<uint> labeling(xDim,yDim);  
  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real max_var = -1.0;
      Real sum = 0.0;

      float arg_max_x = MAX_UINT;
      float arg_max_y = MAX_UINT;
      
      uint arg_max = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          Real val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) * inv_spacing + min_x_disp;
            arg_max_y = ((int) ly) * inv_spacing + min_y_disp;
            arg_max = ly*nHorLabels + lx;
          }
        }
      }

      labeling(x,y) = arg_max;

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);

  return energy;
}

/******************************************************************************************************************/

inline double psi(double x, double epsilon) {

  double ax = fabs(x);
  if (ax < epsilon)
    return 0.5*ax*ax/epsilon;
  else
    return ax - 0.5*epsilon;
}

inline double psi_prime(double x, double epsilon) {

  double ax = fabs(x);
  if (ax >= epsilon)
    return sign(x);
  else
    return x / epsilon;
}

double motion_estimation_convprog_nesterov_smoothapprox(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                                        int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                                        uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                                        double epsilon, bool use_cuda) {

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint nVars = xDim*yDim*nLabels;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));

  const uint zero_label = nHorLabels* (  (-min_y_disp + 1)*spacing - (spacing-1)    )
    + (-min_x_disp + 1)*spacing - (spacing-1);
  Math2D::Matrix<uint> labeling(xDim,yDim,zero_label);
  Math2D::Matrix<uint> temp_labeling(xDim,yDim,zero_label);  

  double best_discrete_energy = 1e300;

  float inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= 1.0 / psi(1.0,epsilon); //to get exact costs for integral points

  lambda *= inv_spacing;
  const Real diag_lambda = lambda / sqrt(2.0);

  const Real inv_sqrt2 = 1.0 / sqrt(2.0);

  //correct for scaled version
  //Real lipschitz_alpha = epsilon / (lambda * sqrt(2.0*nLabels*neighborhood));

  //correct for unscaled version??
  Real lipschitz_alpha = epsilon / (lambda * 2.0*nLabels*sqrt(neighborhood));

  if (neighborhood == 8)
    lipschitz_alpha *= inv_sqrt2;

  //Real lipschitz_alpha = epsilon / (lambda * (2.0*nLabels*neighborhood));
  //Real alpha = 0.05 * epsilon / (lambda);

  //Real alpha = 0.98*lipschitz_alpha;
  Real alpha = 0.5*lipschitz_alpha;

  //Real alpha = lipschitz_alpha;

  std::cerr.precision(12);
  std::cerr << "lipschitz-alpha: " << lipschitz_alpha << std::endl;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  Math3D::NamedTensor<Real> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

#if 1
  /***** variable exclusion stage *****/
  Math2D::Matrix<Real>  smoothWorstCost(nHorLabels, nVertLabels,0.0);

  for (uint h=0; h < nHorLabels; h++) {
    for (uint v=0; v < nVertLabels; v++) {
      
      Real dist = std::max(h, nHorLabels-1-h) + std::max(v, nVertLabels-1-v);

      if (neighborhood == 4)
        dist *= 4.0 * lambda;
      else {
        assert(neighborhood == 8);

        dist *= 4.0 * (1.0 + sqrt(0.5)) * lambda;
      }

      smoothWorstCost(h,v) = dist;
    }
  }

  uint nVarExcluded = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real best_label_cost = 1e300;
      uint arg_min = MAX_UINT;
      
      for (uint l=0; l < nLabels; l++) {

        Real hyp_cost = label_cost(x,y,l);
        if (hyp_cost < best_label_cost) {
          best_label_cost = hyp_cost;
          arg_min = l;
        }
      }

      uint lx_best = arg_min % nHorLabels;
      uint ly_best = arg_min / nHorLabels;

      best_label_cost += smoothWorstCost(lx_best,ly_best);

      uint nCurExcluded = 0;

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          Real hyp_cost = label_cost(x,y,ly*nHorLabels+lx);
	
          if (hyp_cost > best_label_cost) { // + smoothWorstCost(lx,ly)) {

            label_cost(x,y,ly*nHorLabels+lx) += 100.0;

            nVarExcluded++;

            nCurExcluded++;
            var(x,y,ly*nHorLabels+lx) = 0.0;
          }
        }
      }

      double fill = 1.0 / (nLabels - nCurExcluded);
      for (uint i=0; i < nLabels; i++) {
        if (var(x,y,i) > 0.0)
          var(x,y,i) = fill;
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars. That's " <<  (((double) nVarExcluded) / ((double) nVars)) << "%." << std::endl;
#endif


  /*** initialization ***/
#if 0
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      Real min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

#if 0
  //initialization by expansion moves

  std::cerr << "-------- initializing by expansion moves" << std::endl;
  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling, 3);
  
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++)
    for (uint x=0; x < xDim; x++)
      var(x,y,labeling(x,y)) = 1.0;

  //alpha *= 0.02;
#endif


#ifndef USE_CUDA
  use_cuda = false;
#endif

  uint iter = 1;

  Real energy = 0.0;

  if (use_cuda) {

    std::cerr << "--starting CUDA" << std::endl;
#ifdef USE_CUDA
    std::cerr << "calling..." << std::endl;
    energy = cuda_convmotion_nesterov(first.direct_access(), second.direct_access(), xDim, yDim, nChannels,
                                      min_x_disp, max_x_disp, min_y_disp, max_y_disp, spacing,
                                      epsilon, neighborhood, lambda*spacing, alpha, var.direct_access());
#endif
  }
  else {

    const float* const_label_cost = label_cost.direct_access();

    Math3D::NamedTensor<Real> aux_var(xDim,yDim,nLabels, MAKENAME(aux_var));

    //NOTE: by construction hmarginal(.,.,nHorLabels-1) would always be 1
    Math3D::NamedTensor<Real> hmarginal(xDim,yDim,nHorLabels-1, 0.0, MAKENAME(hmarginal));
    //NOTE: by construction vmarginal(.,.,nVertLabels-1) would always be 1
    Math3D::NamedTensor<Real> vmarginal(xDim,yDim,nVertLabels-1, 0.0, MAKENAME(vmarginal));  

#ifdef USE_EXPLICIT_GRADIENT
    Math3D::NamedTensor<Real> grad(xDim,yDim,nLabels,MAKENAME(grad));
#endif

    double best_lower_bound = -MAX_DOUBLE;

    uint nSubsequentLowerBoundDecreases = 0;

    uint nOuterIter = 1;

    Math1D::NamedVector<Real> hnorm(nHorLabels-1, 0.0, MAKENAME(hnorm));
    Math1D::NamedVector<Real> vnorm(nVertLabels-1, 0.0, MAKENAME(vnorm));  

    //double sum_lipschitz_norms = 0.0;
    for (uint h=0; h < nHorLabels-1; h++) {
      double temp = sqrt(2*(h+1)*nVertLabels);
      //sum_lipschitz_norms += temp;
      hnorm[h] = temp;
    }
    for (uint v=0; v < nVertLabels-1; v++) {
      double temp = sqrt(2*(v+1)*nHorLabels);
      //sum_lipschitz_norms += temp;
      vnorm[v] = temp;
    }

    //DEBUG
    hnorm.set_constant(1.0);
    vnorm.set_constant(1.0);
    //END_DEBUG
    
    for (uint outer_iter = 1; outer_iter <= 1; outer_iter++) {

      std::cerr << "################ epsilon " << epsilon << std::endl;
      
      aux_var = var;
      
      Real prev_t = 1.0;
      
      const Real neg_threshold = 1e-12;
      
      Real last_energy = 1e50;
      Real best_energy = last_energy;
      
      uint iter_since_restart = 0;

      uint restart_threshold = 8;
      
      //uint nInnerIter = (outer_iter == nOuterIter) ? 45000 : 50 + (outer_iter-1)*5;

      uint nInnerIter = 500;

      Real save_energy = 1e50;
      const Real cutoff = 1e-6;

      iter = 1;
      for (; iter <= nInnerIter; iter++) { 

        std::cerr << "*********** iteration " << iter << std::endl;

        const Real* const_var = var.direct_access();

        /** 1.) energy computation **/
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum_var = 0.0;

            const uint hmbase = (y*xDim+x)*(nHorLabels-1);
            const uint vmbase = (y*xDim+x)*(nVertLabels-1);
            const uint vbase = (y*xDim+x)*nLabels;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum_var += var(x,y,lv*nHorLabels + lh);
                sum_var += const_var[vbase + lv*nHorLabels + lh];
              }
	  
              //hmarginal(x,y,lh) = sum_var;
              hmarginal.direct_access(hmbase + lh) = sum_var;
            }
	
            sum_var = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum_var += var(x,y,label_offs + lh);
                sum_var += const_var[vbase + label_offs + lh];
              }

              //vmarginal(x,y,lv) = sum_var;
              vmarginal.direct_access(vmbase+lv) = sum_var;
            }
          }
        }
      
        energy = (Real) 0.0;
      
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real curvar_marginal = hmarginal(x,y,h);

              if (x > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x-1,y,h);

                energy += hnorm[h] * psi(temp2/hnorm[h],epsilon);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x,y-1,h);

                energy += hnorm[h] * psi(temp2/hnorm[h],epsilon);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x-1,y-1,h);

                  energy += hnorm[h] * inv_sqrt2*psi(temp2/hnorm[h],epsilon);
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x+1,y-1,h);

                  energy += hnorm[h] * inv_sqrt2*psi(temp2/hnorm[h],epsilon);
                }
              }
            }

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real curvar_marginal = vmarginal(x,y,v);

              if (x > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x-1,y,v);

                energy += vnorm[v] * psi(temp2/vnorm[v],epsilon);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x,y-1,v);

                energy += vnorm[v] * psi(temp2/vnorm[v],epsilon);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x-1,y-1,v);

                  energy += vnorm[v] * inv_sqrt2*psi(temp2/vnorm[v],epsilon);
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x+1,y-1,v);

                  energy += vnorm[v] * inv_sqrt2*psi(temp2/vnorm[v],epsilon);
                }
              }
            }
          }
        }

        energy *= lambda;

        //std::cerr << "intermediate energy: " << energy << std::endl;
  
        for (uint i=0; i < nVars; i++) {
          //energy += const_var[i] * label_cost.direct_access(i);
          energy += const_var[i] * const_label_cost[i];
        }


        if ((iter % 10) == 0) {

          //compute lower bound
          Math1D::Vector<Real> cur_grad(nLabels);

          double lower_bound = energy;

          for (uint y=0; y < yDim; y++) {
            for (uint x=0; x < xDim; x++) {

              for (uint i=0; i < nLabels; i++) {
                cur_grad[i] = const_label_cost[(y*xDim+x)*nLabels+i];
              }


              Real hgrad = (Real) 0.0;

              for (int h=nHorLabels-2; h >= 0; h--) {
		
                const Real cur_marginal = hmarginal(x,y,h);
		
                Real diff = (Real) 0.0;
                if (x > 0) {
                  const Real temp = cur_marginal - hmarginal(x-1,y,h);
		  
                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (x+1 < xDim) {
                  const Real temp = cur_marginal - hmarginal(x+1,y,h);
		  
                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (y > 0) {
                  const Real temp = cur_marginal - hmarginal(x,y-1,h);
		  
                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (y+1 < yDim) {
                  const Real temp = cur_marginal - hmarginal(x,y+1,h);
		  
                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
		
                diff *= lambda;
		
                hgrad += diff;
		
                if (neighborhood >= 8) {
		  
                  diff = (Real) 0.0;
		  
                  if (x > 0 && y > 0) {
		    
                    const Real temp = cur_marginal - hmarginal(x-1,y-1,h);

                    diff += psi_prime(temp/hnorm[h],epsilon);
                  }
                  if (x+1 < xDim && y > 0) {
		    
                    const Real temp = cur_marginal - hmarginal(x+1,y-1,h);
		    
                    diff += psi_prime(temp/hnorm[h],epsilon);
                  }
                  if (x > 0 && y+1 < yDim) {
		    
                    const Real temp = cur_marginal - hmarginal(x-1,y+1,h);
		    
                    diff += psi_prime(temp/hnorm[h],epsilon);
                  }
                  if (x+1 < xDim && y+1 < yDim) {
		    
                    const Real temp = cur_marginal - hmarginal(x+1,y+1,h);
		    
                    diff += psi_prime(temp/hnorm[h],epsilon);
                  }
		  
                  hgrad += diff * diag_lambda;
                }
		
                for (uint v=0; v < nVertLabels; v++) {
                  cur_grad[v*nHorLabels + h] += hgrad;
                }
              }
	      
              Real vgrad = 0.0;

              for (int v=nVertLabels-2; v>= 0; v--) {
		
                const Real cur_marginal = vmarginal(x,y,v);
		
                const uint label_offs = v*nHorLabels;
		
                Real diff = (Real) 0.0;
                if (x > 0) {
                  const Real temp = cur_marginal - vmarginal(x-1,y,v);
		  
                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (x+1 < xDim) {
                  const Real temp = cur_marginal - vmarginal(x+1,y,v);
		  
                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (y > 0) {
                  const Real temp = cur_marginal - vmarginal(x,y-1,v);
		  
                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x,y+1,v);
		  
                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
		
                diff *= lambda;
                vgrad += diff;
		
                if (neighborhood >= 8) {
		  
                  diff = (Real) 0.0;
		  
                  if (x > 0 && y > 0) {
                    const Real temp = cur_marginal - vmarginal(x-1,y-1,v);
		    
                    diff += psi_prime(temp/vnorm[v],epsilon);
                  }
                  if (x+1 < xDim && y > 0) {
                    const Real temp = cur_marginal - vmarginal(x+1,y-1,v);
		    
                    diff += psi_prime(temp/vnorm[v],epsilon);
                  }
                  if (x > 0 && y+1 < yDim) {
                    const Real temp = cur_marginal - vmarginal(x-1,y+1,v);
		    
                    diff += psi_prime(temp/vnorm[v],epsilon);
                  }
                  if (x+1 < xDim && y+1 < yDim) {
                    const Real temp = cur_marginal - vmarginal(x+1,y+1,v);
		    
                    diff += psi_prime(temp/vnorm[v],epsilon);
                  }
		  
                  vgrad += diff * diag_lambda;
                }

                for (uint h=0; h < nHorLabels; h++) {
                  cur_grad[label_offs + h] += vgrad;
                }
              }

              for (uint i=0; i < cur_grad.size(); i++)
                lower_bound -= cur_grad.direct_access(i) * var.direct_access((y*xDim+x)*nLabels+i);
	      
              double cur_min = 1e300;
              for (uint i=0; i < cur_grad.size(); i++) {
                if (cur_grad[i] < cur_min) {
                  cur_min = cur_grad[i];
                }
              }

              lower_bound += cur_min;	      
            }
          }
          if (lower_bound >= best_lower_bound) {
            best_lower_bound = lower_bound;
            nSubsequentLowerBoundDecreases = 0;
          }
          else
            nSubsequentLowerBoundDecreases++;
	  

          for (uint y=0; y < yDim; y++) {
            for (uint x=0; x < xDim; x++) {
	      
              Real max_var = -1.0;
	      
              uint arg_max = MAX_UINT;
	      
              for (uint lx=0; lx < nHorLabels; lx++) {
                for (uint ly=0; ly < nVertLabels; ly++) {
		  
                  Real val = var(x,y,ly*nHorLabels+lx);
		  
                  if (val > max_var) {
                    max_var = val;
                    arg_max = ly*nHorLabels + lx;
                  }
                }
              }
	      
              temp_labeling(x,y) = arg_max;
            }
          }


          std::cerr << "current lower bound: " << lower_bound << ", best known: " << best_lower_bound << std::endl;	

          double cur_discrete_energy = motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                     neighborhood, temp_labeling);

          std::cerr << "discrete energy: " << cur_discrete_energy << std::endl;
          if (cur_discrete_energy < best_discrete_energy) {
            best_discrete_energy = cur_discrete_energy;
            labeling = temp_labeling;
          }
        }

        std::cerr.precision(10);
        std::cerr << "energy: " << energy << std::endl;

#if 0
        if ((iter % 15) == 0) {

          if (iter >= 45 && fabs(save_energy - energy) < cutoff) {

            std::cerr << "OSCILLATION OR (SLOW) CONVERGENCE DETECTED -> CUTOFF)" << std::endl;
            break;
          }

          save_energy = energy;
        }
#endif

        if  ((energy > 1.15*best_energy || (energy > last_energy && iter_since_restart >= restart_threshold)  ) 
             && alpha > lipschitz_alpha) {

#if 0
          std::cerr << "WARNING: ASCENT" << std::endl;
#else
          //std::cerr << "BREAK because of energy increase" << std::endl;
          //break;
          iter_since_restart = 0;

          aux_var = var; 
          prev_t = 1.0;
          if (energy > last_energy) {
            alpha *= 0.25; //0.1; //0.65;
          }
          std::cerr << "RESTART because of energy increase, new alpha " << alpha << std::endl;
          std::cerr << "lipschitz alpha: " << lipschitz_alpha << std::endl;
#endif
        }

#if 1
        if (nSubsequentLowerBoundDecreases >= 3 && iter_since_restart >= 5000) {
          iter_since_restart = 0;

          aux_var = var; 
          prev_t = 1.0;
          std::cerr << "RESTART because of lower bound decrease, new alpha " << alpha << std::endl;
        }
#endif

        iter_since_restart++;

        /** 2.) gradient computation **/
        const Real* const_aux_var = aux_var.direct_access();

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum = 0.0;

            const uint hmbase = (y*xDim+x)*(nHorLabels-1);
            const uint vmbase = (y*xDim+x)*(nVertLabels-1);
            const uint vbase = (y*xDim+x)*nLabels;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum += aux_var(x,y,lv*nHorLabels + lh);
                sum += const_aux_var[vbase + lv*nHorLabels + lh];
              }
	  
              //hmarginal(x,y,lh) = sum;
              hmarginal.direct_access(hmbase + lh) = sum;
            }
	
            sum = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum += aux_var(x,y,label_offs + lh);
                sum += const_aux_var[vbase + label_offs + lh];
              }
	  
              //vmarginal(x,y,lv) = sum;
              vmarginal.direct_access(vmbase + lv) = sum;
            }
          }
        }

#ifdef USE_EXPLICIT_GRADIENT
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          grad.direct_access(v) = const_label_cost[v];
#else
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          aux_var.direct_access(v) -= alpha * const_label_cost[v];
#endif

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real hgrad = (Real) 0.0;

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real cur_marginal = hmarginal(x,y,h);

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - hmarginal(x-1,y,h);

                diff += psi_prime(temp/hnorm[h],epsilon);
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - hmarginal(x+1,y,h);

                diff += psi_prime(temp/hnorm[h],epsilon);
              }
              if (y > 0) {
                const Real temp = cur_marginal - hmarginal(x,y-1,h);

                diff += psi_prime(temp/hnorm[h],epsilon);
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - hmarginal(x,y+1,h);

                diff += psi_prime(temp/hnorm[h],epsilon);
              }

              diff *= lambda;

              hgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x-1,y-1,h);

                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x+1,y-1,h);

                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (x > 0 && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x-1,y+1,h);

                  diff += psi_prime(temp/hnorm[h],epsilon);
                }
                if (x+1 < xDim && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x+1,y+1,h);

                  diff += psi_prime(temp/hnorm[h],epsilon);
                }

                hgrad += diff * diag_lambda;
              }

              for (uint v=0; v < nVertLabels; v++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, v*nHorLabels + h) += hgrad;
#else
                aux_var(x,y, v*nHorLabels + h) -= alpha* hgrad;
#endif
              }
            }

            Real vgrad = 0.0;

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real cur_marginal = vmarginal(x,y,v);

              const uint label_offs = v*nHorLabels;

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - vmarginal(x-1,y,v);

                diff += psi_prime(temp/vnorm[v],epsilon);
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - vmarginal(x+1,y,v);

                diff += psi_prime(temp/vnorm[v],epsilon);
              }
              if (y > 0) {
                const Real temp = cur_marginal - vmarginal(x,y-1,v);

                diff += psi_prime(temp/vnorm[v],epsilon);
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - vmarginal(x,y+1,v);

                diff += psi_prime(temp/vnorm[v],epsilon);
              }

              diff *= lambda;
              vgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x-1,y-1,v);

                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x+1,y-1,v);

                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (x > 0 && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x-1,y+1,v);

                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
                if (x+1 < xDim && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x+1,y+1,v);

                  diff += psi_prime(temp/vnorm[v],epsilon);
                }
	  
                vgrad += diff * diag_lambda;
              }

              for (uint h=0; h < nHorLabels; h++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, label_offs + h) += vgrad;
#else
                aux_var(x,y, label_offs + h) -= alpha * vgrad;
#endif
              }
            }
          }
        }

        last_energy = energy;
        best_energy = std::min(best_energy,last_energy);

        /** 3.) perform a step of gradient descent **/

#ifdef USE_EXPLICIT_GRADIENT

        grad *= alpha;
        aux_var -= grad;
#endif

        /** 4.) reprojection [Michelot 1986] **/
    
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            projection_on_simplex(aux_var.direct_access() + (y*xDim+x)*nLabels, nLabels);
          }
        }

        const Real new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
        const Real nesterov_fac = (prev_t - 1) / new_t;
        //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));	  
	  
        for (uint i=0; i < aux_var.size(); i++) {
      
          const Real old_aux = aux_var.direct_access(i);
          aux_var.direct_access(i) = old_aux + nesterov_fac*(old_aux - var.direct_access(i)) ;
          var.direct_access(i) = old_aux;
        }

        prev_t = new_t;
      }  
    }
  } //end of outer loop

  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real max_var = -1.0;
      Real sum = 0.0;

      float arg_max_x = MAX_UINT;
      float arg_max_y = MAX_UINT;
      
      uint arg_max = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          Real val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) * inv_spacing + min_x_disp;
            arg_max_y = ((int) ly) * inv_spacing + min_y_disp;
            arg_max = ly*nHorLabels + lx;
          }
        }
      }

      labeling(x,y) = arg_max;

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);

  return energy;

}

/******************************************************************************************************************/

double motion_estimation_convprog_standardrelax_nesterov_smoothapprox(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                                                      int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                                                      uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                                                      double epsilon) {
  
  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint nVars = xDim*yDim*nLabels;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));

  const uint zero_label = nHorLabels* (  (-min_y_disp + 1)*spacing - (spacing-1)    )
    + (-min_x_disp + 1)*spacing - (spacing-1);
  Math2D::Matrix<uint> labeling(xDim,yDim,zero_label);
  
  float inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= inv_spacing;
  //const Real diag_lambda = lambda / sqrt(2.0);

  const Real inv_sqrt2 = 1.0 / sqrt(2.0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

#if 0
  /***** variable exclusion stage *****/
  Math2D::Matrix<Real>  smoothWorstCost(nHorLabels, nVertLabels,0.0);

  for (uint h=0; h < nHorLabels; h++) {
    for (uint v=0; v < nVertLabels; v++) {
      
      Real dist = std::max(h, nHorLabels-1-h) + std::max(v, nVertLabels-1-v);

      if (neighborhood == 4)
        dist *= 4.0 * lambda;
      else {
        assert(neighborhood == 8);

        dist *= 4.0 * (1.0 + sqrt(0.5)) * lambda;
      }

      smoothWorstCost(h,v) = dist;
    }
  }

  uint nVarExcluded = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real best_label_cost = 1e300;
      uint arg_max = MAX_UINT;

      for (uint l=0; l < nLabels; l++) {

        Real hyp_cost = label_cost(x,y,l);
        if (hyp_cost < best_label_cost) {
          best_label_cost = hyp_cost;
          arg_max = l;
        }
      }

      uint lx_best = arg_min % nHorLabels;
      uint ly_best = arg_min / nHorLabels;

      best_label_cost += smoothWorstCost(lx_best,ly_best);

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          Real hyp_cost = label_cost(x,y,ly*nHorLabels+lx);
	
          if (hyp_cost > best_label_cost + smoothWorstCost(lx,ly)) {

            label_cost(x,y,ly*nHorLabels+lx) += 100.0;
            nVarExcluded++;
          }
        }
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars." << std::endl;
#endif

  Math3D::NamedTensor<Real> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

  /*** initialization ***/
#if 1
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      Real min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

#ifndef USE_CUDA
  //  use_cuda = false;
#endif

  double h_offs = min_x_disp;
  double v_offs = min_y_disp;

  uint iter = 1;

  Real energy = 0.0;

  if (false) {

#ifdef USE_CUDA    
#endif   
  }
  else {

    const float* const_label_cost = label_cost.direct_access();

    Math3D::NamedTensor<Real> aux_var(xDim,yDim,nLabels, MAKENAME(aux_var));

    Math2D::NamedMatrix<Real> hsum(xDim,yDim, 0.0, MAKENAME(hsum));
    Math2D::NamedMatrix<Real> vsum(xDim,yDim, 0.0, MAKENAME(vsum));  

#ifdef USE_EXPLICIT_GRADIENT
    Math3D::NamedTensor<Real> grad(xDim,yDim,nLabels,MAKENAME(grad));
#endif

    uint nOuterIter = 1;
    Real alpha = 0.1*epsilon / lambda;  
    
    for (uint outer_iter = 1; outer_iter <= nOuterIter; outer_iter++) {

      aux_var = var;
      
      Real prev_t = 1.0;
      
      Real last_energy = 1e50;
      Real best_energy = last_energy;
      
      uint iter_since_restart = 0;

      uint restart_threshold = (outer_iter == 1) ? 8 : 5;

      uint nInnerIter = (outer_iter == nOuterIter) ? 45000 : 50 + (outer_iter-1)*5;

      Real save_energy = 1e50;
      const Real cutoff = 1e-5;

      uint nEnergyIncreases = 0;

      bool restart = false;

      iter = 1;
      for (; iter <= nInnerIter; iter++) { 

        std::cerr << "*********** iteration " << iter << std::endl;

        const Real* const_var = var.direct_access();

        /** 1.) energy computation **/
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const uint vbase = (y*xDim+x)*nLabels;

            hsum(x,y) = 0.0;

            //update h-sum
            for (uint lh = 0; lh < nHorLabels; lh++) {
	  
              Real sum_var = 0.0;

              for (uint lv = 0; lv < nVertLabels; lv++) {
                sum_var += const_var[vbase + lv*nHorLabels + lh];
              }
	  
              hsum(x,y) += (lh + h_offs) * sum_var;
            }

            vsum(x,y) = 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              Real sum_var = 0.0;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                sum_var += const_var[vbase + label_offs + lh];
              }

              vsum(x,y) += (lv+v_offs)*sum_var;
            }
          }
        }
      
        energy = (Real) 0.0;
      
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const Real cur_hsum = hsum(x,y);
            const Real cur_vsum = vsum(x,y);
	    
            if (x > 0) {

              const Real hdiff = cur_hsum - hsum(x-1,y);
              energy += psi(hdiff,epsilon); 

              const Real vdiff = cur_vsum - vsum(x-1,y);
              energy += psi(vdiff,epsilon); 
            }
            if (y > 0) {
		
              const Real hdiff = cur_hsum - hsum(x,y-1);
              energy += psi(hdiff,epsilon);

              const Real vdiff = cur_vsum - vsum(x,y-1);
              energy += psi(vdiff,epsilon); 
            }

            if (neighborhood >= 8) {
	      
              if (x > 0 && y > 0) {
                const Real hdiff = cur_hsum - hsum(x-1,y-1);
                energy += inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x-1,y-1);
                energy += inv_sqrt2*psi(vdiff,epsilon);
              }
              if (x+1 < xDim && y > 0) {
                const Real hdiff = cur_hsum - hsum(x+1,y-1);
                energy += inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x+1,y-1);
                energy += inv_sqrt2*psi(vdiff,epsilon);
              }
            }
          }
        }

        energy *= lambda;

        //std::cerr << "intermediate energy: " << energy << std::endl;
  
        for (uint i=0; i < nVars; i++) {
          energy += const_var[i] * const_label_cost[i];
        }

        std::cerr.precision(8);
        std::cerr << "energy: " << energy << std::endl;

        if ((iter % 15) == 0) {

          if (iter >= 45 && fabs(save_energy - energy) < cutoff) {

            std::cerr << "OSCILLATION OR (SLOW) CONVERGENCE DETECTED -> CUTOFF)" << std::endl;
            break;
          }

          save_energy = energy;
        }

        if (energy > last_energy)
          nEnergyIncreases++;

#if 1
        std::cerr << "restart: " << restart << std::endl;
        if  (restart || energy > 1.2*best_energy || (energy > last_energy && (energy > best_energy + 10.0 || nEnergyIncreases > 3)
                                                     && iter_since_restart >= restart_threshold)) {

          iter_since_restart = 0;
          nEnergyIncreases = 0;

          aux_var = var; 
          prev_t = 1.0;
          if (restart || energy > last_energy) {
            alpha *= 0.65;
          }

          restart = false;

          std::cerr << "RESTART because of energy increase, new alpha " << alpha << std::endl;
#ifdef USE_EXPLICIT_GRADIENT
          std::cerr << "gradient norm: " << grad.norm() << std::endl;
#endif
        }

#endif
        iter_since_restart++;

        /** 2.) gradient computation **/
        const Real* const_aux_var = aux_var.direct_access();

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
	    
            const uint vbase = (y*xDim+x)*nLabels;

            hsum(x,y) = 0.0;
            vsum(x,y) = 0.0;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels; lh++) {

              Real sum = 0.0;
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum += aux_var(x,y,lv*nHorLabels + lh);
                sum += const_aux_var[vbase + lv*nHorLabels + lh];
              }
	  
              hsum(x,y) += (lh+h_offs)*sum;
            }

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              Real sum = 0.0;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum += aux_var(x,y,label_offs + lh);
                sum += const_aux_var[vbase + label_offs + lh];
              }
	  
              vsum(x,y) += (lv+v_offs)*sum;
            }
          }
        }

#ifdef USE_EXPLICIT_GRADIENT
        double cur_aux_energy = 0.0;
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          cur_aux_energy += const_aux_var[v] * const_label_cost[v];

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const Real cur_hsum = hsum(x,y);
            const Real cur_vsum = vsum(x,y);
	    
            if (x > 0) {

              const Real hdiff = cur_hsum - hsum(x-1,y);
              cur_aux_energy += lambda*psi(hdiff,epsilon);

              const Real vdiff = cur_vsum - vsum(x-1,y);
              cur_aux_energy += lambda*psi(vdiff,epsilon);
            }
            if (y > 0) {
		
              const Real hdiff = cur_hsum - hsum(x,y-1);
              cur_aux_energy += lambda*psi(hdiff,epsilon);

              const Real vdiff = cur_vsum - vsum(x,y-1);
              cur_aux_energy += lambda*psi(vdiff,epsilon);
            }

            if (neighborhood >= 8) {
	      
              if (x > 0 && y > 0) {
                const Real hdiff = cur_hsum - hsum(x-1,y-1);
                cur_aux_energy += lambda*inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x-1,y-1);
                cur_aux_energy += lambda*inv_sqrt2*psi(vdiff,epsilon);
              }
              if (x+1 < xDim && y > 0) {
                const Real hdiff = cur_hsum - hsum(x+1,y-1);
                cur_aux_energy += lambda*inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x+1,y-1);
                cur_aux_energy += lambda*inv_sqrt2*psi(vdiff,epsilon);
              }
            }
          }
        }

        Math3D::Tensor<Real> save_aux = aux_var;
#endif


#ifdef USE_EXPLICIT_GRADIENT
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          grad.direct_access(v) = const_label_cost[v];
#else
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          aux_var.direct_access(v) -= alpha * const_label_cost[v];
#endif

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const Real cur_hsum = hsum(x,y);
            const Real cur_vsum = vsum(x,y);

            Real hgrad = 0.0;
            Real vgrad = 0.0;

            if (x > 0) {
              const Real temp_hdiff = cur_hsum - hsum(x-1,y);
              hgrad += psi_prime(temp_hdiff,epsilon);

              const Real temp_vdiff = cur_vsum - vsum(x-1,y);
              vgrad += psi_prime(temp_vdiff,epsilon);
            }
            if (x+1 < xDim) {
              const Real temp_hdiff = cur_hsum - hsum(x+1,y);
              hgrad += psi_prime(temp_hdiff,epsilon);

              const Real temp_vdiff = cur_vsum - vsum(x+1,y);
              vgrad += psi_prime(temp_vdiff,epsilon);
            }
            if (y > 0) {
              const Real temp_hdiff = cur_hsum - hsum(x,y-1);
              hgrad += psi_prime(temp_hdiff,epsilon);

              const Real temp_vdiff = cur_vsum - vsum(x,y-1);
              vgrad += psi_prime(temp_vdiff,epsilon);
            }
            if (y+1 < yDim) {
              const Real temp_hdiff = cur_hsum - hsum(x,y+1);
              hgrad += psi_prime(temp_hdiff,epsilon);

              const Real temp_vdiff = cur_vsum - vsum(x,y+1);
              vgrad += psi_prime(temp_vdiff,epsilon);
            }


            if (neighborhood >= 8) {

              if (x > 0 && y > 0) {
                const Real temp_hdiff = cur_hsum - hsum(x-1,y-1);
                hgrad += inv_sqrt2*psi_prime(temp_hdiff,epsilon);
		
                const Real temp_vdiff = cur_vsum - vsum(x-1,y-1);
                vgrad += inv_sqrt2*psi_prime(temp_vdiff,epsilon);
              }
              if (x+1 < xDim && y+1 < yDim) {
                const Real temp_hdiff = cur_hsum - hsum(x+1,y+1);
                hgrad += inv_sqrt2*psi_prime(temp_hdiff,epsilon);
		
                const Real temp_vdiff = cur_vsum - vsum(x+1,y+1);
                vgrad += inv_sqrt2*psi_prime(temp_vdiff,epsilon);
              }
              if (x > 0 && y+1 < yDim) {
                const Real temp_hdiff = cur_hsum - hsum(x-1,y+1);
                hgrad += inv_sqrt2*psi_prime(temp_hdiff,epsilon);
		
                const Real temp_vdiff = cur_vsum - vsum(x-1,y+1);
                vgrad += inv_sqrt2*psi_prime(temp_vdiff,epsilon);
              }
              if (x+1 < xDim && y > 0) {	      
                const Real temp_hdiff = cur_hsum - hsum(x+1,y-1);
                hgrad += inv_sqrt2*psi_prime(temp_hdiff,epsilon);
		
                const Real temp_vdiff = cur_vsum - vsum(x+1,y-1);
                vgrad += inv_sqrt2*psi_prime(temp_vdiff,epsilon);
              }
            }
	    
            hgrad *= lambda;
            vgrad *= lambda;

            for (uint lh = 0; lh < nHorLabels; lh++) {

              Real cur_grad = hgrad * (lh+h_offs);

              for (uint lv = 0; lv < nVertLabels; lv++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, lv*nHorLabels + lh) += cur_grad;
#else
                aux_var(x,y, lv*nHorLabels + lh) -= alpha* cur_grad;
#endif
              }
            }
	    
            for (uint lv = 0; lv < nVertLabels; lv++) {

              Real cur_grad = vgrad * (lv+v_offs);

              for (uint lh = 0; lh < nHorLabels; lh++) {
#ifdef USE_EXPLICIT_GRADIENT
                grad(x,y, lv*nHorLabels + lh) += cur_grad;
#else
                aux_var(x,y, lv*nHorLabels + lh) -= alpha* cur_grad;
#endif
              }
            }
          }
        }

        last_energy = energy;
        best_energy = std::min(best_energy,last_energy);

        /** 3.) perform a step of gradient descent **/

#ifdef USE_EXPLICIT_GRADIENT
        grad *= alpha;
        aux_var -= grad;
#endif

        /** 4.) reprojection [Michelot 1986] **/
    
        for (uint y=0; y < yDim; y++) {

          //std::cerr << "y: " << y << std::endl;

          for (uint x=0; x < xDim; x++) {

            projection_on_simplex(aux_var.direct_access() + (y*xDim+x)*nLabels, nLabels);
          }
        }

#ifdef USE_EXPLICIT_GRADIENT
        double new_aux_energy = 0.0;

        new_aux_energy = 0.0;
        for (uint v=0; v < xDim*yDim*nLabels; v++)
          new_aux_energy += aux_var.direct_access(v) * const_label_cost[v];

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
	    
            const uint vbase = (y*xDim+x)*nLabels;

            hsum(x,y) = 0.0;
            vsum(x,y) = 0.0;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels; lh++) {

              Real sum = 0.0;
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                //sum += aux_var(x,y,lv*nHorLabels + lh);
                sum += aux_var.direct_access(vbase + lv*nHorLabels + lh);
              }
	  
              hsum(x,y) += (lh+h_offs)*sum;
            }

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              Real sum = 0.0;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                //sum += aux_var(x,y,label_offs + lh);
                sum += aux_var.direct_access(vbase + label_offs + lh);
              }
	  
              vsum(x,y) += (lv+v_offs)*sum;
            }
          }
        }

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const Real cur_hsum = hsum(x,y);
            const Real cur_vsum = vsum(x,y);
	    
            if (x > 0) {

              const Real hdiff = cur_hsum - hsum(x-1,y);
              new_aux_energy += lambda*psi(hdiff,epsilon);

              const Real vdiff = cur_vsum - vsum(x-1,y);
              new_aux_energy += lambda*psi(vdiff,epsilon);
            }
            if (y > 0) {
		
              const Real hdiff = cur_hsum - hsum(x,y-1);
              new_aux_energy += lambda*psi(hdiff,epsilon);

              const Real vdiff = cur_vsum - vsum(x,y-1);
              new_aux_energy += lambda*psi(vdiff,epsilon);
            }

            if (neighborhood >= 8) {
	      
              if (x > 0 && y > 0) {
                const Real hdiff = cur_hsum - hsum(x-1,y-1);
                new_aux_energy += lambda*inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x-1,y-1);
                new_aux_energy += lambda*inv_sqrt2*psi(vdiff,epsilon);
              }
              if (x+1 < xDim && y > 0) {
                const Real hdiff = cur_hsum - hsum(x+1,y-1);
                new_aux_energy += lambda*inv_sqrt2*psi(hdiff,epsilon);
		
                const Real vdiff = cur_vsum - vsum(x+1,y-1);
                new_aux_energy += lambda*inv_sqrt2*psi(vdiff,epsilon);
              }
            }
          }
        }	

        double check_rhs = cur_aux_energy;
        std::cerr << "cur_aux_energy: " << cur_aux_energy << std::endl;
        for (uint v=0; v < xDim*yDim*nLabels; v++) {

          double diff = save_aux.direct_access(v) - aux_var.direct_access(v);

          //NOTE: multiplication with alpha is already included in the gradient
          check_rhs -= diff*(grad.direct_access(v) / alpha);	  
	  
          check_rhs += (0.5 / alpha) * diff*diff;
        }
        assert(check_rhs < cur_aux_energy);

        std::cerr << "check_rhs: " << check_rhs << std::endl;
        std::cerr << "new aux energy: " << new_aux_energy << std::endl;
        std::cerr << "valid step size: " << (new_aux_energy <= check_rhs) << std::endl;

        restart = (new_aux_energy > check_rhs);

        // 	if (new_aux_energy > check_rhs) {
        // 	  alpha *= 0.75;

        // 	  aux_var = save_aux;
        // 	}
#endif

        //TEST
        //if (new_aux_energy <=  check_rhs) {
        //END_TEST

        const Real new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
        const Real nesterov_fac = (prev_t - 1) / new_t;
        //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+2));
	  
        //Real nesterov_fac = (prev_t - 1) / new_t;
        //if (iter <= 1500)
        //  nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+1));

        //const Real old_nesterov_fac = ((double) (iter_since_restart-2)) / ((double) (iter_since_restart));
        //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+1));
        //std::cerr << "lee-way: " << (old_nesterov_fac*old_nesterov_fac - nesterov_fac*nesterov_fac + nesterov_fac) << std::endl;

        //const Real nesterov_fac = ((double) (iter_since_restart+1)) / 2.0;

        for (uint i=0; i < aux_var.size(); i++) {
	    
          const Real old_aux = aux_var.direct_access(i);
          aux_var.direct_access(i) = old_aux + nesterov_fac*(old_aux - var.direct_access(i)) ;
          var.direct_access(i) = old_aux;
        }
	  
        prev_t = new_t;
        //TEST
        //}
        //END_TEST
      }  
    }
  } //end of outer loop

  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real max_var = -1.0;
      Real sum = 0.0;

      float arg_max_x = MAX_UINT;
      float arg_max_y = MAX_UINT;
      
      uint arg_max = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          Real val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) * inv_spacing + min_x_disp;
            arg_max_y = ((int) ly) * inv_spacing + min_y_disp;
            arg_max = ly*nHorLabels + lx;
          }
        }
      }

      labeling(x,y) = arg_max;

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);

  return energy;
}

/******************************************************************************************************************/

inline double gc_pairterm(double sum, double epsilon) {

  if (sum < 1.0 - 4.0*epsilon)
    return 0.0;
  else if (sum < 1.0 + 4.0*epsilon) {
    double sqr = sum - (1-4.0*epsilon);
    sqr *= sqr;
    return sqr / (16.0*epsilon);
  }
  else
    return sum - 1.0;
}

inline double gc_pairgrad(double sum, double epsilon) {

  if (sum < 1.0 - 4.0*epsilon)
    return 0.0;
  else if (sum < 1.0 + 4.0*epsilon) {
    return (sum - (1-4.0*epsilon)) / (8.0*epsilon);
  }
  else
    return 1.0;
}

double motion_estimation_goldluecke_cremers(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {


  // WARNING: this does not make sense as the optimum is always 0

  double abs_epsilon = 0.01;
  double pair_epsilon = 0.01;

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  
  float inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= inv_spacing;
  const Real diag_lambda = lambda / sqrt(2.0);

  const Real inv_sqrt2 = 1.0 / sqrt(2.0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  Math3D::NamedTensor<Real> hvar(xDim,yDim,nHorLabels,1.0 / nHorLabels, MAKENAME(hvar));
  Math3D::NamedTensor<Real> vvar(xDim,yDim,nVertLabels,1.0 / nVertLabels, MAKENAME(vvar));

  Math3D::NamedTensor<Real> aux_hvar(xDim,yDim,nHorLabels,0.0, MAKENAME(aux_hvar));
  Math3D::NamedTensor<Real> aux_vvar(xDim,yDim,nVertLabels,0.0 / nVertLabels, MAKENAME(aux_vvar));
  
  Math3D::NamedTensor<Real> hgrad(xDim,yDim,nHorLabels,0.0, MAKENAME(hgrad));
  Math3D::NamedTensor<Real> vgrad(xDim,yDim,nVertLabels,0.0 / nVertLabels, MAKENAME(vgrad));

  Math3D::Tensor<Real> accu(xDim,yDim,2);

  double alpha  = 0.001;

  aux_hvar = hvar;
  aux_vvar = vvar;

  double prev_t = 1.0;
  
  for (uint iter=1; iter <= 45000; iter++) {

    std::cerr << "********** iteration " << iter << std::endl;
    
    //1.) compute energy of current point
    
    double energy = 0.0;

    //a) compute the regularity term
    
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        double haccu = 0.0;
        for (uint h=0; h < nHorLabels; h++) {
          haccu += h*hvar(x,y,h);
        }

        accu(x,y,0) = haccu;

        double vaccu = 0.0;
        for (uint v=0; v < nVertLabels; v++) {
          vaccu += v*vvar(x,y,v);
        }

        accu(x,y,1) = vaccu;
      }
    }

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        double cur_haccu = accu(x,y,0);
        double cur_vaccu = accu(x,y,1);

        if (x > 0) {
          energy += psi(cur_haccu - accu(x-1,y,0),abs_epsilon)
            + psi(cur_vaccu - accu(x-1,y,1),abs_epsilon);
        }
        if (y > 0) {
          energy += psi(cur_haccu - accu(x,y-1,0),abs_epsilon)
            + psi(cur_vaccu - accu(x,y-1,1),abs_epsilon);
        }

        if (neighborhood >= 8) {

          if (x > 0 && y > 0) {
            energy += inv_sqrt2 * (psi(cur_haccu - accu(x-1,y-1,0),abs_epsilon)
                                   + psi(cur_vaccu - accu(x-1,y-1,1),abs_epsilon));
          }
          if (x > 0 && y+1 < yDim) {
            energy += inv_sqrt2 * (psi(cur_haccu - accu(x-1,y+1,0),abs_epsilon)
                                   + psi(cur_vaccu - accu(x-1,y+1,1),abs_epsilon));
          }
        }
      }
    }    

    energy *= lambda;

    //b) compute the data term
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        for (uint v=0; v < nVertLabels; v++)
          for (uint h=0; h < nHorLabels; h++) 
            energy += label_cost(x,y,v*nHorLabels+h) * gc_pairterm(hvar(x,y,h)+vvar(x,y,v), pair_epsilon);
      }
    }    

    std::cerr << "energy: " << energy << std::endl;
    
    // 2.) compute the gradient at the auxiliary point
    hgrad.set_constant(0.0);
    vgrad.set_constant(0.0);

    //a) gradient of regularity term
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        double haccu = 0.0;
        for (uint h=0; h < nHorLabels; h++) {
          haccu += h*aux_hvar(x,y,h);
        }

        accu(x,y,0) = haccu;

        double vaccu = 0.0;
        for (uint v=0; v < nVertLabels; v++) {
          vaccu += v*aux_vvar(x,y,v);
        }

        accu(x,y,1) = vaccu;
      }
    }


    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        double cur_haccu = accu(x,y,0);
        double cur_vaccu = accu(x,y,1);

        double cur_hgrad = 0.0;
        double cur_vgrad = 0;

        if (x > 0) {
          cur_hgrad += psi_prime(cur_haccu - accu(x-1,y,0),abs_epsilon);
          cur_vgrad += psi_prime(cur_vaccu - accu(x-1,y,1),abs_epsilon);
        }
        if (x+1 < xDim) {
          cur_hgrad += psi_prime(cur_haccu - accu(x+1,y,0),abs_epsilon);
          cur_vgrad += psi_prime(cur_vaccu - accu(x+1,y,1),abs_epsilon);
        }	
        if (y > 0) {
          cur_hgrad += psi_prime(cur_haccu - accu(x,y-1,0),abs_epsilon);
          cur_vgrad += psi_prime(cur_vaccu - accu(x,y-1,1),abs_epsilon);
        }
        if (y+1 < yDim) {
          cur_hgrad += psi_prime(cur_haccu - accu(x,y+1,0),abs_epsilon);
          cur_vgrad += psi_prime(cur_vaccu - accu(x,y+1,1),abs_epsilon);
        }

        if (neighborhood >= 8) {
	  
          if (x > 0 && y > 0) {
            cur_hgrad += inv_sqrt2 * psi_prime(cur_haccu - accu(x-1,y-1,0),abs_epsilon);
            cur_vgrad += inv_sqrt2 * psi_prime(cur_vaccu - accu(x-1,y-1,1),abs_epsilon);
          }
          if (x > 0 && y+1 < yDim) {
            cur_hgrad += inv_sqrt2 * psi_prime(cur_haccu - accu(x-1,y+1,0),abs_epsilon);
            cur_vgrad += inv_sqrt2 * psi_prime(cur_vaccu - accu(x-1,y+1,1),abs_epsilon);
          }
          if (x+1 < xDim && y > 0) {
            cur_hgrad += inv_sqrt2 * psi_prime(cur_haccu - accu(x+1,y-1,0),abs_epsilon);
            cur_vgrad += inv_sqrt2 * psi_prime(cur_vaccu - accu(x+1,y-1,1),abs_epsilon);
          }
          if (x+1 < xDim && y+1 < yDim) {
            cur_hgrad += inv_sqrt2 * psi_prime(cur_haccu - accu(x+1,y+1,0),abs_epsilon);
            cur_vgrad += inv_sqrt2 * psi_prime(cur_vaccu - accu(x+1,y+1,1),abs_epsilon);
          }
        }

        for (uint h=0; h < nHorLabels; h++)
          hgrad(x,y,h) += h*cur_hgrad;
        for (uint v=0; v < nVertLabels; v++)
          vgrad(x,y,v) += v*cur_vgrad;
      }
    }

    hgrad *= lambda;
    vgrad *= lambda;
    
    //b) gradient of data term
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        for (uint v=0; v < nVertLabels; v++) {
          for (uint h=0; h < nHorLabels; h++) {
            double grad = label_cost(x,y,v*nHorLabels+h) * gc_pairgrad(aux_hvar(x,y,h)+aux_vvar(x,y,v), pair_epsilon);

            hgrad(x,y,h) += grad;
            vgrad(x,y,v) += grad;
          }
        }
      }
    }    


    // 3.) go in the direction of the negative gradient and reproject
    hgrad *= alpha;
    vgrad *= alpha;

    aux_hvar -= hgrad;
    aux_vvar -= vgrad;

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        projection_on_simplex(aux_hvar.direct_access() + (y*xDim+x)*nHorLabels, nHorLabels);
        projection_on_simplex(aux_vvar.direct_access() + (y*xDim+x)*nVertLabels, nVertLabels);
      }
    }    

    // 4.) update the nesterov variables
    const Real new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
    const Real nesterov_fac = (prev_t - 1) / new_t;

    for (uint i=0; i < aux_hvar.size(); i++) {
      
      const Real old_aux = aux_hvar.direct_access(i);
      aux_hvar.direct_access(i) = old_aux + nesterov_fac*(old_aux - hvar.direct_access(i)) ;
      hvar.direct_access(i) = old_aux;
    }
    for (uint i=0; i < aux_vvar.size(); i++) {
      
      const Real old_aux = aux_vvar.direct_access(i);
      aux_vvar.direct_access(i) = old_aux + nesterov_fac*(old_aux - vvar.direct_access(i)) ;
      hvar.direct_access(i) = old_aux;
    }

    prev_t = new_t;
  }
  

  return 0.0; //TODO: correct

}

/******************************************************************************************************************/

double motion_estimation_smoothabs_nesterov(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                            double epsilon, bool use_cuda) {

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint nVars = xDim*yDim*nLabels;

  Math3D::NamedTensor<Real> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));

  const uint zero_label = nHorLabels* (  (-min_y_disp + 1)*spacing - (spacing-1)    )
    + (-min_x_disp + 1)*spacing - (spacing-1);
  Math2D::Matrix<uint> labeling(xDim,yDim,zero_label);
  
  float inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= inv_spacing;
  const Real diag_lambda = lambda / sqrt(2.0);

  const Real inv_sqrt2 = 1.0 / sqrt(2.0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

#if 0
  /***** variable exclusion stage *****/
  Math2D::Matrix<Real>  smoothWorstCost(nHorLabels, nVertLabels,0.0);

  for (uint h=0; h < nHorLabels; h++) {
    for (uint v=0; v < nVertLabels; v++) {
      
      Real dist = std::max(h, nHorLabels-1-h) + std::max(v, nVertLabels-1-v);

      if (neighborhood == 4)
        dist *= 4.0 * lambda;
      else {
        assert(neighborhood == 8);

        dist *= 4.0 * (1.0 + sqrt(0.5)) * lambda;
      }

      smoothWorstCost(h,v) = dist;
    }
  }

  uint nVarExcluded = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real best_label_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        Real hyp_cost = label_cost(x,y,l);
        if (hyp_cost < best_label_cost)
          best_label_cost = hyp_cost;
      }


      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          Real hyp_cost = label_cost(x,y,ly*nHorLabels+lx);
	
          if (hyp_cost > best_label_cost) { // + smoothWorstCost(lx,ly)) {

            label_cost(x,y,ly*nHorLabels+lx) += 100.0;
            nVarExcluded++;
          }
        }
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars." << std::endl;
#endif

  Math3D::NamedTensor<Real> var(xDim,yDim,nLabels,1.0 / nLabels, MAKENAME(var));

  /*** initialization ***/
#if 0
  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint min_l = MAX_UINT;
      Real min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      var(x,y,min_l) = 1.0;
    }
  }
#endif

#if 1
  //initialization by expansion moves

  std::cerr << "-------- initializing by expansion moves" << std::endl;
  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling, 3);

  var.set_constant(0.0);
  for (uint y=0; y < yDim; y++)
    for (uint x=0; x < xDim; x++)
      var(x,y,labeling(x,y)) = 1.0;
#endif


#ifndef USE_CUDA
  use_cuda = false;
#endif

  uint iter = 1;

  Real energy = 0.0;

  if (use_cuda) {

    TODO("cuda implementation of smoothed absolutes");
#ifdef USE_CUDA    

#endif   
  }
  else {

    Math3D::NamedTensor<Real> aux_var(xDim,yDim,nLabels, MAKENAME(aux_var));

    //NOTE: by construction hmarginal(.,.,nHorLabels-1) would always be 1
    Math3D::NamedTensor<Real> hmarginal(xDim,yDim,nHorLabels-1, 0.0, MAKENAME(hmarginal));
    //NOTE: by construction vmarginal(.,.,nVertLabels-1) would always be 1
    Math3D::NamedTensor<Real> vmarginal(xDim,yDim,nVertLabels-1, 0.0, MAKENAME(vmarginal));  
    Math3D::NamedTensor<Real> grad(xDim,yDim,nLabels,MAKENAME(grad));

    uint nOuterIter = 1;

    //TODO: try successively decreasing epsilon

    Real alpha = 0.0000125 / lambda;  
    
    for (uint outer_iter = 1; outer_iter <= nOuterIter; outer_iter++) {

      aux_var = var;
      
      Real prev_t = 1.0;
      
      const Real neg_threshold = 1e-12;
      
      Real last_energy = 1e50;
      Real best_energy = last_energy;
      
      uint iter_since_restart = 0;

      uint restart_threshold = (outer_iter == 1) ? 7 : 5;

      uint nInnerIter = (outer_iter == nOuterIter) ? 4500 : 50 + (outer_iter-1)*5;

      iter = 1;
      for (; iter <= nInnerIter; iter++) { 

        std::cerr << "*********** iteration " << iter << std::endl;

        /** 1.) energy computation **/
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum_var = 0.0;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                sum_var += var(x,y,lv*nHorLabels + lh);
              }
	  
              hmarginal(x,y,lh) = sum_var;
            }
	
            sum_var = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                sum_var += var(x,y,label_offs + lh);
              }

              vmarginal(x,y,lv) = sum_var;
            }
          }
        }
      
        energy = (Real) 0.0;
      
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real curvar_marginal = hmarginal(x,y,h);

              if (x > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x-1,y,h);

                energy += sqrt(temp2*temp2+epsilon);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - hmarginal(x,y-1,h);

                energy += sqrt(temp2*temp2+epsilon);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x-1,y-1,h);

                  energy += inv_sqrt2*sqrt(temp2*temp2+epsilon);
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp2 = curvar_marginal - hmarginal(x+1,y-1,h);

                  energy += inv_sqrt2*sqrt(temp2*temp2+epsilon);
                }
              }
            }

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real curvar_marginal = vmarginal(x,y,v);

              if (x > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x-1,y,v);

                energy += sqrt(temp2*temp2+epsilon);
              }
              if (y > 0) {
                const Real temp2 = curvar_marginal - vmarginal(x,y-1,v);

                energy += sqrt(temp2*temp2+epsilon);
              }

              if (neighborhood >= 8) {

                if (x > 0 && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x-1,y-1,v);

                  energy += inv_sqrt2*sqrt(temp2*temp2+epsilon);
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp2 = curvar_marginal - vmarginal(x+1,y-1,v);

                  energy += inv_sqrt2*sqrt(temp2*temp2+epsilon);
                }
              }
            }
          }
        }

        energy *= lambda;

        //std::cerr << "intermediate energy: " << energy << std::endl;
  
        for (uint i=0; i < nVars; i++)
          energy += var.direct_access(i) * label_cost.direct_access(i);

        std::cerr << "energy: " << energy << std::endl;

        if  ((iter_since_restart >= 2 && energy > 1.5*best_energy) 
             || (energy > last_energy && iter_since_restart >= restart_threshold) 
             /*||  (iter_since_restart == 100 )*/  ) {

          //std::cerr << "BREAK because of energy increase" << std::endl;
          //break;
          iter_since_restart = 1;

          aux_var = var; 
          prev_t = 1.0;
          if (energy > last_energy) {
            //alpha *= 0.25;
            alpha *= 0.5;
          }

          std::cerr << "RESTART because of energy increase, new alpha " << alpha << std::endl;
          std::cerr << "gradient norm: " << grad.norm() << std::endl;
        }
        else
          iter_since_restart++;


        /** 2.) gradient computation **/

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real sum = 0.0;

            //update h-marginals
            for (uint lh = 0; lh < nHorLabels-1; lh++) {
	  
              for (uint lv = 0; lv < nVertLabels; lv++) {
                sum += aux_var(x,y,lv*nHorLabels + lh);
              }
	  
              //these bounds are not guaranteed for the auxiliary variable 
              //assert(sum >= -0.05);
              //assert(sum <= 1.05);
	  
              hmarginal(x,y,lh) = sum;
            }
	
            sum = (Real) 0.0;

            //update v-marginals
            for (uint lv = 0; lv < nVertLabels-1; lv++) {
	  
              const uint label_offs = lv*nHorLabels;

              for (uint lh = 0; lh < nHorLabels; lh++)  {
                sum += aux_var(x,y,label_offs + lh);
              }

              //if (! (sum >= -1e-8))
              //  std::cerr << "sum: " << sum << std::endl;

              //these bounds are not guaranteed for the auxiliary variable 	  
              //assert(sum >= -0.05);
              //assert(sum <= 1.05);
	  
              vmarginal(x,y,lv) = sum;
            }

          }
        }

        grad = label_cost;

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            Real hgrad = (Real) 0.0;

            for (int h=nHorLabels-2; h >= 0; h--) {

              const Real cur_marginal = hmarginal(x,y,h);

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - hmarginal(x-1,y,h);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - hmarginal(x+1,y,h);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (y > 0) {
                const Real temp = cur_marginal - hmarginal(x,y-1,h);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - hmarginal(x,y+1,h);

                diff += temp / sqrt(temp*temp+epsilon); 
              }

              diff *= lambda;

              hgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x-1,y-1,h);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x+1 < xDim && y > 0) {

                  const Real temp = cur_marginal - hmarginal(x+1,y-1,h);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x > 0 && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x-1,y+1,h);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x+1 < xDim && y+1 < yDim) {

                  const Real temp = cur_marginal - hmarginal(x+1,y+1,h);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }

                hgrad += diff * diag_lambda;
              }

              for (uint v=0; v < nVertLabels; v++)
                grad(x,y, v*nHorLabels + h) += hgrad;
            }

            Real vgrad = 0.0;

            for (int v=nVertLabels-2; v>= 0; v--) {

              const Real cur_marginal = vmarginal(x,y,v);

              const uint label_offs = v*nHorLabels;

              Real diff = (Real) 0.0;
              if (x > 0) {
                const Real temp = cur_marginal - vmarginal(x-1,y,v);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (x+1 < xDim) {
                const Real temp = cur_marginal - vmarginal(x+1,y,v);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (y > 0) {
                const Real temp = cur_marginal - vmarginal(x,y-1,v);

                diff += temp / sqrt(temp*temp+epsilon); 
              }
              if (y+1 < yDim) {
                const Real temp = cur_marginal - vmarginal(x,y+1,v);

                diff += temp / sqrt(temp*temp+epsilon); 
              }

              diff *= lambda;
              vgrad += diff;

              if (neighborhood >= 8) {

                diff = (Real) 0.0;

                if (x > 0 && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x-1,y-1,v);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x+1 < xDim && y > 0) {
                  const Real temp = cur_marginal - vmarginal(x+1,y-1,v);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x > 0 && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x-1,y+1,v);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
                if (x+1 < xDim && y+1 < yDim) {
                  const Real temp = cur_marginal - vmarginal(x+1,y+1,v);

                  diff += temp / sqrt(temp*temp+epsilon); 
                }
	  
                vgrad += diff * diag_lambda;
              }

              for (uint h=0; h < nHorLabels; h++)
                grad(x,y, label_offs + h) += vgrad;
            }

          }
        }

        last_energy = energy;
        best_energy = std::min(best_energy,last_energy);

        /** 3.) perform a step of gradient descent **/

        grad *= ((Real) (-1.0)) * alpha;
        aux_var += grad;

        /** 4.) reprojection [Michelot 1986] **/
    
        for (uint y=0; y < yDim; y++) {

          //std::cerr << "y: " << y << std::endl;

          for (uint x=0; x < xDim; x++) {
    
            uint nNonZeros = nLabels;

            Real* aux_var_ptr = aux_var.direct_access() + (y*xDim+x)*nLabels;
	
            while (nNonZeros > 0) {

              // 	  std::cerr << "nNonZeros: " << nNonZeros << std::endl;

              //a) project onto the plane
              Real mean_dev = ((Real) (- 1.0));
              for (uint l=0; l < nLabels; l++) {
                //mean_dev += aux_var(x,y,l);
                mean_dev += aux_var_ptr[l];
              }
	  
              mean_dev /= nNonZeros;
	  
#if 1
              uint nPrevNonZeros = nNonZeros;
              //nNonZeros = nLabels;

              bool all_pos = true;

              for (uint l=0; l < nLabels; l++) {

                Real temp = aux_var_ptr[l];

                if (nPrevNonZeros == nLabels || temp != (Real) 0.0) {

                  temp -= mean_dev;

                  if (temp <= neg_threshold) {

                    all_pos = false;
                    temp = (Real) 0.0;
                    nNonZeros--;
                  }

                  aux_var_ptr[l] = temp;
                }
              }

              if (all_pos)
                break;
#else
              //b) subtract mean
              bool all_pos = true;

              for (uint l=0; l < nLabels; l++) {

                // if (nNonZeros == nLabels || aux_var(x,y,l) != 0.0) {
                //   aux_var(x,y,l) -= mean_dev;
	      
                //   if (aux_var(x,y,l) < 0.0)
                // 	all_pos = false;
                // }
                if (nNonZeros == nLabels || aux_var_ptr[l] != (Real) 0.0) {
                  aux_var_ptr[l] -= mean_dev;
	      
                  if (aux_var_ptr[l] < (Real) 0.0)
                    all_pos = false;
                }
              }

              if (all_pos)
                break;

              uint nPrevNonZeros = nNonZeros;

              nNonZeros = nLabels;
              for (uint l=0; l < nLabels; l++) {

                // if (aux_var(x,y,l) <= neg_threshold) {
                //   aux_var(x,y,l) = 0.0;
                //   nNonZeros--;
                // }
                if (aux_var_ptr[l] <= neg_threshold) {
                  aux_var_ptr[l] = (Real) 0.0;
                  nNonZeros--;
                }

              }

              if (! (nPrevNonZeros > nNonZeros) ) {
                std::cerr << "prev non-zero: " << nPrevNonZeros << ", cur non-zero:" << nNonZeros  << std::endl;
              }

              assert(nPrevNonZeros > nNonZeros);
#endif
            }
          }
        }

        const Real new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
        const Real nesterov_fac = (prev_t - 1) / new_t;

        for (uint i=0; i < aux_var.size(); i++) {
      
          const Real old_aux = aux_var.direct_access(i);
          aux_var.direct_access(i) = old_aux + nesterov_fac*(old_aux - var.direct_access(i)) ;
          var.direct_access(i) = old_aux;
        }

        prev_t = new_t;
      }  
    }
  } //end of outer loop

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      Real max_var = -1.0;
      Real sum = 0.0;

      float arg_max_x = MAX_UINT;
      float arg_max_y = MAX_UINT;
      
      uint arg_max = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          Real val = var(x,y,ly*nHorLabels+lx);
	  
          sum += val;

          if (val > max_var) {
            max_var = val;

            arg_max_x = ((int) lx) * inv_spacing + min_x_disp;
            arg_max_y = ((int) ly) * inv_spacing + min_y_disp;
            arg_max = ly*nHorLabels + lx;
          }
        }
      }

      labeling(x,y) = arg_max;

      velocity(x,y,0) = arg_max_x;
      velocity(x,y,1) = arg_max_y;
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);

  return energy;

}
