/**** written by Thomas Schoenemann as an employee of Lund University, February 2010 ****/

#include "motion_lp.hh"
#include "matrix.hh"

#include "sparse_matrix_description.hh"
#include "conv_lp_solving.hh"
#include "timing.hh"
#include "tensor_interpolation.hh"

#ifdef HAS_CLP
#include "ClpSimplex.hpp"
#include "ClpPresolve.hpp"
#include "OsiSolverInterface.hpp"
#include "CbcModel.hpp"
#include "OsiClpSolverInterface.hpp"
#include "CglGomory.hpp"
#include "CglProbing.hpp"
#include "CglKnapsackCover.hpp"
#include "CglRedSplit.hpp"
#include "CglClique.hpp"
#include "CglFlowCover.hpp"
#include "CglMixedIntegerRounding.hpp"
#include "CglMixedIntegerRounding2.hpp"
#include "CglFlowCover.hpp"
#include "CglOddHole.hpp"
#include "CglTwomir.hpp"
#include "CbcHeuristic.hpp"
#include "CbcHeuristicLocal.hpp"
#endif

#include "conv_lp_solving.hh"
#include "projection.hh"
#include "motion_discrete.hh"
#include "motion_moves.hh"

//#define EXPLICIT_GRADIENT

#ifdef USE_OMP
//#include <omp.h>
#endif

//#define USE_MOSEK
//#define USE_GUROBI

#ifdef USE_GUROBI
#include "gurobi_c++.h"
#endif

#ifdef USE_MOSEK
#include "mosek.h"

static void MSKAPI printstr(void *handle,
                            char str[])
{
  std::cerr << "MOSEK  " << str; // << std::endl;
  //printf("%s",str);
} /* printstr */
#endif


double lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                            int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                            uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

#ifdef HAS_CLP
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

  const uint nLabelVars = xDim*yDim*nLabels;

  const uint nHorLevelVars = xDim*yDim*nHorLabels;
  const uint nVertLevelVars = xDim*yDim*nVertLabels;

  const uint nStraightTransitions = yDim * (xDim-1) + (yDim-1) * xDim;
  uint nDiagTransitions = 0;
  if (neighborhood == 8) {

    nDiagTransitions += 2*(xDim-1)*(yDim-1);
  }

  const uint nTransitions = nStraightTransitions + nDiagTransitions;

  uint nAbsVars = 2*nTransitions*(nHorLabels + nVertLabels);

  const uint nVars = nLabelVars + nHorLevelVars + nVertLevelVars + nAbsVars;

  const uint hlevel_var_offs = nLabelVars;
  const uint vlevel_var_offs = hlevel_var_offs + nHorLevelVars; 
  const uint abs_var_offs = nLabelVars + nHorLevelVars + nVertLevelVars;

  const uint nConstraints = xDim*yDim // unity constraints
    + xDim*yDim*(nHorLabels + nVertLabels)  //level constraints
    + nTransitions*(nHorLabels + nVertLabels); //constraints to model the absolutes

  Math1D::NamedVector<double> cost(nVars,0.0,MAKENAME(cost));

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint base_id = (y*xDim+x)*nLabels;

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

          cost[base_id + ly*nHorLabels+lx] = disp_cost;
        }
      }
    }
  }



  for (uint v=0; v < 2*nStraightTransitions*(nHorLabels+nVertLabels); v++)
    cost[abs_var_offs + v] = lambda;
  
  if (neighborhood == 8) {
    const uint diag_abs_var_offs = abs_var_offs + 2*nStraightTransitions*(nHorLabels+nVertLabels);
    double diag_lambda = lambda / sqrt(2.0);
    
    for (uint v=0; v < 2*nDiagTransitions*(nHorLabels+nVertLabels); v++)
      cost[diag_abs_var_offs+v] = diag_lambda;
  }

  Math1D::NamedVector<double> var_lb(nVars,0.0,MAKENAME(var_lb));
  Math1D::NamedVector<double> var_ub(nVars,1.0,MAKENAME(var_ub));

  /***** variable exclusion stage *****/
  Math2D::Matrix<double>  smoothWorstCost(nHorLabels, nVertLabels,0.0);

  for (uint h=0; h < nHorLabels; h++) {
    for (uint v=0; v < nVertLabels; v++) {
      
      double dist = std::max(h, nHorLabels-1-h) + std::max(v, nVertLabels-1-v);

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

      double best_label_cost = 1e300;
      uint arg_min = MAX_UINT;

      for (uint l=0; l < nLabels; l++) {

        double hyp_cost = cost[(y*xDim+x)*nLabels + l];
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

          double hyp_cost = cost[(y*xDim+x)*nLabels + ly*nHorLabels+lx];
	
          if (hyp_cost > best_label_cost) { // + smoothWorstCost(lx,ly)) {

            var_ub[(y*xDim+x)*nLabels + ly*nHorLabels+lx] = 0.0;
            nVarExcluded++;
          }
        }
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars." << std::endl;


  Math1D::NamedVector<double> rhs(nConstraints,0.0,MAKENAME(rhs));
  for (uint i=0; i < xDim*yDim; i++)
    rhs[i] = 1.0;

  uint nMatrixEntries = nLabelVars // for the unity constraints
    + nHorLevelVars*(nVertLabels+1) + nVertLevelVars*(nHorLabels+1) // for the level constraints
    + 8*nTransitions*nLabels; //for the absolutes

  SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);

  //set up unity constraints (label vars for a pixel must sum to 1.0)
  for (uint i=0; i < yDim*xDim; i++) {
    
    for (uint j=i*nLabels; j < (i+1)*nLabels; j++)
      lp_descr.add_entry(i,j,1.0);
  }

  //set up horizontal level constraints
  const uint hlevel_con_offs = xDim*yDim;
  for (uint i=0; i < yDim*xDim; i++) {

    for (uint lx = 0; lx < nHorLabels; lx++) {
      
      const uint row = hlevel_con_offs + i*nHorLabels + lx;

      lp_descr.add_entry(row, hlevel_var_offs+i*nHorLabels+lx, -1.0);
      if (lx > 0)
        lp_descr.add_entry(row, hlevel_var_offs+i*nHorLabels+lx-1, 1.0);

      for (uint ly = 0; ly < nVertLabels; ly++) {

        lp_descr.add_entry(row, i*nLabels + ly*nHorLabels+lx , 1.0);
      }	
    }
  }
  
  //set up vertical level constraints
  const uint vlevel_con_offs = hlevel_con_offs + xDim*yDim*nHorLabels;
  for (uint i=0; i < yDim*xDim; i++) {

    for (uint ly = 0; ly < nVertLabels; ly++) {

      const uint row = vlevel_con_offs + i*nVertLabels + ly;

      lp_descr.add_entry(row, vlevel_var_offs+i*nVertLabels+ly, -1.0);
      if (ly > 0)
        lp_descr.add_entry(row, vlevel_var_offs+i*nVertLabels+ly-1, 1.0);

      for (uint lx=0; lx < nHorLabels; lx++) {

        lp_descr.add_entry(row, i*nLabels + ly*nHorLabels+lx, 1.0);
      }
    }
  }

  //set up the constraints to model the absolutes of the horizontal and of the vertical displacement
  const uint abs_con_offs = vlevel_con_offs + yDim*xDim*nVertLabels;

  uint trans_num = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint pixel_id = y*xDim+x;

      if (x+1 < xDim) {

        const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);

        const uint neighbor_id = y*xDim+x+1;

        //horizontal displacement
        for (uint lx=0; lx < nHorLabels; lx++) {

          const uint row = row_base + lx;
	  
          lp_descr.add_entry(row, abs_var_base + 2*lx, 1.0);
          lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1.0);
	  
          lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1.0);
          lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1.0);	  
        }

        //vertical displacement
        for (uint ly=0; ly < nVertLabels; ly++) {

          const uint row = row_base + nHorLabels + ly;

          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1.0);
          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1.0);
	  
          lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1.0);
          lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1.0);
        }

        trans_num++;
      }

      if (y+1 < yDim) {

        const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);

        const uint neighbor_id = (y+1)*xDim+x;

        //horizontal displacement
        for (uint lx=0; lx < nHorLabels; lx++) {

          const uint row = row_base + lx;
	  
          lp_descr.add_entry(row, abs_var_base + 2*lx, 1.0);
          lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1.0);
	  
          lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1.0);
          lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1.0);	  
        }

        //vertical displacement
        for (uint ly=0; ly < nVertLabels; ly++) {

          const uint row = row_base + nHorLabels + ly;

          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1.0);
          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1.0);
	  
          lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1.0);
          lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1.0);
        }

        trans_num++;
      }
    }
  }

  if (neighborhood == 8) {

    //set up the constraints to model the absolutes of diagonal displacements
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint pixel_id = y*xDim+x;
	
        if (x+1 < xDim && y+1 < yDim) {
	  
          const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);
	  
          const uint neighbor_id = (y+1)*xDim+x+1;
	  
          //horizontal displacement
          for (uint lx=0; lx < nHorLabels; lx++) {
	    
            const uint row = row_base + lx;
	    
            lp_descr.add_entry(row, abs_var_base + 2*lx, 1.0);
            lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1.0);
	    
            lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1.0);
            lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1.0);	  
          }
	  
          //vertical displacement
          for (uint ly=0; ly < nVertLabels; ly++) {
	    
            const uint row = row_base + nHorLabels + ly;
	    
            lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1.0);
            lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1.0);
	    
            lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1.0);
            lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1.0);
          }
	  
          trans_num++;
        }

        if (x > 0 && y+1 < yDim) {

          const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);
	  
          const uint neighbor_id = (y+1)*xDim+x-1;
	
          //horizontal displacement
          for (uint lx=0; lx < nHorLabels; lx++) {
	    
            const uint row = row_base + lx;
	    
            lp_descr.add_entry(row, abs_var_base + 2*lx, 1.0);
            lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1.0);
	    
            lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1.0);
            lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1.0);	  
          }

          //vertical displacement
          for (uint ly=0; ly < nVertLabels; ly++) {
	    
            const uint row = row_base + nHorLabels + ly;
	    
            lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1.0);
            lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1.0);
	    
            lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1.0);
            lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1.0);
          }
	  
          trans_num++;
        }
      }
    }
  }

  //optimize
  std::cerr << nVars << " variables, " << nConstraints << " constraints" << std::endl;

#ifdef USE_OWN_CONV
  Math1D::Vector<double> conv_solution(nVars,0.0 / nLabels);
  for (uint v=0; v < xDim*yDim*nLabels; v++)
    conv_solution[v] = 1.0 / nLabels;

  eq_constrained_lp_solving_auglagrange_nesterov(nVars, nConstraints, cost.direct_access(), var_lb.direct_access(), var_ub.direct_access(),
                                                 lp_descr, rhs.direct_access(), conv_solution.direct_access());
#endif

  std::cerr << "converting matrix" << std::endl;

  timeval tStartCLP, tEndCLP;  

  CoinPackedMatrix coinMatrix(false,(int*) lp_descr.row_indices(),(int*) lp_descr.col_indices(),
                              lp_descr.value(),lp_descr.nEntries());

  ClpSimplex lpSolver;
  lpSolver.loadProblem (coinMatrix, var_lb.direct_access(), var_ub.direct_access(),   
                        cost.direct_access(), rhs.direct_access(), rhs.direct_access());

  coinMatrix.cleanMatrix();

  //lpSolver.writeMps("motion.mps");
  //std::cerr << "saved" << std::endl;

  for (uint i=0; i < nLabelVars; i++)
    lpSolver.setInteger(i);

  gettimeofday(&tStartCLP,0);

  int error = 0;

  Math1D::Vector<uint> row_start(nConstraints+1);
  lp_descr.sort_by_row(row_start);

#ifdef USE_GUROBI
  std::cerr << "trying the gurobi solver" << std::endl;

  //   GRBEnv grb_env = GRBEnv();
  
  //   GRBModel grb_model = GRBModel(grb_env);

  //   GRBVar* grb_vars = grb_model.addVars(nVars, GRB_CONTINUOUS);
  
  //   grb_model.update();

  //   delete[] grb_vars;

  
  GRBenv   *grb_env   = NULL;
  GRBmodel *grb_model = NULL;

  /* Create environment */

  error = GRBloadenv(&grb_env,NULL);
  GRBsetintparam(grb_env, GRB_INT_PAR_LPMETHOD, GRB_LPMETHOD_BARRIER);
  GRBsetdblparam(grb_env, "BarConvTol", 1e-10);

  assert (!error && grb_env != NULL);

  /* Create an empty model */

  error = GRBnewmodel(grb_env, &grb_model, "motion-lp", 0, NULL, NULL, NULL, NULL, NULL);
  assert(!error);

  Storage1D<char> vtype(nVars,GRB_CONTINUOUS);
  
  error = GRBaddvars(grb_model,nVars,0,NULL,NULL,NULL,cost.direct_access(),var_lb.direct_access(),
                     var_ub.direct_access(),vtype.direct_access(),NULL);
  assert(!error);

  error = GRBupdatemodel(grb_model);
  assert(!error);

  for (uint c=0; c < nConstraints; c++) {

    //     if ((c % 250) == 0)
    //       std::cerr << "c: " << c << std::endl;

    //     std::string s = "c" + toString(c);
    //     char cstring [256];
    //     for (uint i=0; i < s.size(); i++)
    //       cstring[i] = s[i];
    //     cstring[s.size()] = 0;

    error = GRBaddconstr(grb_model, row_start[c+1]-row_start[c], ((int*) lp_descr.col_indices()) + row_start[c], 
                         lp_descr.value() + row_start[c], GRB_EQUAL, rhs[c], NULL);

    if (error) {
      std::cerr << "abs_con_offs: " << abs_con_offs << std::endl;

      std::cerr << "error for con " << c << ": " << error << " " << GRBgeterrormsg(grb_env) << std::endl;
      error = GRBupdatemodel(grb_model);
      assert(!error);

      for (uint k=row_start[c]; k < row_start[c+1]; k++) {

        std::cerr << lp_descr.col_indices()[k] << ", ";
      }
      std::cerr << std::endl;

      error = GRBaddconstr(grb_model, row_start[c+1]-row_start[c], ((int*) lp_descr.col_indices()) + row_start[c], 
                           lp_descr.value() + row_start[c], GRB_EQUAL, rhs[c], NULL);
      assert(!error);
    }
  }

  //   Storage1D<char> ctype(nConstraints,GRB_EQUAL);
  //   GRBaddconstrs(grb_model, nConstraints, row_start[nConstraints], (int*) row_start.direct_access(),
  // 		(int*) lp_descr.col_indices(), lp_descr.value(), ctype.direct_access(), rhs.direct_access(), NULL);

  //error = GRBupdatemodel(grb_model);
  //assert(!error);
  
  /* Optimize model */
  error = GRBoptimize(grb_model);
  assert(!error);

  GRBfreemodel(grb_model);
  GRBfreeenv(grb_env);
#endif

#ifdef USE_MOSEK

  MSKrescodee  r;

  MSKenv_t     env  = NULL;
  MSKtask_t    task = NULL; 
  
  /* Create the mosek environment. */
  r = MSK_makeenv(&env,NULL,NULL,NULL,NULL);
  
  /* Directs the env log stream to the 'printstr' function. */
  if ( r==MSK_RES_OK )
    MSK_linkfunctoenvstream(env,MSK_STREAM_LOG,NULL,printstr);

  /* Initialize the environment. */
  if ( r==MSK_RES_OK )
    r = MSK_initenv(env);
  
  if ( r==MSK_RES_OK )
    {
      /* Create the optimization task. */
      r = MSK_maketask(env,nConstraints,nVars,&task);    
  
      //r =  MSK_putintparam(task, MSK_IPAR_OPTIMIZER,MSK_OPTIMIZER_DUAL_SIMPLEX);


      /* Directs the log task stream to the 'printstr' function. */
      if ( r==MSK_RES_OK )
        MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);

      /* Give MOSEK an estimate of the size of the input data. 
         This is done to increase the speed of inputting data. 
         However, it is optional. */
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumvar(task,nVars);
  
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumcon(task,nConstraints);
    
      if (r == MSK_RES_OK)
        r = MSK_putmaxnumanz(task,lp_descr.nEntries());

      /* Append 'NUMCON' empty constraints.
         The constraints will initially have no bounds. */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_CON,nConstraints);

      /* Append 'NUMVAR' variables.
         The variables will initially be fixed at zero (x=0). */
      if ( r == MSK_RES_OK )
        r = MSK_append(task,MSK_ACC_VAR,nVars);

      // set the cost function
      for(uint j=0; j< nVars && r == MSK_RES_OK; ++j)
        {
          /* Set the linear term c_j in the objective.*/  
          if(r == MSK_RES_OK)
            r = MSK_putcj(task,j,cost[j]);
        }

      for(uint j=0; j< nVars && r == MSK_RES_OK; ++j) {

        double lower = var_lb[j];
        double upper = var_ub[j];
      
        MSKboundkeye key = MSK_BK_RA;

        if (lower == upper) 
          key = MSK_BK_FX;

        r = MSK_putbound(task,
                         MSK_ACC_VAR, /* Put bounds on variables.*/
                         j,           /* Index of variable.*/
                         key,      /* Bound key.*/
                         lower,      /* Numerical value of lower bound.*/
                         upper);     /* Numerical value of upper bound.*/
      }

      //set the constraint right-hand-sides
      for (uint c=0; c < nConstraints; c++) {

        double lower = rhs[c];
        double upper = rhs[c];
      
        MSKboundkeye key = MSK_BK_RA;

        if (lower == upper) 
          key = MSK_BK_FX;

        r = MSK_putbound(task,
                         MSK_ACC_CON, /* Put bounds on variables.*/
                         c,           /* Index of variable.*/
                         key,      /* Bound key.*/
                         lower,      /* Numerical value of lower bound.*/
                         upper);     /* Numerical value of upper bound.*/
      }
 
      for (uint c=0; c < nConstraints; c++) {

        r = MSK_putavec(task,MSK_ACC_CON, c, row_start[c+1]-row_start[c],
                        ((int*) lp_descr.col_indices()) + row_start[c],
                        lp_descr.value() + row_start[c]);
      }

      if (r == MSK_RES_OK)
        r = MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE);
    }
  else {
    INTERNAL_ERROR << "failed to created MOSEK environment" << std::endl;
    exit(1);
  }

  
  MSKrescodee trmcode;
  
  //TRIAL
  //   for (uint v=0; v < nVars; v++)
  //     r = MSK_putvartype(task,v,MSK_VAR_TYPE_INT);
  //END_TRIAL

  /* Choose dual simplex */
  //   if ( r==MSK_RES_OK )              
  //     r =  MSK_putintparam(task,
  //  			 MSK_IPAR_OPTIMIZER,MSK_OPTIMIZER_DUAL_SIMPLEX);
  timeval tStartMosek, tEndMosek;

  gettimeofday(&tStartMosek,0);

  //   /* Run optimizer */
  r = MSK_optimizetrm(task,&trmcode);

  gettimeofday(&tEndMosek,0);
  std::cerr << "mosek needed " << diff_seconds(tEndMosek,tStartMosek) << " seconds to compute a solution" << std::endl;
  
  //   /* Print a summary containing information
  //      about the solution for debugging purposes. */
  //   MSK_solutionsummary (task,MSK_STREAM_LOG);

#endif

  lp_descr.reset(0);

  //error = lpSolver.dual();
  //error = lpSolver.primal();

  //lpSolver.initialSolve();

  ClpSolve solve_options;
  solve_options.setSolveType(ClpSolve::useDual);
  //solve_options.setSolveType(ClpSolve::useBarrier);
  //solve_options.setPresolveType(ClpSolve::presolveNumber,5);
  lpSolver.initialSolve(solve_options);

  double* solution =  lpSolver.primalColumnSolution();

  
  gettimeofday(&tEndCLP,0);
  std::cerr << "CLP-time: " << diff_seconds(tEndCLP,tStartCLP) << " seconds, return status: " << error << std::endl;

  Math2D::Matrix<float> integrality_matrix(xDim,yDim,255.0);

  //extract the optimal displacements
  uint nNonIntegral = 0;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint base = (y*xDim+x)*nLabels;
      
      double max_var = -1.0;
      double sum = 0.0;

      int arg_max_x = MAX_UINT;
      int arg_max_y = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          uint v = base + ly*nHorLabels+lx;
          double val = solution[v];
	  
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

      assert(fabs(sum-1.0) < 1e-3);

      integrality_matrix(x,y) = max_var*255.0;

      if (max_var < 0.99) {
        nNonIntegral++;
        //integrality_matrix(x,y) = 0;
        //std::cerr << "var value " << max_var << std::endl;
      }
    }
  }
  
  integrality_matrix.savePGM("int.pgm",255);
  
  std::cerr << nNonIntegral << " out of " << (yDim*xDim) << " pixels have non-integral label variables" << std::endl;

  double energy = 0.0;
  for (uint v=0; v < nVars; v++)
    energy += cost[v] * solution[v];

#ifdef USE_MOSEK

  MSK_deletetask(&task); 
  MSK_deleteenv(&env);
#endif

  return energy;
#else
  return 0.0;
#endif
}

/************************************************************************************************************/

double lp_motion_estimation_standard_relax(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                           int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                           uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

#ifdef HAS_CLP
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

  const uint nLabelVars = xDim*yDim*nLabels;

  const uint nStraightTransitions = yDim * (xDim-1) + (yDim-1) * xDim;
  uint nDiagTransitions = 0;
  if (neighborhood == 8) {

    nDiagTransitions += 2*(xDim-1)*(yDim-1);
  }

  const uint nTransitions = nStraightTransitions + nDiagTransitions;

  uint nAbsVars = 4*nTransitions;

  const uint nVars = nLabelVars + nAbsVars;
  const uint abs_var_offs = nLabelVars;

  const uint nConstraints = xDim*yDim // unity constraints
    + 2*nTransitions; //constraints to model the absolutes (in u and v separately)

  uint abs_con_offs = xDim*yDim;

  Math1D::NamedVector<double> cost(nVars,0.0,MAKENAME(cost));

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint base_id = (y*xDim+x)*nLabels;

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

          cost[base_id + ly*nHorLabels+lx] = disp_cost;
        }
      }
    }
  }

  for (uint v=0; v < 4*nStraightTransitions; v++)
    cost[abs_var_offs + v] = lambda;
  
  if (neighborhood == 8) {
    const uint diag_abs_var_offs = abs_var_offs + 4*nStraightTransitions;
    double diag_lambda = lambda / sqrt(2.0);
    
    for (uint v=0; v < 4*nDiagTransitions; v++)
      cost[diag_abs_var_offs+v] = diag_lambda;
  }

  Math1D::NamedVector<double> var_lb(nVars,0.0,MAKENAME(var_lb));
  Math1D::NamedVector<double> var_ub(nVars,10000.0,MAKENAME(var_ub));

  for (uint v=0; v < nLabelVars; v++)
    var_ub[v] = 1.0;

  Math1D::NamedVector<double> rhs(nConstraints,0.0,MAKENAME(rhs));
  for (uint i=0; i < xDim*yDim; i++)
    rhs[i] = 1.0;

  uint nMatrixEntries = nLabelVars // for the unity constraints
    + 8*nTransitions*nLabels + nAbsVars; //for the absolutes

  SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);

  //set up unity constraints (label vars for a pixel must sum to 1.0)
  for (uint i=0; i < yDim*xDim; i++) {
    
    for (uint j=i*nLabels; j < (i+1)*nLabels; j++)
      lp_descr.add_entry(i,j,1.0);
  }

  assert(neighborhood == 4);

  uint trans_num = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint pixel_id = y*xDim+x;

      if (x > 0) {

        uint neighbor_id = y*xDim+x-1;

        uint base_row = abs_con_offs + trans_num*2;

        for (uint lv=0; lv < nVertLabels; lv++) {
          for (uint lh=0; lh < nHorLabels; lh++) {

            lp_descr.add_entry(base_row, pixel_id*nLabels+lv*nHorLabels+lh, lh);
            lp_descr.add_entry(base_row, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lh);

            lp_descr.add_entry(base_row+1, pixel_id*nLabels+lv*nHorLabels+lh, lv);
            lp_descr.add_entry(base_row+1, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lv);
          }
        }

        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num, 1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+1, -1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+2, 1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+3, -1.0);

        trans_num++;
      }

      if (y > 0) {

        uint neighbor_id = y*xDim+x-xDim;

        uint base_row = abs_con_offs + trans_num*2;

        for (uint lv=0; lv < nVertLabels; lv++) {
          for (uint lh=0; lh < nHorLabels; lh++) {

            lp_descr.add_entry(base_row, pixel_id*nLabels+lv*nHorLabels+lh, lh);
            lp_descr.add_entry(base_row, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lh);

            lp_descr.add_entry(base_row+1, pixel_id*nLabels+lv*nHorLabels+lh, lv);
            lp_descr.add_entry(base_row+1, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lv);
          }
        }

        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num, 1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+1, -1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+2, 1.0);
        lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+3, -1.0);

        trans_num++;
      }
    }
  }

  if (neighborhood >= 8) {

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        uint pixel_id = y*xDim+x;
	
        if (x > 0 && y > 0) {
	  
          uint neighbor_id = (y-1)*xDim+x-1;
	  
          uint base_row = abs_con_offs + trans_num*2;
	  
          for (uint lv=0; lv < nVertLabels; lv++) {
            for (uint lh=0; lh < nHorLabels; lh++) {
	      
              lp_descr.add_entry(base_row, pixel_id*nLabels+lv*nHorLabels+lh, lh);
              lp_descr.add_entry(base_row, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lh);
	      
              lp_descr.add_entry(base_row+1, pixel_id*nLabels+lv*nHorLabels+lh, lv);
              lp_descr.add_entry(base_row+1, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lv);
            }
          }

          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num, 1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+1, -1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+2, 1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+3, -1.0);
	  
          trans_num++;
        }
	
        if (x+1 < xDim && y > 0) {
	  
          uint neighbor_id = y*xDim+x+1-xDim;

          uint base_row = abs_con_offs + trans_num*2;
	  
          for (uint lv=0; lv < nVertLabels; lv++) {
            for (uint lh=0; lh < nHorLabels; lh++) {
	      
              lp_descr.add_entry(base_row, pixel_id*nLabels+lv*nHorLabels+lh, lh);
              lp_descr.add_entry(base_row, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lh);
	      
              lp_descr.add_entry(base_row+1, pixel_id*nLabels+lv*nHorLabels+lh, lv);
              lp_descr.add_entry(base_row+1, neighbor_id*nLabels+lv*nHorLabels+lh, -1.0 *lv);
            }
          }
	  
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num, 1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+1, -1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+2, 1.0);
          lp_descr.add_entry(base_row, abs_var_offs + 4*trans_num+3, -1.0);
	  
          trans_num++;
        }
      }
    }
  }

  //timeval tStartCLP, tEndCLP;  

  CoinPackedMatrix coinMatrix(false,(int*) lp_descr.row_indices(),(int*) lp_descr.col_indices(),
                              lp_descr.value(),lp_descr.nEntries());

  ClpSimplex lpSolver;
  lpSolver.loadProblem (coinMatrix, var_lb.direct_access(), var_ub.direct_access(),   
                        cost.direct_access(), rhs.direct_access(), rhs.direct_access());

  coinMatrix.cleanMatrix();
  lp_descr.reset(0);

  lpSolver.dual();

  double energy = lpSolver.getObjValue();

  const double* solution =  lpSolver.primalColumnSolution();

  //extract solution
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint base = (y*xDim+x)*nLabels;
      
      double max_var = -1.0;
      double sum = 0.0;

      int arg_max_x = MAX_UINT;
      int arg_max_y = MAX_UINT;

      for (uint lx=0; lx < nHorLabels; lx++) {
        for (uint ly=0; ly < nVertLabels; ly++) {

          uint v = base + ly*nHorLabels+lx;
          double val = solution[v];
	  
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

      assert(fabs(sum-1.0) < 1e-3);
    }
  }
  
  return energy; 
#else
  return 0.0;
#endif  
}


/************************************************************************************************************/

double conv_lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
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

  const uint nLabelVars = xDim*yDim*nLabels;

  const uint nHorLevelVars = xDim*yDim*nHorLabels;
  const uint nVertLevelVars = xDim*yDim*nVertLabels;

  const uint nStraightTransitions = yDim * (xDim-1) + (yDim-1) * xDim;
  uint nDiagTransitions = 0;
  if (neighborhood == 8) {

    nDiagTransitions += 2*(xDim-1)*(yDim-1);
  }

  const uint nTransitions = nStraightTransitions + nDiagTransitions;

  uint nAbsVars = 2*nTransitions*(nHorLabels + nVertLabels);

  const uint nVars = nLabelVars + nHorLevelVars + nVertLevelVars + nAbsVars;

  const uint hlevel_var_offs = nLabelVars;
  const uint vlevel_var_offs = hlevel_var_offs + nHorLevelVars; 
  const uint abs_var_offs = nLabelVars + nHorLevelVars + nVertLevelVars;

  const uint nConstraints = xDim*yDim*(nHorLabels + nVertLabels)  //level constraints
    + nTransitions*(nHorLabels + nVertLabels); //constraints to model the absolutes

  Math1D::NamedVector<double> cost(nVars,0.0,MAKENAME(cost));

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint base_id = (y*xDim+x)*nLabels;

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

          cost[base_id + ly*nHorLabels+lx] = disp_cost;
        }
      }
    }
  }

  for (uint v=0; v < 2*nStraightTransitions*(nHorLabels+nVertLabels); v++)
    cost[abs_var_offs + v] = lambda;
  
  if (neighborhood == 8) {
    const uint diag_abs_var_offs = abs_var_offs + 2*nStraightTransitions*(nHorLabels+nVertLabels);
    double diag_lambda = lambda / sqrt(2.0);

    for (uint v=0; v < 2*nDiagTransitions*(nHorLabels+nVertLabels); v++)
      cost[diag_abs_var_offs+v] = diag_lambda;
  }

  uint nMatrixEntries = nHorLevelVars*(nVertLabels+1) + nVertLevelVars*(nHorLabels+1) // for the level constraints
    + 8*nTransitions*nLabels; //for the absolutes

  SparseMatrixDescription<char> lp_descr(nMatrixEntries, nConstraints, nVars);

  //set up horizontal level constraints
  const uint hlevel_con_offs = 0;
  for (uint i=0; i < yDim*xDim; i++) {

    for (uint lx = 0; lx < nHorLabels; lx++) {
      
      const uint row = hlevel_con_offs + i*nHorLabels + lx;

      lp_descr.add_entry(row, hlevel_var_offs+i*nHorLabels+lx, -1);
      if (lx > 0)
        lp_descr.add_entry(row, hlevel_var_offs+i*nHorLabels+lx-1, 1);

      for (uint ly = 0; ly < nVertLabels; ly++) {

        lp_descr.add_entry(row, i*nLabels + ly*nHorLabels+lx , 1);
      }	
    }
  }
  
  //set up vertical level constraints
  const uint vlevel_con_offs = hlevel_con_offs + xDim*yDim*nHorLabels;
  for (uint i=0; i < yDim*xDim; i++) {

    for (uint ly = 0; ly < nVertLabels; ly++) {

      const uint row = vlevel_con_offs + i*nVertLabels + ly;

      lp_descr.add_entry(row, vlevel_var_offs+i*nVertLabels+ly, -1);
      if (ly > 0)
        lp_descr.add_entry(row, vlevel_var_offs+i*nVertLabels+ly-1, 1);

      for (uint lx=0; lx < nHorLabels; lx++) {

        lp_descr.add_entry(row, i*nLabels + ly*nHorLabels+lx, 1);
      }
    }
  }

  //set up the constraints to model the absolutes of the horizontal and of the vertical displacement
  const uint abs_con_offs = vlevel_con_offs + yDim*xDim*nVertLabels;

  uint trans_num = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint pixel_id = y*xDim+x;

      if (x+1 < xDim) {

        const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);

        const uint neighbor_id = y*xDim+x+1;

        //horizontal displacement
        for (uint lx=0; lx < nHorLabels; lx++) {

          const uint row = row_base + lx;
	  
          lp_descr.add_entry(row, abs_var_base + 2*lx, 1);
          lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1);
	  
          lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1);
          lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1);	  
        }

        //vertical displacement
        for (uint ly=0; ly < nVertLabels; ly++) {

          const uint row = row_base + nHorLabels + ly;

          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1);
          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1);
	  
          lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1);
          lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1);
        }

        trans_num++;
      }

      if (y+1 < yDim) {

        const uint row_base = abs_con_offs + trans_num*(nHorLabels+nVertLabels);
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels+nVertLabels);

        const uint neighbor_id = (y+1)*xDim+x;

        //horizontal displacement
        for (uint lx=0; lx < nHorLabels; lx++) {

          const uint row = row_base + lx;
	  
          lp_descr.add_entry(row, abs_var_base + 2*lx, 1);
          lp_descr.add_entry(row, abs_var_base + 2*lx+1, -1);
	  
          lp_descr.add_entry(row, hlevel_var_offs + pixel_id*nHorLabels+lx, 1);
          lp_descr.add_entry(row, hlevel_var_offs + neighbor_id*nHorLabels+lx, -1);	  
        }

        //vertical displacement
        for (uint ly=0; ly < nVertLabels; ly++) {

          const uint row = row_base + nHorLabels + ly;

          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly, 1);
          lp_descr.add_entry(row, abs_var_base + 2*nHorLabels + 2*ly+1, -1);
	  
          lp_descr.add_entry(row, vlevel_var_offs + pixel_id*nVertLabels+ly, 1);
          lp_descr.add_entry(row, vlevel_var_offs + neighbor_id*nVertLabels+ly, -1);
        }

        trans_num++;
      }
    }
  }
  
  if (neighborhood == 8) {
    TODO("8-neighborhood");
  }

  //optimize
  std::cerr << nVars << " variables, " << nConstraints << " constraints" << std::endl;

  Math1D::Vector<double> conv_solution(nVars,0.0 / nLabels);
  for (uint v=0; v < xDim*yDim*nLabels; v++)
    conv_solution[v] = 1.0 / nLabels;

  uint* simplex_starts = new uint[xDim*yDim+1];

  for (uint i=0; i <= xDim*yDim; i++)
    simplex_starts[i] = i*nLabels;

  eq_and_simplex_constrained_lp_solve_auglagr_nesterov(nVars, nConstraints, cost.direct_access(), 0, 0,
                                                       lp_descr, 0, xDim*yDim, simplex_starts,
                                                       conv_solution.direct_access());

  delete[] simplex_starts;

  Math2D::Matrix<uint> labeling(xDim,yDim,0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      int min_l = MAX_UINT;
      double max_val = 0.0;

      for (uint l=0; l < nLabels; l++) {

        if (conv_solution[(y*xDim+x)*nLabels + l] > max_val) {
          max_val = conv_solution[(y*xDim+x)*nLabels + l];
          min_l = l;
        }
      }

      labeling(x,y) = min_l;

      velocity(x,y,0) = (min_l % nHorLabels);// + min_x_disp;
      velocity(x,y,1) = (min_l / nHorLabels);// + min_y_disp;

      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 

    }
  }

  return 0.0; //TODO: calculate cost
}


/************************************************************************************************************/
//typedef float Real;
typedef double Real;

double implicit_conv_lp_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                          int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                          uint neighborhood, uint spacing, double lambda, 
                                          Math3D::Tensor<double>& velocity, Math2D::Matrix<uint>* given_labeling) {

  //const bool explicit_level_vars = true; //experimentally, it is better to include the level variables explicitly

  Real inv_spacing = 1.0 / spacing;

  double org_lambda = lambda;

  lambda *= inv_spacing;

  //std::cerr << "using lambda " << lambda << std::endl;

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());
  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp + 1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint nHorLabels_m1 = nHorLabels - 1;
  const uint nVertLabels_m1 = nVertLabels - 1;

  //std::cerr << "nLabels: " << nLabels << std::endl;

  const uint nLabelVars = xDim*yDim*nLabels;

  const uint nHorLevelVars = xDim*yDim*(nHorLabels_m1);
  const uint nVertLevelVars = xDim*yDim*(nVertLabels_m1);


  const uint nStraightTransitions = yDim * (xDim-1) + (yDim-1) * xDim;
  uint nDiagTransitions = 0;
  if (neighborhood == 8) {

    nDiagTransitions += 2*(xDim-1)*(yDim-1);
  }

  const uint nTransitions = nStraightTransitions + nDiagTransitions;
  const uint nAbsVars = 2*nTransitions*(nHorLabels_m1 + nVertLabels_m1);

  const uint nVars = nLabelVars + nHorLevelVars + nVertLevelVars + nAbsVars;

  const uint hlevel_var_offs = nLabelVars;
  const uint vlevel_var_offs = hlevel_var_offs + nHorLevelVars; 
  const uint abs_var_offs = nLabelVars + nHorLevelVars + nVertLevelVars;

  const uint nConstraints = xDim*yDim*(nHorLabels_m1 + nVertLabels_m1)  //level constraints
    + nTransitions*(nHorLabels_m1 + nVertLabels_m1); //constraints to model the absolutes


  const uint hlevel_con_offs = 0;
  const uint vlevel_con_offs = hlevel_con_offs + xDim*yDim*nHorLabels_m1;
  const uint abs_con_offs = vlevel_con_offs + yDim*xDim*nVertLabels_m1;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  Math2D::Matrix<uint> labeling(xDim,yDim,0);

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
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

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

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          Real hyp_cost = label_cost(x,y,ly*nHorLabels+lx);
	
          if (hyp_cost > best_label_cost) { // ; + smoothWorstCost(lx,ly)) {

            label_cost(x,y,ly*nHorLabels+lx) += 1000.0;
            nVarExcluded++;
          }
        }
      }
    }
  }

  std::cerr << "excluded " << nVarExcluded << " vars. That's " 
            << (100.0*((double) nVarExcluded) / ((double) nLabels*xDim*yDim)) << "%." << std::endl;
#endif



  const float* cost = label_cost.direct_access();

  Math1D::Vector<Real> solution(nVars,0.0);

  if (given_labeling == 0) {
    for (uint v=0; v < nLabelVars; v++)
      solution[v] = 1.0 / nLabels;
  }
  else {

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint i=y*xDim+x;
        solution[i*nLabels + (*given_labeling)(x,y)] = 1.0;
      }
    }
  }

#if 0
  //initialization by expansion moves

  std::cerr << "-------- initializing by expansion moves" << std::endl;
  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling, 3);
  
  solution.set_constant(0.0);
  for (uint y=0; y < yDim; y++)
    for (uint x=0; x < xDim; x++)
      solution[(y*xDim+x)*nLabels + labeling(x,y)] = 1.0;
#endif

  /*** calculate marginals ****/

  //if (explicit_level_vars) {
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {
      
      const uint i=y*xDim+x;
      
      Real sum = 0.0;
      
      //update h-marginals
      for (uint lh = 0; lh < nHorLabels_m1; lh++) {
        for (uint lv = 0; lv < nVertLabels; lv++) {
          sum += solution[i*nLabels + lv*nHorLabels+lh];
        }

        assert(sum <= 1.01);	
        solution[hlevel_var_offs + i*nHorLabels_m1 + lh] = sum;
      }
      
      sum = 0.0;
      
      //update v-marginals
      for (uint lv = 0; lv < nVertLabels_m1; lv++) {
        for (uint lh = 0; lh < nHorLabels; lh++) {
          sum += solution[i*nLabels + lv*nHorLabels+lh];
        }

        assert(sum <= 1.01);
	
        solution[vlevel_var_offs + i*nVertLabels_m1 + lv] = sum;
      }  
    }
  }
  //   }
  
  /**** calculate absolute differences ****/
  uint trans_num = 0;
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      uint pixel_id = y*xDim+x;
      
      if (x+1 < xDim) {
	
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
        const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	
        const uint neighbor_id = pixel_id+1;
	
        //horizontal displacement
        for (uint lx=0; lx < nHorLabels_m1; lx++) {
	  
          Real diff = solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
            - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
	  
          if (diff >= 0.0)
            solution[abs_var_base + 2*lx+1] = diff;
          else
            solution[abs_var_base + 2*lx] = -diff;
        }
	
        //vertical displacement
        for (uint ly=0; ly < nVertLabels_m1; ly++) {

          Real diff = solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
            - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

          if (diff >= 0.0)
            solution[v_abs_var_base + 2*ly+1] = diff;
          else
            solution[v_abs_var_base + 2*ly] = -diff;
        }
	
        trans_num++;
      }

      if (y+1 < yDim) {
	
        const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
        const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	
        const uint neighbor_id = pixel_id + xDim; 
	
        //horizontal displacement
        for (uint lx=0; lx < nHorLabels_m1; lx++) {
	  
          Real diff = solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx]
            - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

          if (diff >= 0.0) 
            solution[abs_var_base + 2*lx+1] = diff;
          else
            solution[abs_var_base + 2*lx] = -diff;
        }
	
        //vertical displacement
        for (uint ly=0; ly < nVertLabels_m1; ly++) {
	  
          Real diff = solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly]
            - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

          if (diff >= 0.0)
            solution[v_abs_var_base + 2*ly+1] = diff;
          else
            solution[v_abs_var_base + 2*ly] = -diff;
        }
	
        trans_num++;
      }      
    }
  }
  
  if (neighborhood >= 8) {

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint pixel_id = y*xDim+x;
	
        if (x>0 && y >0) {
	  
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
          const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

          const uint neighbor_id = pixel_id-1-xDim;
	  
          //horizontal displacement
          for (uint lx=0; lx < nHorLabels_m1; lx++) {
	    
            Real diff = solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
              - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

            if (diff >= 0.0)
              solution[abs_var_base + 2*lx+1] = diff;
            else
              solution[abs_var_base + 2*lx] = -diff;
          }

          //vertical displacement
          for (uint ly=0; ly < nVertLabels_m1; ly++) {
	    
            Real diff = solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
              - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

            if (diff >= 0.0)
              solution[v_abs_var_base + 2*ly+1] = diff;
            else
              solution[v_abs_var_base + 2*ly] = -diff;
          }
	      
          trans_num++;
        }

        if (x>0 && y+1 < yDim) {
	    
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
          const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
          const uint neighbor_id = pixel_id-1+xDim;

          //horizontal displacement
          for (uint lx=0; lx < nHorLabels_m1; lx++) {
	    
            Real diff = solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
              - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

            if (diff >= 0.0)
              solution[abs_var_base + 2*lx+1] = diff;
            else
              solution[abs_var_base + 2*lx] = -diff;
          }
	  
          //vertical displacement
          for (uint ly=0; ly < nVertLabels_m1; ly++) {
	    
            Real diff = solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
              - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

            if (diff >= 0.0)
              solution[v_abs_var_base + 2*ly+1] = diff;
            else
              solution[v_abs_var_base + 2*ly] = -diff;
          }
	      
          trans_num++;
        }
      }
    }
  }

  /**** now start the augmented Lagrangian - Nesterov scheme *****/

  Math1D::NamedVector<Real> aux_solution(nVars,MAKENAME(aux_solution));

  Math1D::NamedVector<Real> lagrange_multiplier(nConstraints,0.0,MAKENAME(lagrange_multiplier));

  const uint diag_abs_var_offs = abs_var_offs + 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1);
  const Real diag_lambda = lambda / sqrt(2.0);

  Real penalty = 100.0;

  Real last_energy = 1e50;

  Math1D::NamedVector<Real> ax(nConstraints,0.0,MAKENAME(ax));

#ifdef EXPLICIT_GRADIENT
  Math1D::Vector<Real> grad(nVars);
#endif

  //const uint nOuterIter = 30;
  const uint nOuterIter = 6;


  for (uint outer_iter = 1; outer_iter <= nOuterIter; outer_iter++) {

    double best_lower_bound = -MAX_DOUBLE;

    const Real increase_factor = (neighborhood >= 8) ? 1.25 : 1.5;

    if (outer_iter != 1) {
      penalty *= increase_factor;
    }

    std::cerr << "############# outer iter " << outer_iter << ",  penalty " << penalty << std::endl;

    last_energy = 1e50;

    double prev_t = 1.0;

    Real alpha = 1e-1 / penalty;

    uint iter_since_restart = 0;

    //const uint iter_limit = (outer_iter == 1) ? 500 : 50;
    //const uint iter_limit = 350 + 5*(outer_iter-1);

    const uint iter_limit = 200;
    //const uint iter_limit = 750;
    //const uint iter_limit = (outer_iter != nOuterIter) ? 500 : 15000;

    for (uint v=0; v < nVars; v++)      
      aux_solution[v] = solution[v];

    double save_energy = 1e50;
    double energy_landmark = 1e50;

    //const float cutoff_threshold = (outer_iter==nOuterIter) ? 0.05 : 0.5;
    const double cutoff_threshold = (outer_iter==nOuterIter) ? 1e-5 : 1e-4;

    for (uint iter = 1; iter <= iter_limit; iter++) {

      timeval tStartEnergy, tEndEnergy;
      gettimeofday(&tStartEnergy,0);

      /*** 1. calculate current energy ****/
      Real energy = 0.0;
      for (uint v=0; v < xDim*yDim*nLabels; v++)
        energy += solution[v] * cost[v];
      //std::cerr << "data energy: " << energy << std::endl;
      Real sum = 0.0;
      for (uint v=0; v < 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
        sum += solution[abs_var_offs + v];
      energy += lambda* sum;

      if (neighborhood >= 8) {
        sum = 0.0;
        for (uint v=0; v < 2*nDiagTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
          sum += solution[diag_abs_var_offs+v];

        energy += diag_lambda * sum;
      }

      //std::cerr << "energy without Lagrangian terms: " << energy << std::endl;

      ax.set_constant(0.0);

      /*** calculate marginals ****/

#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint i=0; i < xDim*yDim; i++) {

        const uint base = i*nLabels;

        const uint cur_hlevel_offs = hlevel_var_offs + i*nHorLabels_m1;
        const uint cur_vlevel_offs = vlevel_var_offs + i*nVertLabels_m1;
	
        const uint cur_hcon_offs = hlevel_con_offs + i*nHorLabels_m1;
        const uint cur_vcon_offs = vlevel_con_offs + i*nVertLabels_m1;


        //update h-marginals
        for (uint lh = 0; lh < nHorLabels_m1; lh++) {

          Real sum = 0.0;
          for (uint lv = 0; lv < nVertLabels; lv++)
            sum += solution[base + lv*nHorLabels + lh]; 
	  
          if (lh > 0)
            sum += solution[cur_hlevel_offs + lh-1];
	  
          //if (explicit_level_vars) {

          ax[cur_hcon_offs + lh] += sum - solution[cur_hlevel_offs + lh];

          //	  }
          // 	  else {
          // 	    assert(sum >= 0.0);
          // 	    assert(sum <= 1.01);

          // 	    solution[cur_level_offs + lh] = sum;
          // 	  }
        }
	
        //update v-marginals
        for (uint lv = 0; lv < nVertLabels_m1; lv++) {
	  
          Real sum  = 0.0;
          for (uint lh = 0; lh < nHorLabels; lh++) 
            sum += solution[base + lv*nHorLabels + lh];

          if (lv > 0)
            sum += solution[cur_vlevel_offs + lv-1];
	  
          //if (explicit_level_vars) {

          ax[cur_vcon_offs + lv] += sum - solution[cur_vlevel_offs + lv];

          //        }
          // 	  else {

          // 	    assert(sum >= 0.0);
          // 	    assert(sum <= 1.01);

          // 	    solution[cur_level_offs + lv] = sum;
          // 	  }
        }  
      }

      /** calculate differences **/
      
      uint trans_num = 0;
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {
	  
          const uint pixel_id = y*xDim+x;
	  
          if (x+1 < xDim) {
	    
            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

            const uint neighbor_id = pixel_id+1;

            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
	      
              ax[row] = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              //const uint row = row_base + nHorLabels + ly;
              const uint row = row_base + nHorLabels_m1 + ly;

              ax[row] = solution[v_abs_var_base + 2*ly] 
                - solution[v_abs_var_base + 2*ly+1]
                + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
            }
	    
            trans_num++;
          }

          if (y+1 < yDim) {

            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
            const uint neighbor_id = pixel_id + xDim; 
	    
            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
	      
              ax[row] = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx]
                - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;	

              ax[row] = solution[v_abs_var_base + 2*ly]
                - solution[v_abs_var_base + 2*ly+1]
                + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly]
                - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
      	    }
	    
            trans_num++;
          }
        }
      }

      if (neighborhood >= 8) {

        assert(2*(nHorLabels_m1+nVertLabels_m1)*trans_num + abs_var_offs == diag_abs_var_offs);

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const uint pixel_id = y*xDim+x;
	    
            if (x>0 && y >0) {
	    
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

              const uint neighbor_id = pixel_id-1-xDim;

              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
                const uint row = row_base + lx;
		
                ax[row] = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                  + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                  - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                //const uint row = row_base + nHorLabels + ly;
                const uint row = row_base + nHorLabels_m1 + ly;
		
                ax[row] = solution[v_abs_var_base + 2*ly] 
                  - solution[v_abs_var_base + 2*ly+1]
                  + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                  - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
              }
	      
              trans_num++;
            }

            if (x>0 && y+1 < yDim) {
	    
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
              const uint neighbor_id = pixel_id-1+xDim;

              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
                const uint row = row_base + lx;
		
                ax[row] = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                  + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                  - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
		
                ax[row] = solution[v_abs_var_base + 2*ly] 
                  - solution[v_abs_var_base + 2*ly+1]
                  + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                  - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
              }
	      
              trans_num++;
            }
          }
        }
      }

      //std::cerr << "inter energy: " << energy << std::endl;      

      Real penalty_energy = 0.0;

      for (uint c=0; c < nConstraints; c++) {

        const Real temp = ax[c];

        energy += lagrange_multiplier[c] * temp;
        penalty_energy += temp*temp;
      }

      energy += 0.5*penalty*penalty_energy;

      if ((iter % 10) == 0) {

        double lower_bound = energy;

        //std::cerr << "calc lower bound" << std::endl;

        Math1D::NamedVector<Real> cur_grad(nVars,MAKENAME(cur_grad));
	
        ax *= penalty;
        ax += lagrange_multiplier;
	
        for (uint v=0; v < nLabelVars; v++)
          cur_grad[v] = cost[v];
        for (uint v=0; v < xDim*yDim*(nHorLabels_m1 + nVertLabels_m1); v++)
          cur_grad[hlevel_var_offs + v] = 0.0;
        for (uint v=0; v < 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
          cur_grad[abs_var_offs + v] = lambda;
      
        if (neighborhood == 8) {
	  
          for (uint v=0; v < 2*nDiagTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
            cur_grad[diag_abs_var_offs+v] = diag_lambda;
        }

        /*** calculate the gradient of the current solution *****/

        //std::cerr << "A" << std::endl;
		
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
	    
            const uint base = (y*xDim+x)*nLabels;
            const uint i=y*xDim+x;
	    
            const uint hlevel_base = hlevel_var_offs + i*nHorLabels_m1;
            const uint vlevel_base = vlevel_var_offs + i*nVertLabels_m1;
	    
            const uint cur_hcon_offs = hlevel_con_offs + i*nHorLabels_m1;

            //update h-marginals
            Real last_ax = 0.0;

            for (int lh = nHorLabels_m1-1; lh >=0; lh--) {
	    
              const Real cur_ax = ax[cur_hcon_offs + lh];

              for (uint lv = 0; lv < nVertLabels; lv++) {
                cur_grad[base + lv*nHorLabels + lh] += cur_ax;
              }

              cur_grad[hlevel_base + lh] += last_ax - cur_ax;
	      
              last_ax = cur_ax;
            }
	
            //update v-marginals
            last_ax = 0.0;
	    
            const uint cur_vcon_offs = vlevel_con_offs + i*nVertLabels_m1;
	  
            for (int lv = nVertLabels_m1-1; lv >= 0; lv--) {
	      
              const Real cur_ax = ax[cur_vcon_offs + lv];

              for (uint lh = 0; lh < nHorLabels; lh++) {
                cur_grad[base + lv*nHorLabels + lh] += cur_ax;
              }
	    
              cur_grad[vlevel_base + lv] += last_ax - cur_ax;

              last_ax = cur_ax;
            }
          }  
        }
	
        /** calculate differences **/
      
        //std::cerr << "B" << std::endl;

        trans_num = 0;
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
	    
            const uint pixel_id = y*xDim+x;
	    
            if (x+1 < xDim) {
	      
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
              const uint neighbor_id = y*xDim+x+1;
	      
              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
                const uint row = row_base + lx;
		
                const Real cur_ax = ax[row];
	      
                cur_grad[abs_var_base + 2*lx] += cur_ax;
                cur_grad[abs_var_base + 2*lx+1] -= cur_ax;
		
                cur_grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                cur_grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
                const Real cur_ax = ax[row];
		
                cur_grad[vabs_var_base + 2*ly] += cur_ax;
                cur_grad[vabs_var_base + 2*ly+1] -= cur_ax;
		
                cur_grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                cur_grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;
              }
	      
              trans_num++;
            }

            if (y+1 < yDim) {

              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	      
              const uint neighbor_id = (y+1)*xDim+x;
	      
              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
		
                const uint row = row_base + lx;
                const Real cur_ax = ax[row];
	      
                cur_grad[abs_var_base + 2*lx] += cur_ax;
                cur_grad[abs_var_base + 2*lx+1] -= cur_ax;
                cur_grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                cur_grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
                const Real cur_ax = ax[row];
		
                cur_grad[vabs_var_base + 2*ly] += cur_ax;
                cur_grad[vabs_var_base + 2*ly+1] -= cur_ax;
                cur_grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                cur_grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;
              }
	    
              trans_num++;
            }
          }
        } 

        //std::cerr << "B2" << std::endl;

        if (neighborhood >= 8) {

          for (uint y=0; y < yDim; y++) {
            for (uint x=0; x < xDim; x++) {
	      
              const uint pixel_id = y*xDim+x;
	      
              if (x > 0 && y > 0) {
		
                const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
                const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
                const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	  
                const uint neighbor_id = pixel_id-1-xDim;
	  
                //horizontal displacement
                for (uint lx=0; lx < nHorLabels_m1; lx++) {
		  
                  const uint row = row_base + lx;
		  
                  const Real cur_ax = ax[row];
		  
                  cur_grad[abs_var_base + 2*lx] += cur_ax;
                  cur_grad[abs_var_base + 2*lx+1] -= cur_ax;
		
                  cur_grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                  cur_grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;
                }
	      
                //vertical displacement
                for (uint ly=0; ly < nVertLabels_m1; ly++) {
		  
                  const uint row = row_base + nHorLabels_m1 + ly;
                  const Real cur_ax = ax[row];
		  
                  cur_grad[vabs_var_base + 2*ly] += cur_ax;
                  cur_grad[vabs_var_base + 2*ly+1] -= cur_ax;
		  
                  cur_grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                  cur_grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;		  
                }
	      
                trans_num++;
              }

              if (x > 0 && y+1 < yDim) {
		
                const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
                const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
                const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;

                const uint neighbor_id = pixel_id-1+xDim;
		
                //horizontal displacement
                for (uint lx=0; lx < nHorLabels_m1; lx++) {
		  
                  const uint row = row_base + lx;
		  
                  const Real cur_ax = ax[row];
	
                  cur_grad[abs_var_base + 2*lx] += cur_ax;
                  cur_grad[abs_var_base + 2*lx+1] -= cur_ax;
		  
                  cur_grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                  cur_grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;
                }
	      
                //vertical displacement
                for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                  const uint row = row_base + nHorLabels_m1 + ly;
                  const Real cur_ax = ax[row];
	
                  cur_grad[vabs_var_base + 2*ly] += cur_ax;
                  cur_grad[vabs_var_base + 2*ly+1] -= cur_ax;

                  cur_grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                  cur_grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;

                }	      
                trans_num++;
              }		
            }
          }
        }
	
        //std::cerr << "finishing lower bound" << std::endl;

        for (uint v=0; v < nVars; v++)
          lower_bound -= cur_grad[v] * solution[v];
	
        for (uint i=0; i < xDim*yDim; i++)
          lower_bound += *std::min_element(cur_grad.direct_access()+i*nLabels,cur_grad.direct_access()+(i+1)*nLabels);
        for (uint a = xDim*yDim*nLabels; a < nVars; a++) {
          lower_bound += std::min(0.0, cur_grad[a]);
        }

        best_lower_bound = std::max(best_lower_bound,lower_bound);

        std::cerr << "lower bound: " << lower_bound << ", best known: " << best_lower_bound << std::endl;
      }

      gettimeofday(&tEndEnergy,0);
      //std::cerr << "energy calc. took " << diff_seconds(tEndEnergy,tStartEnergy) << " seconds." << std::endl;

      if (((iter-1) % 15) == 0) {
	
        if (fabs(energy_landmark - energy) < cutoff_threshold && iter >= 45) {

          std::cerr << "OSCILLATION or slow convergence detected -> going to next outer iteration" << std::endl;
          break;
        }

        energy_landmark = energy;	
      }
 
      if ((iter_since_restart % 15) == 0) {

        if (iter_since_restart >= 15 && fabs(save_energy - energy) < 1e-6) {

          std::cerr << "iter converged -> CUTOFF" << std::endl;
          break;
        }

        save_energy = energy;
      }

      std::cerr << "iteration " << iter << ", energy: ";
      std::cerr.precision(10);
      std::cerr << energy << std::endl;

      //bool check_stepsize = false;
      //double aux_energy = 1e50;
      
      if (energy > 1.1*last_energy || (energy > last_energy && iter_since_restart >= 5)) {

        // 	std::cerr << "checking step size" << std::endl;

        alpha *= 0.75;
        iter_since_restart = 0;

        prev_t = 1.0;

        std::cerr << "RESTART, alpha = " << alpha << std::endl;
	
        for (uint v=0; v < nVars; v++)
          aux_solution[v] = solution[v];
      }

      iter_since_restart++;

      last_energy = energy;

      /*** 2. calculate gradient ***/

      timeval tStartGrad, tEndGrad;
      gettimeofday(&tStartGrad,0);

#ifdef EXPLICIT_GRADIENT
#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint v=0; v < nLabelVars; v++)
        grad[v] = cost[v];
#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint v=0; v < xDim*yDim*(nHorLabels_m1 + nVertLabels_m1); v++)
        grad[hlevel_var_offs + v] = 0.0;
#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint v=0; v < 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
        grad[abs_var_offs + v] = lambda;
      
      if (neighborhood == 8) {
	
        for (uint v=0; v < 2*nDiagTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
          grad[diag_abs_var_offs+v] = diag_lambda;
      }
#endif

      /**** same for aux_solution now  **/
      ax.set_constant(0.0);

      /*** calculate marginals ****/

#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint i=0; i < xDim*yDim; i++) {

        const uint base = i*nLabels;

        const uint cur_hvar_offs = hlevel_var_offs + i*nHorLabels_m1;
        const uint cur_hcon_offs = hlevel_con_offs + i*nHorLabels_m1;

        //update h-marginals
        for (uint lh = 0; lh < nHorLabels_m1; lh++) {
	  
          Real sum = 0.0;
          for (uint lv = 0; lv < nVertLabels; lv++) {
            sum += aux_solution[base + lv*nHorLabels + lh]; 
          }
	  
          if (lh > 0) {
            sum += aux_solution[cur_hvar_offs + lh-1];
          }
	  
          //if (explicit_level_vars)	    
          ax[cur_hcon_offs + lh] = sum - aux_solution[cur_hvar_offs + lh];
          //else
          //  aux_solution[cur_hvar_offs + lh] = sum;
        }

        const uint cur_vvar_offs = vlevel_var_offs + i*nVertLabels_m1;
        const uint cur_vcon_offs = vlevel_con_offs + i*nVertLabels_m1;
	
        //update v-marginals
        for (uint lv = 0; lv < nVertLabels_m1; lv++) {
	  
          Real sum  = 0.0;
          for (uint lh = 0; lh < nHorLabels; lh++) {
            sum += aux_solution[base + lv*nHorLabels + lh]; 
          }	  
	  
          if (lv > 0) {
            sum += aux_solution[cur_vvar_offs + lv-1];
          }
	  
          //if (explicit_level_vars)	  
          ax[cur_vcon_offs + lv] = sum - aux_solution[cur_vvar_offs + lv];
          //else
          //  aux_solution[cur_vvar_offs + lv] = sum;
        }  
      }

      /** calculate differences **/
      
      trans_num = 0;
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {
	  
          const uint pixel_id = y*xDim+x;
	  
          if (x+1 < xDim) {
	    
            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
            const uint neighbor_id = pixel_id + 1; 

            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
	      
              ax[row] = aux_solution[abs_var_base + 2*lx] - aux_solution[abs_var_base + 2*lx+1]
                + aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                - aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;

              ax[row] = aux_solution[v_abs_var_base + 2*ly] 
                - aux_solution[v_abs_var_base + 2*ly+1]
                + aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                - aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
            }
	    
            trans_num++;
          }

          if (y+1 < yDim) {

            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

            const uint neighbor_id = pixel_id + xDim; 
	    
            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
	      
              ax[row] = aux_solution[abs_var_base + 2*lx] - aux_solution[abs_var_base + 2*lx+1]
                + aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx]
                - aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              //const uint row = row_base + nHorLabels + ly;
              const uint row = row_base + nHorLabels_m1 + ly;
	
              ax[row] = aux_solution[v_abs_var_base + 2*ly]
                - aux_solution[v_abs_var_base + 2*ly+1]
                + aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly]
                - aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
      	    }
	    
            trans_num++;
          }
        }
      }

      if (neighborhood >= 8) {

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const uint pixel_id = y*xDim+x;
	    
            if (x>0 && y >0) {
	    
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

              const uint neighbor_id = pixel_id-1-xDim;

              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
                const uint row = row_base + lx;
		
                ax[row] = aux_solution[abs_var_base + 2*lx] - aux_solution[abs_var_base + 2*lx+1]
                  + aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                  - aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                //const uint row = row_base + nHorLabels + ly;
                const uint row = row_base + nHorLabels_m1 + ly;
		
                ax[row] = aux_solution[v_abs_var_base + 2*ly] 
                  - aux_solution[v_abs_var_base + 2*ly+1]
                  + aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                  - aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
              }
	      
              trans_num++;
            }

            if (x>0 && y+1 < yDim) {
	    
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint v_abs_var_base = abs_var_base + 2*nHorLabels_m1;

              const uint neighbor_id = pixel_id-1+xDim;

              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
                const uint row = row_base + lx;
		
                ax[row] = aux_solution[abs_var_base + 2*lx] - aux_solution[abs_var_base + 2*lx+1]
                  + aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                  - aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];
              }

              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
		
                ax[row] = aux_solution[v_abs_var_base + 2*ly] 
                  - aux_solution[v_abs_var_base + 2*ly+1]
                  + aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                  - aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];
              }
	      
              trans_num++;
            }
          }
        }
      }

      /**** now the transpose multiply ****/

      ax *= penalty;
      ax += lagrange_multiplier;


#ifndef EXPLICIT_GRADIENT
      ax *= alpha;

      const double alph_lam = alpha*lambda;
      const double diag_alph_lam = alpha*diag_lambda;

      for (uint v=0; v < nLabelVars; v++)
        aux_solution[v] -= alpha*cost[v];
      for (uint v=0; v < 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
        aux_solution[abs_var_offs + v] -= alph_lam;
      
      if (neighborhood == 8) {
	
        for (uint v=0; v < 2*nDiagTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
          aux_solution[diag_abs_var_offs+v] -= diag_alph_lam;
      }
#endif
      
      /*** calculate marginals ****/

#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {

          const uint base = (y*xDim+x)*nLabels;
          const uint i=y*xDim+x;

          const uint hlevel_base = hlevel_var_offs + i*nHorLabels_m1;
          const uint vlevel_base = vlevel_var_offs + i*nVertLabels_m1;

          const uint cur_hcon_offs = hlevel_con_offs + i*nHorLabels_m1;

          //update h-marginals
          Real last_ax = 0.0;

          for (int lh = nHorLabels_m1-1; lh >=0; lh--) {
	    
            const Real cur_ax = ax[cur_hcon_offs + lh];

#ifdef EXPLICIT_GRADIENT
            for (uint lv = 0; lv < nVertLabels; lv++) {
              grad[base + lv*nHorLabels + lh] += cur_ax;
            }

            grad[hlevel_base + lh] += last_ax - cur_ax;
#else
            //NOTE: multiplication with alpha is already included in ax
            for (uint lv = 0; lv < nVertLabels; lv++) {
              aux_solution[base + lv*nHorLabels + lh] -= cur_ax;
            }
            aux_solution[hlevel_base + lh] -= (last_ax - cur_ax);
#endif

            last_ax = cur_ax;
          }
	
          //update v-marginals
          last_ax = 0.0;

          const uint cur_vcon_offs = vlevel_con_offs + i*nVertLabels_m1;

          for (int lv = nVertLabels_m1-1; lv >= 0; lv--) {
	      
            const Real cur_ax = ax[cur_vcon_offs + lv];

#ifdef EXPLICIT_GRADIENT    
            for (uint lh = 0; lh < nHorLabels; lh++) {
              grad[base + lv*nHorLabels + lh] += cur_ax;
            }
	    
            grad[vlevel_base + lv] += last_ax - cur_ax;
#else
            //NOTE: multiplication with alpha is already included in ax
            for (uint lh = 0; lh < nHorLabels; lh++) {
              aux_solution[base + lv*nHorLabels + lh] -= cur_ax;
            }
            aux_solution[vlevel_base + lv] -= (last_ax - cur_ax);
#endif
            last_ax = cur_ax;
          }
        }  
      }

      /** calculate differences **/
      
      trans_num = 0;
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {
	  
          const uint pixel_id = y*xDim+x;
	  
          if (x+1 < xDim) {
	    
            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
            const uint neighbor_id = y*xDim+x+1;

            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;

              const Real cur_ax = ax[row];

#ifdef EXPLICIT_GRADIENT
              grad[abs_var_base + 2*lx] += cur_ax;
              grad[abs_var_base + 2*lx+1] -= cur_ax;

              //if (explicit_level_vars) {

              grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
              grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;

              // 	      }
              // 	      else {
		
              // 		//TODO: make this more efficient
              // 		for (uint lxx = 0; lxx <= lx; lxx++) {
              // 		  for (uint ly = 0; ly < nHorLabels; ly++) {

              // 		    grad[pixel_id*nLabels + ly*nHorLabels+lxx] += cur_ax;
              // 		    grad[neighbor_id*nLabels +ly*nHorLabels+lxx] -= cur_ax;
              // 		  }
              // 		}   
              // 	      }
#else
              //NOTE: multilpication with alpha is already included in ax
              aux_solution[abs_var_base + 2*lx] -= cur_ax;
              aux_solution[abs_var_base + 2*lx+1] += cur_ax;

              aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] -= cur_ax;
              aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] += cur_ax;
#endif
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;
              const Real cur_ax = ax[row];

#ifdef EXPLICIT_GRADIENT
              grad[vabs_var_base + 2*ly] += cur_ax;
              grad[vabs_var_base + 2*ly+1] -= cur_ax;


              //if (explicit_level_vars) {

              grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
              grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;

              // 	      }
              // 	      else {

              // 		//TODO: make this more efficient
              // 		for (uint lyy = 0; lyy <= ly; lyy++) {
              // 		  for (uint lx=0; lx < nHorLabels; lx++) {

              // 		    grad[pixel_id+nLabels + lyy*nHorLabels+lx] += cur_ax;
              // 		    grad[neighbor_id*nLabels + lyy*nHorLabels+lx] -= cur_ax;
              // 		  }
              // 		}

              // 	      }
#else
              //NOTE: multiplication with alpha is already included in ax
              aux_solution[vabs_var_base + 2*ly] -= cur_ax;
              aux_solution[vabs_var_base + 2*ly+1] += cur_ax;
              aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] -= cur_ax;
              aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] += cur_ax;
#endif
            }
	    
            trans_num++;
          }

          if (y+1 < yDim) {

            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	    
            const uint neighbor_id = (y+1)*xDim+x;
	    
            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
              const Real cur_ax = ax[row];

#ifdef EXPLICIT_GRADIENT
              grad[abs_var_base + 2*lx] += cur_ax;
              grad[abs_var_base + 2*lx+1] -= cur_ax;
              grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
              grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;

#else
              //NOTE: multilpication with alpha is already included in ax
              aux_solution[abs_var_base + 2*lx] -= cur_ax;
              aux_solution[abs_var_base + 2*lx+1] += cur_ax;
              aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] -= cur_ax;
              aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] += cur_ax;
#endif
            }

            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;
              const Real cur_ax = ax[row];

#ifdef EXPLICIT_GRADIENT
              grad[vabs_var_base + 2*ly] += cur_ax;
              grad[vabs_var_base + 2*ly+1] -= cur_ax;
              grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
              grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;
#else
              //NOTE: multilpication with alpha is already included in ax
              aux_solution[vabs_var_base + 2*ly] -= cur_ax;
              aux_solution[vabs_var_base + 2*ly+1] += cur_ax;
              aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] -= cur_ax;
              aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] += cur_ax;
#endif
      	    }
	    
            trans_num++;
          }
        }
      } 

      if (neighborhood >= 8) {

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {

            const uint pixel_id = y*xDim+x;
	    
            if (x > 0 && y > 0) {
	      
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;
	  
              const uint neighbor_id = pixel_id-1-xDim;
	  
              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
		
                const uint row = row_base + lx;
		
                const Real cur_ax = ax[row];
	
#ifdef EXPLICIT_GRADIENT	
                grad[abs_var_base + 2*lx] += cur_ax;
                grad[abs_var_base + 2*lx+1] -= cur_ax;
		
                //if (explicit_level_vars) {

                grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;

                // 		}
                // 		else {
		  
                // 		  //TODO: make this more efficient
                // 		  for (uint lxx = 0; lxx <= lx; lxx++) {
                // 		    for (uint ly = 0; ly < nHorLabels; ly++) {
		      
                // 		      grad[pixel_id*nLabels + ly*nHorLabels+lxx] += cur_ax;
                // 		      grad[neighbor_id*nLabels +ly*nHorLabels+lxx] -= cur_ax;
                // 		    }
                // 		  }   
                // 		}
#else
                //NOTE: multilpication with alpha is already included in ax
                aux_solution[abs_var_base + 2*lx] -= cur_ax;
                aux_solution[abs_var_base + 2*lx+1] += cur_ax;
		
                aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] -= cur_ax;
                aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] += cur_ax;

#endif
              }
	      
              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
                const Real cur_ax = ax[row];
	
#ifdef EXPLICIT_GRADIENT
                grad[vabs_var_base + 2*ly] += cur_ax;
                grad[vabs_var_base + 2*ly+1] -= cur_ax;
		
                //if (explicit_level_vars) {

                grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;

                // 		}
                // 		else {

                // 		  //TODO: make this more efficient
                // 		  for (uint lyy = 0; lyy <= ly; lyy++) {
                // 		    for (uint lx=0; lx < nHorLabels; lx++) {
		      
                // 		      grad[pixel_id+nLabels + lyy*nHorLabels+lx] += cur_ax;
                // 		      grad[neighbor_id*nLabels + lyy*nHorLabels+lx] -= cur_ax;
                // 		    }
                // 		  }

                // 		}
#else
                //NOTE: multiplication with alpha is already included in ax
                aux_solution[vabs_var_base + 2*ly] -= cur_ax;
                aux_solution[vabs_var_base + 2*ly+1] += cur_ax;
                aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] -= cur_ax;
                aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] += cur_ax;
#endif
              }
	      
              trans_num++;
            }

            if (x > 0 && y+1 < yDim) {
	      
              const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
              const uint vabs_var_base = abs_var_base + 2*nHorLabels_m1;

              const uint neighbor_id = pixel_id-1+xDim;
	  
              //horizontal displacement
              for (uint lx=0; lx < nHorLabels_m1; lx++) {
		
                const uint row = row_base + lx;
		
                const Real cur_ax = ax[row];
	
#ifdef EXPLICIT_GRADIENT	
                grad[abs_var_base + 2*lx] += cur_ax;
                grad[abs_var_base + 2*lx+1] -= cur_ax;
		
                //if (explicit_level_vars) {

                grad[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] += cur_ax;
                grad[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] -= cur_ax;

                // 		}
                // 		else {
		  
                // 		  //TODO: make this more efficient
                // 		  for (uint lxx = 0; lxx <= lx; lxx++) {
                // 		    for (uint ly = 0; ly < nHorLabels; ly++) {
		      
                // 		      grad[pixel_id*nLabels + ly*nHorLabels+lxx] += cur_ax;
                // 		      grad[neighbor_id*nLabels +ly*nHorLabels+lxx] -= cur_ax;
                // 		    }
                // 		  }   
                // 		}
#else
                //NOTE: multilpication with alpha is already included in ax
                aux_solution[abs_var_base + 2*lx] -= cur_ax;
                aux_solution[abs_var_base + 2*lx+1] += cur_ax;
                aux_solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] -= cur_ax;
                aux_solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx] += cur_ax;
#endif
              }
	      
              //vertical displacement
              for (uint ly=0; ly < nVertLabels_m1; ly++) {
		
                const uint row = row_base + nHorLabels_m1 + ly;
                const Real cur_ax = ax[row];
	
#ifdef EXPLICIT_GRADIENT	
                grad[vabs_var_base + 2*ly] += cur_ax;
                grad[vabs_var_base + 2*ly+1] -= cur_ax;

                //if (explicit_level_vars) {

                grad[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] += cur_ax;
                grad[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] -= cur_ax;

                // 		}
                // 		else {

                // 		  //TODO: make this more efficient
                // 		  for (uint lyy = 0; lyy <= ly; lyy++) {
                // 		    for (uint lx=0; lx < nHorLabels; lx++) {
		      
                // 		      grad[pixel_id+nLabels + lyy*nHorLabels+lx] += cur_ax;
                // 		      grad[neighbor_id*nLabels + lyy*nHorLabels+lx] -= cur_ax;
                // 		    }
                // 		  }

                // 		}
#else
                //NOTE: multiplication with alpha is already included in ax
                aux_solution[vabs_var_base + 2*ly] -= cur_ax;
                aux_solution[vabs_var_base + 2*ly+1] += cur_ax;
                aux_solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] -= cur_ax;
                aux_solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly] += cur_ax;
#endif
              }
	      
              trans_num++;
            }

          }
        }
      }

      gettimeofday(&tEndGrad,0);
      //std::cerr << "gradient calculation took " << diff_seconds(tEndGrad,tStartGrad) << " seconds." << std::endl;

#ifdef EXPLICIT_GRADIENT     
      /*** 3. go in the direction of the negative gradient  ***/
      for (uint v=0; v < nVars; v++) {	
        aux_solution[v] -= alpha*grad[v];
      }
#endif
      
      timeval tStartProj, tEndProj;
      gettimeofday(&tStartProj,0);

      /*** 4. reproject to the convex set defined by the simplices and the variable bounds ***/      
#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint s=0; s < xDim*yDim; s++) {
        const uint start = s*nLabels;
        projection_on_simplex(aux_solution.direct_access()+start, nLabels);
      } 
      
#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint v=xDim*yDim*nLabels; v < nVars; v++) {

        if (aux_solution[v] < 0.0)
          aux_solution[v] = 0.0;
        else if (aux_solution[v] > 1.0)
          aux_solution[v] = 1.0;
      }

      gettimeofday(&tEndProj,0);
      //std::cerr << "projection took " << diff_seconds(tEndProj,tStartProj) << " seconds" << std::endl;

      /*** 5. update variables according to Nesterov scheme ***/
      const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
      const Real nesterov_fac = (prev_t - 1.0) / new_t;
      //const Real nesterov_fac = ((double) (iter_since_restart-1)) / ((double) (iter_since_restart+1));	  
	  

#ifdef USE_OMP
      //#pragma omp parallel for
#endif
      for (uint i=0; i < nVars; i++) {
	
        const Real old_aux = aux_solution.direct_access(i);
        aux_solution.direct_access(i) = old_aux + nesterov_fac*(old_aux - solution[i]);
        solution[i] = old_aux;
      }
      
      prev_t = new_t;
      
    } //end of inner iterations
  
    Real energy = 0.0;
    for (uint v=0; v < xDim*yDim*nLabels; v++)
      energy += solution[v] * cost[v];
    for (uint v=0; v < 2*nStraightTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
      energy += solution[abs_var_offs + v] * lambda;
      
    if (neighborhood >= 8) {
      
      for (uint v=0; v < 2*nDiagTransitions*(nHorLabels_m1+nVertLabels_m1); v++)
        energy += solution[diag_abs_var_offs+v] * diag_lambda;
    }

    std::cerr << "lp-energy: " << energy << std::endl;

    Real infeas_penalty = 0.0;

    /*** update lagrange multipliers ****/
    ax.set_constant(0.0);

    //assert(!explicit_level_vars);

    /*** calculate marginals ****/
    
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint base = (y*xDim+x)*nLabels;
        const uint i = y*xDim+x;
	
        //update h-marginals
        for (uint lh = 0; lh < nHorLabels_m1; lh++) {
	  
          Real sum = 0.0;
          for (uint lv = 0; lv < nVertLabels; lv++)
            sum += solution[base + lv*nHorLabels + lh]; 
	  
          if (lh > 0)
            sum += solution[hlevel_var_offs + i*nHorLabels_m1 + lh-1];
	  
          assert(sum >= 0.0);
          assert(sum <= 1.001);
	  
          const Real addon = sum - solution[hlevel_var_offs + i*nHorLabels_m1 + lh];
          infeas_penalty += addon*addon;

          ax[hlevel_con_offs + i*nHorLabels_m1 + lh] += addon;
        }
	
        //update v-marginals
        for (uint lv = 0; lv < nVertLabels_m1; lv++) {
	  
          Real sum  = 0.0;
          for (uint lh = 0; lh < nHorLabels; lh++) 
            sum += solution[base + lv*nHorLabels + lh]; 
	  
          if (lv > 0)
            sum += solution[vlevel_var_offs + i*nVertLabels_m1 + lv-1];
	  
          assert(sum >= 0.0);
          assert(sum <= 1.001);
	  
          const Real addon = sum - solution[vlevel_var_offs + i*nVertLabels_m1 + lv];
          infeas_penalty += addon*addon;

          ax[vlevel_con_offs + i*nVertLabels_m1 + lv] += addon;
        }
      }
    }
    
    /** calculate differences **/
    
    uint trans_num = 0;
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint pixel_id = y*xDim+x;
	
        if (x+1 < xDim) {
	    
          const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
	  
          const uint neighbor_id = y*xDim+x+1;
	  
          //horizontal displacement
          for (uint lx=0; lx < nHorLabels_m1; lx++) {
	    
            const uint row = row_base + lx;

            const Real addon = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
              + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
              - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

            infeas_penalty += addon*addon;
	    
            ax[row] = addon;
          }
	  
          //vertical displacement
          for (uint ly=0; ly < nVertLabels_m1; ly++) {
	    
            const uint row = row_base + nHorLabels_m1 + ly;

            const Real addon = solution[abs_var_base + 2*nHorLabels_m1 + 2*ly] 
              - solution[abs_var_base + 2*nHorLabels_m1 + 2*ly+1]
              + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
              - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

            infeas_penalty += addon*addon;
	    
            ax[row] = addon;
          }
	  
          trans_num++;
        }
	
        if (y+1 < yDim) {
	  
          const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
          const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
	  
          const uint neighbor_id = (y+1)*xDim+x;
	  
          //horizontal displacement
          for (uint lx=0; lx < nHorLabels_m1; lx++) {
	    
            const uint row = row_base + lx;
	    
            const Real addon = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
              + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx]
              - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

            infeas_penalty += addon*addon;

            ax[row] = addon; 
          }

          //vertical displacement
          for (uint ly=0; ly < nVertLabels_m1; ly++) {
	    
            const uint row = row_base + nHorLabels_m1 + ly;

            const Real addon = solution[abs_var_base + 2*nHorLabels_m1 + 2*ly]
              - solution[abs_var_base + 2*nHorLabels_m1 + 2*ly+1]
              + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly]
              - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

            infeas_penalty += addon*addon;
	    
            ax[row] = addon;
          }
	  
          trans_num++;
        }
      }
    }


    if (neighborhood >= 8) {
      for (uint y=0; y < yDim; y++) {
        for (uint x=0; x < xDim; x++) {
	
          const uint pixel_id = y*xDim+x;

          if (x > 0 && y > 0) {
	    
            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);
	    
            const uint neighbor_id = pixel_id-1-xDim;
	    
            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;

              const Real addon = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

              infeas_penalty += addon*addon;
	      
              ax[row] = addon;
            }
	    
            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;

              const Real addon = solution[abs_var_base + 2*nHorLabels_m1 + 2*ly] 
                - solution[abs_var_base + 2*nHorLabels_m1 + 2*ly+1]
                + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

              infeas_penalty += addon*addon;
	      
              ax[row] = addon;
            }
	  
            trans_num++;
          }	

          if (x > 0 && y+1 < yDim) {
	    
            const uint row_base = abs_con_offs + trans_num*(nHorLabels_m1+nVertLabels_m1);
            const uint abs_var_base = abs_var_offs + 2*trans_num*(nHorLabels_m1+nVertLabels_m1);

            const uint neighbor_id = pixel_id-1+xDim;
	    
            //horizontal displacement
            for (uint lx=0; lx < nHorLabels_m1; lx++) {
	      
              const uint row = row_base + lx;
	      
              const Real addon = solution[abs_var_base + 2*lx] - solution[abs_var_base + 2*lx+1]
                + solution[hlevel_var_offs + pixel_id*nHorLabels_m1+lx] 
                - solution[hlevel_var_offs + neighbor_id*nHorLabels_m1+lx];

              infeas_penalty += addon*addon;

              ax[row] = addon;
            }
	    
            //vertical displacement
            for (uint ly=0; ly < nVertLabels_m1; ly++) {
	      
              const uint row = row_base + nHorLabels_m1 + ly;

              const Real addon = solution[abs_var_base + 2*nHorLabels_m1 + 2*ly] 
                - solution[abs_var_base + 2*nHorLabels_m1 + 2*ly+1]
                + solution[vlevel_var_offs + pixel_id*nVertLabels_m1+ly] 
                - solution[vlevel_var_offs + neighbor_id*nVertLabels_m1+ly];

              infeas_penalty += addon*addon;

              ax[row] = addon;
            }
	  
            trans_num++;
          }	
        }
      }
    }

    infeas_penalty *= penalty;

    std::cerr << "augmented lagragian penalty: " << infeas_penalty << std::endl; 

    for (uint c=0; c < nConstraints; c++) {
      lagrange_multiplier[c] += penalty * ax[c];
    }
  }

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      int min_l = MAX_UINT;
      Real max_val = 0.0;

      for (uint l=0; l < nLabels; l++) {

        if (solution[(y*xDim+x)*nLabels + l] > max_val) {
          max_val = solution[(y*xDim+x)*nLabels + l];
          min_l = l;
        }
      }

      labeling(x,y) = min_l;

      velocity(x,y,0) = (min_l % nHorLabels)*inv_spacing;
      velocity(x,y,1) = (min_l / nHorLabels)*inv_spacing;

      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 

    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  if (nOuterIter > 1)
    discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);


  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      int min_l = labeling(x,y);

      velocity(x,y,0) = (min_l % nHorLabels)*inv_spacing;
      velocity(x,y,1) = (min_l / nHorLabels)*inv_spacing;

      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 
    }
  }

  return last_energy;
}


/************************************************************************************************************/

//block coordinate descent version
double lp_motion_estimation_bcd(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp,
                                uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

#ifdef HAS_CLP
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

  assert(neighborhood == 4);

  Math3D::NamedTensor<double> label_cost(xDim,yDim,nLabels, 0.0, MAKENAME(label_cost));

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


  Math3D::NamedTensor<double> label_val(xDim,yDim,nLabels, 0.0, MAKENAME(label_val));
  Math3D::NamedTensor<double> hmarginal(xDim,yDim,nHorLabels, 0.0, MAKENAME(hmarginal));
  Math3D::NamedTensor<double> vmarginal(xDim,yDim,nVertLabels, 0.0, MAKENAME(vmarginal));  

  /*** initialization ***/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

#if 1
      uint min_l = MAX_UINT;
      double min_cost = 1e300;

      for (uint l=0; l < nLabels; l++) {

        if (label_cost(x,y,l) < min_cost) {
          min_cost = label_cost(x,y,l);
          min_l = l;
        }
      }

      label_val(x,y,min_l) = 1.0;
#else
      for (uint l=0; l < nLabels; l++)
        label_val(x,y,l) = 1.0 / nLabels;
#endif

      //update h-marginals
      for (uint lh = 0; lh < nHorLabels; lh++) {
    
        double sum = 0.0;
        for (uint lv = 0; lv < nVertLabels; lv++)
          sum += label_val(x,y,lv*nHorLabels + lh);
	
        if (lh > 0)
          sum += hmarginal(x,y,lh-1);

        assert(sum >= 0.0);
        assert(sum <= 1.001);
	
        hmarginal(x,y,lh) = sum;
      }
      
      //update v-marginals
      for (uint lv = 0; lv < nVertLabels; lv++) {
    
        double sum  = 0.0;
        for (uint lh = 0; lh < nHorLabels; lh++) 
          sum += label_val(x,y,lv*nHorLabels + lh);
	
        if (lv > 0)
          sum += vmarginal(x,y,lv-1);

        assert(sum >= 0.0);
        assert(sum <= 1.001);
	
        vmarginal(x,y,lv) = sum;
      }
    }
  }

  double last_energy = 1e300;
  double energy = 1e300;

  for (uint iter=1; iter <= 75; iter++) {

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        //solve mini-lp
	
        const uint nVars = nLabels + nHorLabels + nVertLabels + 2*neighborhood * (nHorLabels + nVertLabels);
	
        const uint nConstraints = 1 + (1 + 2*neighborhood)*(nHorLabels + nVertLabels);
	
        Math1D::Vector<double> lp_cost(nVars,0.0);
        for (uint l=0; l < nLabels; l++)
          lp_cost[l] = label_cost(x,y,l);

        Math1D::Vector<double> var_lb(nVars,0.0);
        Math1D::Vector<double> var_ub(nVars,1.0);

        Math1D::Vector<double> rhs(nConstraints,0.0);
	
        rhs[0] = 1.0;

        const uint nMatrixEntries = 3*nLabels + 2*(nHorLabels + nVertLabels)
          + 3*neighborhood*(nHorLabels + nVertLabels);

        SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);
	
        for (uint l=0; l < nLabels; l++)
          lp_descr.add_entry(0,l,1);

        for (uint lh = 0; lh < nHorLabels; lh++) {

          lp_descr.add_entry(lh+1, nLabels + lh, -1);

          if (lh > 0)
            lp_descr.add_entry(lh+1, nLabels + lh - 1, 1);
	  
          for (uint lv = 0; lv < nVertLabels; lv++)
            lp_descr.add_entry(lh+1, lv*nHorLabels+lh , 1);
        }

        for (uint lv = 0; lv < nVertLabels; lv++) {

          lp_descr.add_entry(nHorLabels + 1 + lv, nLabels + nHorLabels + lv, -1);

          if (lv > 0)
            lp_descr.add_entry(nHorLabels + 1 + lv, nLabels + nHorLabels + lv-1, 1);

          for (uint lh = 0; lh < nHorLabels; lh++) {

            lp_descr.add_entry(nHorLabels + 1 + lv, lv*nHorLabels+lh , 1);
          }
        }

        const uint abs_var_offs = nLabels + nHorLabels + nVertLabels;
        const uint vabs_var_offs = abs_var_offs + 2*neighborhood*(nHorLabels);
        const uint abs_con_offs = nHorLabels+nVertLabels+1;
	
#if 1
        for (uint i=0; i < neighborhood; i++) {

#if 1
          //smoothness of the horizontal displacements
          for (uint lh = 0; lh < nHorLabels; lh++) {
	  
            uint row = abs_con_offs + i*(nHorLabels+nVertLabels) + lh;
            lp_descr.add_entry(row, nLabels + lh, 1);
            lp_descr.add_entry(row, abs_var_offs + 2*i*(nHorLabels) + 2*lh, -1);
            lp_descr.add_entry(row, abs_var_offs + 2*i*(nHorLabels) + 2*lh + 1, 1);
	    
            //upwards transition
            if (i == 0) {

              if (y > 0) {
                rhs[row] = hmarginal(x,y-1,lh);
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh] = lambda;
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh + 1] = lambda;
              }
            }
	    
            //downwards transition
            if (i == 1) {

              if (y+1 < yDim) {
                rhs[row] = hmarginal(x,y+1,lh);
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh] = lambda;
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh + 1] = lambda;
              }
            }
	    
            //left transition
            if (i == 2) {

              if (x+1 < xDim) {
                rhs[row] = hmarginal(x+1,y,lh);
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh] = lambda;
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh + 1] = lambda;
              }
            }
	    
            //right transition
            if (i == 3) {

              if (x > 0) {
                rhs[row] = hmarginal(x-1,y,lh);
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh] = lambda;
                lp_cost[abs_var_offs + 2*i*(nHorLabels) + 2*lh + 1] = lambda;
              }	      
            }
          }
#endif

#if 1
          //smoothness of the vertical displacements
          for (uint lv = 0; lv < nVertLabels; lv++) {
	  
            uint row = abs_con_offs + i*(nHorLabels+nVertLabels) + nHorLabels + lv;
            lp_descr.add_entry(row, nLabels + nHorLabels + lv, 1);
            lp_descr.add_entry(row, vabs_var_offs + 2*i*(nVertLabels) + 2*lv, -1);
            lp_descr.add_entry(row, vabs_var_offs + 2*i*(nVertLabels) + 2*lv + 1, 1);

            //upwards transition
            if (i == 0) {

              if (y > 0) {
                rhs[row] = vmarginal(x,y-1,lv);
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv] = lambda;
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv + 1] = lambda;
              }
            }
	    
            //downwards transition
            if (i == 1) {

              if (y+1 < yDim) {
                rhs[row] = vmarginal(x,y+1,lv);
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv] = lambda;
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv + 1] = lambda;
              }
            }

            //left transition
            if (i == 2) {

              if (x+1 < xDim) {
                rhs[row] = vmarginal(x+1,y,lv);
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv] = lambda;
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv + 1] = lambda;
              }
            }

            //right transition
            if (i == 3) {

              if (x > 0) {
                rhs[row] = vmarginal(x-1,y,lv);
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv] = lambda;
                lp_cost[vabs_var_offs + 2*i*(nVertLabels) + 2*lv + 1] = lambda;
              }
            }
          }
#endif
        }
#endif

        CoinPackedMatrix coinMatrix(false,(int*) lp_descr.row_indices(),(int*) lp_descr.col_indices(),
                                    lp_descr.value(),lp_descr.nEntries());
	
        ClpSimplex lpSolver;
        lpSolver.loadProblem (coinMatrix, var_lb.direct_access(), var_ub.direct_access(),   
                              lp_cost.direct_access(), rhs.direct_access(), rhs.direct_access());

        coinMatrix.cleanMatrix();

        lpSolver.setLogLevel(0);
	
        int error = lpSolver.dual();
        assert(!error);
	
        //extract labels
        for (uint l=0; l < nLabels; l++)
          label_val(x,y,l) = lpSolver.primalColumnSolution()[l];

        //update h-marginals
        for (uint lh = 0; lh < nHorLabels; lh++) {

          double sum = 0.0;
          for (uint lv = 0; lv < nVertLabels; lv++)
            sum += label_val(x,y,lv*nHorLabels + lh);

          if (lh > 0)
            sum += hmarginal(x,y,lh-1);

          hmarginal(x,y,lh) = sum;
        }

        //update v-marginals
        for (uint lv = 0; lv < nVertLabels; lv++) {

          double sum  = 0.0;
          for (uint lh = 0; lh < nHorLabels; lh++) 
            sum += label_val(x,y,lv*nHorLabels + lh);

          if (lv > 0)
            sum += vmarginal(x,y,lv-1);

          vmarginal(x,y,lv) = sum;
        }

      }
    }

    energy = 0.0;
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
    
        for (uint l=0; l < nLabels; l++)
          energy += label_cost(x,y,l) * label_val(x,y,l);

        if (x+1 < xDim) {

          for (uint lh = 0; lh < nHorLabels; lh++)
            energy += lambda * fabs(hmarginal(x,y,lh) - hmarginal(x+1,y,lh));

          for (uint lv = 0; lv < nVertLabels; lv++)
            energy += lambda * fabs(vmarginal(x,y,lv) - vmarginal(x+1,y,lv));
        }
        if (y+1 < yDim) {

          for (uint lh = 0; lh < nHorLabels; lh++)
            energy += lambda * fabs(hmarginal(x,y,lh) - hmarginal(x,y+1,lh));

          for (uint lv = 0; lv < nVertLabels; lv++)
            energy += lambda * fabs(vmarginal(x,y,lv) - vmarginal(x,y+1,lv));	  
        }
      }
    }

    std::cerr << "energy: " << energy << std::endl;

    if (fabs(energy-last_energy) < 0.001)
      break;

    last_energy = energy;
  }

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      int min_l = MAX_UINT;
      double max_val = 0.0;

      for (uint l=0; l < nLabels; l++) {

        if (label_val(x,y,l) > max_val) {
          max_val = label_val(x,y,l);
          min_l = l;
        }
      }

      velocity(x,y,0) = (min_l % nHorLabels);// + min_x_disp;
      velocity(x,y,1) = (min_l / nHorLabels);// + min_y_disp;

      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 

    }
  }

  return energy;
#else
  return 0.0;
#endif
}

