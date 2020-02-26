/**** written by Thomas Schoenemann as an employee of Lund University, September 2010 ****/


#include "motion_trws.hh"

//inclusion of TRW-S:
#include "MRFEnergy.h"
#include "typeGeneral.h"

#include "tensor_interpolation.hh"
#include "matrix.hh"
#include "vector.hh"
#include "motion_discrete.hh"
#include "motion_moves.hh"
#include "timing.hh"
#include "factorDualOpt.hh"
#include "factorChainDualDecomp.hh"

#include "trws.hh"
#include "factorTRWS.hh"
#include "separatorDualOpt.hh" //only for testing
#include "separatorTRWS.hh" // only for testing

double trws_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                              int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                              uint neighborhood, double lambda, Math3D::Tensor<double>& velocity) {

  double inv_spacing = 1.0 / ((double) spacing);

  double org_lambda = lambda;
  lambda *= inv_spacing;

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp + 1) * spacing - (spacing - 1);

  //NOTE: we presently need typeGeneral as the data edges are general and mixing seems to be impossible

  MRFEnergy<TypeGeneral>* mrf;
  MRFEnergy<TypeGeneral>::Options options;
  TypeGeneral::REAL energy, lowerBound;
  
  mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());

  const uint K = std::max(nHorLabels,nVertLabels); // number of labels
  Math1D::Vector<TypeGeneral::REAL> D(K,0.0);

  Math3D::Tensor<MRFEnergy<TypeGeneral>::NodeId> node_adr(xDim,yDim,2);

  /*** construct nodes (all cost are zero) ****/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      node_adr(x,y,0) = mrf->AddNode(TypeGeneral::LocalSize(nHorLabels), TypeGeneral::NodeData(D.direct_access()));
      node_adr(x,y,1) = mrf->AddNode(TypeGeneral::LocalSize(nVertLabels), TypeGeneral::NodeData(D.direct_access()));
    }
  }

  Math1D::Vector<TypeGeneral::REAL> data_edge(nHorLabels*nVertLabels,0.0);
  Math1D::Vector<TypeGeneral::REAL> hor_smooth_edge(nHorLabels*nHorLabels);
  for (int l1=0; l1 < (int) nHorLabels; l1++) {
    for (int l2=0; l2 < (int) nHorLabels; l2++) {
      hor_smooth_edge[l1*nHorLabels+l2] = lambda * fabs(l1-l2);
    }
  }

  Math1D::Vector<TypeGeneral::REAL> vert_smooth_edge(nVertLabels*nVertLabels);
  for (int l1=0; l1 < (int) nVertLabels; l1++) {
    for (int l2=0; l2 < (int) nVertLabels; l2++) {
      vert_smooth_edge[l1*nVertLabels+l2] = lambda * fabs(l1-l2);
    }
  }

  TypeGeneral::EdgeData data_edge_d(TypeGeneral::GENERAL, data_edge.direct_access());
  TypeGeneral::EdgeData hor_smooth_edge_d(TypeGeneral::GENERAL, hor_smooth_edge.direct_access());
  TypeGeneral::EdgeData vert_smooth_edge_d(TypeGeneral::GENERAL, vert_smooth_edge.direct_access());

  /*** construct edges ****/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      // 1) data edge
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
	  
          double disp_cost = 0.0;
	  
          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }
	  
          data_edge[ly*nHorLabels+lx] = disp_cost;
        }
      }
      
      mrf->AddEdge(node_adr(x,y,0), node_adr(x,y,1), data_edge_d);    
      
      /*** add smoothness edges ***/
      if (x > 0) {
        mrf->AddEdge(node_adr(x,y,0), node_adr(x-1,y,0), hor_smooth_edge_d);    
        mrf->AddEdge(node_adr(x,y,1), node_adr(x-1,y,1), vert_smooth_edge_d);    
      }
      if (y > 0) {
        mrf->AddEdge(node_adr(x,y,0), node_adr(x,y-1,0), hor_smooth_edge_d);    
        mrf->AddEdge(node_adr(x,y,1), node_adr(x,y-1,1), vert_smooth_edge_d);    
      }
    }
  }

  if (neighborhood == 8) {

    hor_smooth_edge *= 1.0 / sqrt(2.0);
    vert_smooth_edge *= 1.0 / sqrt(2.0);

    TypeGeneral::EdgeData diag_hor_smooth_edge_d(TypeGeneral::GENERAL, hor_smooth_edge.direct_access());
    TypeGeneral::EdgeData diag_vert_smooth_edge_d(TypeGeneral::GENERAL, vert_smooth_edge.direct_access());

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        if (x > 0 && y > 0) {
          mrf->AddEdge(node_adr(x,y,0), node_adr(x-1,y-1,0), diag_hor_smooth_edge_d);    
          mrf->AddEdge(node_adr(x,y,1), node_adr(x-1,y-1,1), diag_vert_smooth_edge_d);    
        }
        if (x > 0 && y+1 < yDim) {
          mrf->AddEdge(node_adr(x,y,0), node_adr(x-1,y+1,0), diag_hor_smooth_edge_d);    
          mrf->AddEdge(node_adr(x,y,1), node_adr(x-1,y+1,1), diag_vert_smooth_edge_d);    
        }
      }
    }
  }

  options.m_iterMax = 250; //2500; // maximum number of iterations
  options.m_printIter = 3;     
  options.m_printMinIter = 1;

  std::clock_t tStart = std::clock();

  mrf->Minimize_TRW_S(options, lowerBound, energy);

  std::cerr << "TRW-S needed " << diff_seconds(std::clock(),tStart) << " seconds." << std::endl;

  Math2D::Matrix<uint> labeling(xDim,yDim,0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {
      int u = mrf->GetSolution(node_adr(x,y,0));
      int v = mrf->GetSolution(node_adr(x,y,1));

      labeling(x,y) = v*nHorLabels+u;

      velocity(x,y,0) = u*inv_spacing;
      velocity(x,y,1) = v*inv_spacing;
      
      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 
    }
  }

  delete mrf;

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nHorLabels*nVertLabels,MAKENAME(label_cost));

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
	    
          double disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);


  return energy;
}


double message_passing_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                         int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                         uint neighborhood, double lambda, std::string method,
                                         Math3D::Tensor<double>& velocity) {

  double inv_spacing = 1.0 / ((double) spacing);

  double org_lambda = lambda;
  lambda *= inv_spacing;

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  assert(xDim == second.xDim());
  assert(yDim == second.yDim());

  assert(min_x_disp <= max_x_disp);
  assert(min_y_disp <= max_y_disp);

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp + 1) * spacing - (spacing - 1);

  uint trws_fac = (method == "trws") ? 1 : 0;
  uint sg_fac = (method == "sg") ? 1 : 0;
  uint mplp_fac = (method == "mplp" || method == "msd") ? 1 : 0;

  assert(trws_fac + sg_fac + mplp_fac == 1);

  //TRWS trws(trws_fac*2*xDim*yDim, trws_fac*xDim*yDim*(1+neighborhood));
  //CumTRWS trws(trws_fac*2*xDim*yDim, trws_fac*xDim*yDim*(1+neighborhood));
  CumSeparatorTRWS trws(trws_fac*2*xDim*yDim, 0, trws_fac*xDim*yDim*(1+neighborhood));

  //NaiveTRWS trws(trws_fac*2*xDim*yDim, yxDim*yDim*(1+neighborhood));
  //NaiveFactorTRWS trws(trws_fac*2*xDim*yDim, trws_fac*xDim*yDim*(1+neighborhood));
  FactorChainDualDecomposition dual_decomp(sg_fac*2*xDim*yDim, sg_fac*xDim*yDim*(1+neighborhood));

  //FactorDualOpt facDO(mplp_fac * 2*xDim*yDim, mplp_fac * (neighborhood+1) * xDim*yDim);
  SeparatorDualOptimization facDO(mplp_fac * 2*xDim*yDim, 0, mplp_fac * (neighborhood+1) * xDim*yDim);

  /*** construct nodes (all cost are zero) ****/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      //Math1D::Vector<double> labels(nHorLabels,0.0);
      Math1D::Vector<float> labels(nHorLabels,0.0);

      if (trws_fac == 1) {
        //trws.add_node(labels);
        trws.add_var(labels);
      }
      else if (sg_fac == 1) 
        dual_decomp.add_var(labels);
      else if (mplp_fac == 1)
        facDO.add_var(labels);	

      labels.resize(nVertLabels,0.0);
      if (trws_fac == 1) {
        //trws.add_node(labels);
        trws.add_var(labels);
      }
      else if (sg_fac == 1) 
        dual_decomp.add_var(labels);
      else if (mplp_fac == 1)
        facDO.add_var(labels);
    }
  }

  Math2D::Matrix<float> data_edge(nHorLabels,nVertLabels,0.0);

  Math2D::Matrix<float> hor_smooth_edge(nHorLabels,nHorLabels);
  Math2D::Matrix<float> vert_smooth_edge(nVertLabels,nVertLabels);

  for (int l1=0; l1 < (int) nHorLabels; l1++) {
    for (int l2=0; l2 < (int) nHorLabels; l2++) {
      hor_smooth_edge(l1,l2) = lambda * fabs(l1-l2);
    }
  }

  for (int l1=0; l1 < (int) nVertLabels; l1++) {
    for (int l2=0; l2 < (int) nVertLabels; l2++) {
      vert_smooth_edge(l1,l2) = lambda * fabs(l1-l2);
    }
  }

  /*** construct edges ****/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      // 1) data edge
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
	  
          double disp_cost = 0.0;
	  
          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }
	  
          data_edge(lx,ly) = disp_cost;
        }
      }

      if (trws_fac == 1) {      
        //trws.add_edge(2*(y*xDim+x), 2*(y*xDim+x)+1, data_edge);
        trws.add_binary_factor(2*(y*xDim+x), 2*(y*xDim+x)+1, data_edge);
      }
      else if (sg_fac == 1)       
        dual_decomp.add_binary_factor(2*(y*xDim+x), 2*(y*xDim+x)+1, data_edge);
      else if (mplp_fac == 1) 
        facDO.add_generic_binary_factor(2*(y*xDim+x),2*(y*xDim+x)+1,data_edge);	
      
      /*** add smoothness edges ***/
      if (x > 0) {
        if (trws_fac == 1) {
          //trws.add_edge(2*(y*xDim+x-1), 2*(y*xDim+x), hor_smooth_edge);
          //trws.add_edge(2*(y*xDim+x-1)+1, 2*(y*xDim+x)+1, vert_smooth_edge);

          trws.add_binary_factor(2*(y*xDim+x-1), 2*(y*xDim+x), hor_smooth_edge);
          trws.add_binary_factor(2*(y*xDim+x-1)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
        else if (sg_fac == 1) {
          dual_decomp.add_binary_factor(2*(y*xDim+x-1), 2*(y*xDim+x), hor_smooth_edge);
          dual_decomp.add_binary_factor(2*(y*xDim+x-1)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
        else if (mplp_fac == 1) {
          facDO.add_generic_binary_factor(2*(y*xDim+x-1), 2*(y*xDim+x), hor_smooth_edge);
          facDO.add_generic_binary_factor(2*(y*xDim+x-1)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
      }
      if (y > 0) {
        if (trws_fac == 1) {
          //trws.add_edge(2*(y*xDim+x-xDim), 2*(y*xDim+x), hor_smooth_edge);
          //trws.add_edge(2*(y*xDim+x-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
	  
          trws.add_binary_factor(2*(y*xDim+x-xDim), 2*(y*xDim+x), hor_smooth_edge);
          trws.add_binary_factor(2*(y*xDim+x-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
        else if (sg_fac == 1) {
          dual_decomp.add_binary_factor(2*(y*xDim+x-xDim), 2*(y*xDim+x), hor_smooth_edge);
          dual_decomp.add_binary_factor(2*(y*xDim+x-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
        else if (mplp_fac == 1) {
          facDO.add_generic_binary_factor(2*(y*xDim+x-xDim), 2*(y*xDim+x), hor_smooth_edge);
          facDO.add_generic_binary_factor(2*(y*xDim+x-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
        }
      }
    }
  }

  if (neighborhood == 8) {

    hor_smooth_edge *= 1.0 / sqrt(2.0);
    vert_smooth_edge *= 1.0 / sqrt(2.0);

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        if (x > 0 && y > 0) {
          if (trws_fac == 1) {
            //trws.add_edge(2*(y*xDim+x-1-xDim), 2*(y*xDim+x), hor_smooth_edge);
            //trws.add_edge(2*(y*xDim+x-1-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
	    
            trws.add_binary_factor(2*(y*xDim+x-1-xDim), 2*(y*xDim+x), hor_smooth_edge);
            trws.add_binary_factor(2*(y*xDim+x-1-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
          }
          else if (sg_fac == 1) {
            dual_decomp.add_binary_factor(2*(y*xDim+x-1-xDim), 2*(y*xDim+x), hor_smooth_edge);
            dual_decomp.add_binary_factor(2*(y*xDim+x-1-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
          }
          else if (mplp_fac == 1) {
            facDO.add_generic_binary_factor(2*(y*xDim+x-1-xDim), 2*(y*xDim+x), hor_smooth_edge);
            facDO.add_generic_binary_factor(2*(y*xDim+x-1-xDim)+1, 2*(y*xDim+x)+1, vert_smooth_edge);
          }
        }
        if (x > 0 && y+1 < yDim) {
          if (trws_fac == 1) {
            //trws.add_edge(2*(y*xDim+x), 2*(y*xDim+x-1+xDim),hor_smooth_edge);
            //trws.add_edge(2*(y*xDim+x)+1, 2*(y*xDim+x-1+xDim)+1, vert_smooth_edge);

            trws.add_binary_factor(2*(y*xDim+x), 2*(y*xDim+x-1+xDim),hor_smooth_edge);
            trws.add_binary_factor(2*(y*xDim+x)+1, 2*(y*xDim+x-1+xDim)+1, vert_smooth_edge);
          }
          else if (sg_fac == 1) {
            dual_decomp.add_binary_factor(2*(y*xDim+x), 2*(y*xDim+x-1+xDim),hor_smooth_edge);
            dual_decomp.add_binary_factor(2*(y*xDim+x)+1, 2*(y*xDim+x-1+xDim)+1, vert_smooth_edge);
          }
          else if (mplp_fac == 1) {
            facDO.add_generic_binary_factor(2*(y*xDim+x), 2*(y*xDim+x-1+xDim),hor_smooth_edge);
            facDO.add_generic_binary_factor(2*(y*xDim+x)+1, 2*(y*xDim+x-1+xDim)+1, vert_smooth_edge);
          }
        }
      }
    }
  }

  double bound = 0.0;

  if (trws_fac == 1) {
    double bound = trws.optimize(200);
    //double bound = trws.subgradient_opt(20000,0.2);
  }
  else if (sg_fac == 1)
    double bound = dual_decomp.optimize(200,1.0);
  else {
    if (method == "mplp") {
      //facDO.dual_bcd(2500);
      facDO.optimize(2500); //testing only
    }
    else {
      //facDO.dual_bcd(2500,DUAL_BCD_MODE_MSD);
      facDO.optimize(2500); //testing only
    }
  }

  //retrieve labeling

  Math2D::Matrix<uint> labeling(xDim,yDim,0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {
      int u,v;
      if (trws_fac == 1) {
        u = trws.labeling()[2*(y*xDim+x)];
        v = trws.labeling()[2*(y*xDim+x)+1];
      }
      else if (sg_fac == 1) {
        u = dual_decomp.labeling()[2*(y*xDim+x)];
        v = dual_decomp.labeling()[2*(y*xDim+x)+1];
      }
      else {
        u = facDO.labeling()[2*(y*xDim+x)];
        v = facDO.labeling()[2*(y*xDim+x)+1];
      }

      labeling(x,y) = v*nHorLabels+u;

      velocity(x,y,0) = u*inv_spacing;
      velocity(x,y,1) = v*inv_spacing;
      
      velocity(x,y,0) += min_x_disp;
      velocity(x,y,1) += min_y_disp; 
    }
  }

  Math3D::NamedTensor<float> label_cost(xDim,yDim,nHorLabels*nVertLabels,MAKENAME(label_cost));

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
	    
          double disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            double diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  std::cerr << "discrete energy: " << motion_energy(label_cost, nHorLabels, spacing, org_lambda, 
                                                    neighborhood, labeling) << std::endl;

  discrete_motion_opt(label_cost, nHorLabels, spacing, org_lambda, neighborhood, labeling);

  return bound; 
}
