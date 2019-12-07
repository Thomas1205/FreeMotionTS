/**** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ****/

#ifndef MOTION_MOVES_HH
#define MOTION_MOVES_HH

#include "tensor.hh"
#include "matrix.hh"
#include "graph.h"
#include "timing.hh"
#include "submodular_energy_minimization.hh"

template <typename T>
double motion_energy(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                     uint spacing, double lambda, uint neighborhood, const Math2D::Matrix<uint>& labeling);


template <typename T>
bool motion_expansion_move(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                           uint spacing, double lambda, uint neighborhood, 
                           uint alpha, Math2D::Matrix<uint>& labeling);

template <typename T>
bool motion_swap_move(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                      uint spacing, double lambda, uint neighborhood, 
                      uint alpha, uint beta, Math2D::Matrix<uint>& labeling);

template <typename T>
double discrete_motion_opt(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                           uint spacing, double lambda, uint neighborhood, Math2D::Matrix<uint>& labeling,
                           uint nIter = MAX_UINT);



/***************** implementation **************/

template <typename T>
double motion_energy(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                     uint spacing, double lambda, uint neighborhood, const Math2D::Matrix<uint>& labeling) {

  const uint xDim = labeling.xDim();
  const uint yDim = labeling.yDim();

  Math3D::Tensor<double> flow(xDim,yDim,2);

  double inv_spacing = 1.0 / spacing;

  double energy = 0.0;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint label = labeling(x,y);

      const uint xlabel = label % nHorLabels;
      const uint ylabel = label / nHorLabels;
      
      //NOTE: currently we ignore the displacement offsets as they do not matter for the 
      //  regularity term when using a regular spacing
      flow(x,y,0) = xlabel * inv_spacing;
      flow(x,y,1) = ylabel * inv_spacing;

      energy += label_cost(x,y,label);
    }
  }

  //std::cerr << "using lambda " << lambda << std::endl;
  //std::cerr << "data cost: " << energy << std::endl;

  double diag_lambda = lambda / sqrt(2.0);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {
      
      if (x > 0) {
        energy += lambda * (fabs(flow(x,y,0) - flow(x-1,y,0))
                            + fabs(flow(x,y,1) - flow(x-1,y,1)) );
      }
      if (y > 0) {
        energy += lambda * (fabs(flow(x,y,0) - flow(x,y-1,0))
                            + fabs(flow(x,y,1) - flow(x,y-1,1)) );
      }
      if (neighborhood >= 8) {

        if (x > 0 && y > 0) {
          energy += diag_lambda * (fabs(flow(x,y,0) - flow(x-1,y-1,0))
                                   + fabs(flow(x,y,1) - flow(x-1,y-1,1)) );
        }
        if (x+1 < xDim && y > 0) {
          energy += diag_lambda * (fabs(flow(x,y,0) - flow(x+1,y-1,0))
                                   + fabs(flow(x,y,1) - flow(x+1,y-1,1)) );
        }
      }
    }
  }

  return energy;
}


template <typename T>
bool motion_expansion_move(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                           uint spacing, double lambda, uint neighborhood, 
                           uint alpha, Math2D::Matrix<uint>& labeling) {

  double inv_spacing = 1.0 / spacing;

  assert(alpha < label_cost.zDim());
  
  const uint xDim = label_cost.xDim();
  const uint yDim = label_cost.yDim();  

  double alpha_u = inv_spacing * (alpha % nHorLabels);
  double alpha_v = inv_spacing * (alpha / nHorLabels);

  Graph<double,double,double> graph(xDim*yDim+2, neighborhood*xDim*yDim);
  
  graph.add_node(xDim*yDim);

  Math3D::Tensor<double> cur_flow(xDim,yDim,2);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint cur_label = labeling(x,y);
      
      cur_flow(x,y,0) = inv_spacing * (cur_label % nHorLabels);
      cur_flow(x,y,1) = inv_spacing * (cur_label / nHorLabels);
    }
  }

  timeval tStartData,tEndData;
  gettimeofday(&tStartData,0);

  /** construct nodes and set data terms **/
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const int id = y*xDim+x;
      const uint cur_seg = labeling(x,y);

      double cur_weight = (cur_seg == alpha) ? 1e50 : label_cost(x,y,cur_seg);
      graph.add_tweights(id, label_cost(x,y,alpha), cur_weight );
      
      //double alpha_weight = (cur_seg == alpha) ? 1e50 : label_cost(x,y,alpha);
      //graph.add_tweights(id, alpha_weight, label_cost(x,y,cur_seg) );
    }
  }

  gettimeofday(&tEndData,0);
  //std::cerr << "init of tweights took " << diff_seconds(tEndData,tStartData) << " seconds. " << std::endl;

  /** construct edges and set smoothness terms **/
  const double diag_lambda = lambda / sqrt(2.0);

  double maxflow_energy = 0.0;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint cur_id = y*xDim+x;

      const double e01 = lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                   + fabs(cur_flow(x,y,1) - alpha_v) );

      double diag_e01 = diag_lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                       + fabs(cur_flow(x,y,1) - alpha_v) );

      if (x > 0) {

        double e00 = lambda * (fabs(cur_flow(x,y,0) - cur_flow(x-1,y,0))
                               + fabs(cur_flow(x,y,1) - cur_flow(x-1,y,1)) );
        double e10 = lambda * (fabs(alpha_u - cur_flow(x-1,y,0))
                               + fabs(alpha_v - cur_flow(x-1,y,1)) );
	
        maxflow_energy += add_term2(graph,cur_id, cur_id -1, e00, e01, e10, 0.0);
      }
      if (y > 0) {

        double e00 = lambda * (fabs(cur_flow(x,y,0) - cur_flow(x,y-1,0))
                               + fabs(cur_flow(x,y,1) - cur_flow(x,y-1,1)) );
        double e10 = lambda * (fabs(alpha_u - cur_flow(x,y-1,0))
                               + fabs(alpha_v - cur_flow(x,y-1,1)) );
	
        maxflow_energy += add_term2(graph,cur_id, cur_id -xDim, e00, e01, e10, 0.0);
      }

      if (neighborhood >= 8) {
        if (x > 0 && y > 0) {
	  
          double e00 = diag_lambda * (fabs(cur_flow(x,y,0) - cur_flow(x-1,y-1,0))
                                      + fabs(cur_flow(x,y,1) - cur_flow(x-1,y-1,1)) );
          double e10 = diag_lambda * (fabs(alpha_u - cur_flow(x-1,y-1,0))
                                      + fabs(alpha_v - cur_flow(x-1,y-1,1)) );
	  
          maxflow_energy += add_term2(graph,cur_id, cur_id-1-xDim, e00, diag_e01, e10, 0.0);
        }
        if (x+1 < xDim && y > 0) {

          double e00 = diag_lambda * (fabs(cur_flow(x,y,0) - cur_flow(x+1,y-1,0))
                                      + fabs(cur_flow(x,y,1) - cur_flow(x+1,y-1,1)) );
          double e10 = diag_lambda * (fabs(alpha_u - cur_flow(x+1,y-1,0))
                                      + fabs(alpha_v - cur_flow(x+1,y-1,1)) );
	  
          maxflow_energy += add_term2(graph,cur_id, cur_id+1-xDim, e00, diag_e01, e10, 0.0);
        }
      }
    }
  }  

  timeval tStartMaxflow,tEndMaxflow;
  gettimeofday(&tStartMaxflow,0);
  maxflow_energy += graph.maxflow();
  std::cerr << "maxflow energy: " << maxflow_energy << std::endl;
  gettimeofday(&tEndMaxflow,0);
  //std::cerr << "maxflow time: " << diff_seconds(tEndMaxflow,tStartMaxflow) << std::endl;

  bool changes = false;

  for (uint i=0; i < yDim*xDim; i++) {
    const Graph<double,double,double>::termtype seg = graph.what_segment(i);
    
    if (seg == Graph<double,double,double>::SINK) {
      labeling.direct_access(i) = alpha;
      changes = true;
    }
  }

  //std::cerr << "changes: " << changes << std::endl;
  return changes;

}


template <typename T>
bool motion_swap_move(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                      uint spacing, double lambda, uint neighborhood, 
                      uint alpha, uint beta, Math2D::Matrix<uint>& labeling) {

  double inv_spacing = 1.0 / spacing;

  assert(alpha < label_cost.zDim());
  
  const uint xDim = label_cost.xDim();
  const uint yDim = label_cost.yDim();  

  double alpha_u = inv_spacing * (alpha % nHorLabels);
  double alpha_v = inv_spacing * (alpha / nHorLabels);
  
  double beta_u = inv_spacing * (beta % nHorLabels);
  double beta_v = inv_spacing * (beta / nHorLabels);

  Math3D::Tensor<double> cur_flow(xDim,yDim,2);
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      const uint cur_label = labeling(x,y);
      
      cur_flow(x,y,0) = inv_spacing * (cur_label % nHorLabels);
      cur_flow(x,y,1) = inv_spacing * (cur_label / nHorLabels);
    }
  }
  
  Math2D::Matrix<uint> node_id(xDim,yDim,MAX_UINT);
  uint nParticipants = 0;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      if (labeling(x,y) == alpha || labeling(x,y) == beta) {
        node_id(x,y) = nParticipants;
        nParticipants++;
      }
    }
  }

  bool changes = false;

  if (nParticipants > 0) {

    Graph<double,double,double> graph(xDim*yDim+2, neighborhood*nParticipants);
  
    graph.add_node(nParticipants);
    
    
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
    
        if (node_id(x,y) != MAX_UINT) {

          graph.add_tweights(node_id(x,y),label_cost(x,y,beta), label_cost(x,y,alpha) );
        }
      }
    }

    /** construct edges and set smoothness terms **/
    const double diag_lambda = lambda / sqrt(2.0);

    double maxflow_energy = 0.0;

    const double e_ab = fabs(alpha_u - beta_u) + fabs(alpha_v - beta_v);
      

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
	
        const uint cur_id = y*xDim+x;

        bool cur_active = (node_id(x,y) != MAX_UINT);
	
        if (x > 0) {

          if (cur_active && node_id(x-1,y) != MAX_UINT) {
            maxflow_energy += add_term2(graph,node_id(x,y), node_id(x-1,y), 0.0, lambda*e_ab, lambda*e_ab, 0.0);
          }
          else if (cur_active) {
	    
            double ea = lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                  + fabs(cur_flow(x,y,1) - alpha_v) );
            double eb = lambda * (fabs(cur_flow(x,y,0) - beta_u)
                                  + fabs(cur_flow(x,y,1) - beta_v) );
            graph.add_tweights(node_id(x,y), eb, ea);
          }
          else if (node_id(x-1,y) != MAX_UINT) {

            double ea = lambda * (fabs(cur_flow(x-1,y,0) - alpha_u)
                                  + fabs(cur_flow(x-1,y,1) - alpha_v) );
            double eb = lambda * (fabs(cur_flow(x-1,y,0) - beta_u)
                                  + fabs(cur_flow(x-1,y,1) - beta_v) );
            graph.add_tweights(node_id(x,y), eb, ea);
          }
          else {

            maxflow_energy += lambda * (fabs(cur_flow(x,y,0) - cur_flow(x-1,y,0))
                                        + fabs(cur_flow(x,y,1) - cur_flow(x-1,y,1)) );
          }
        }
        if (y > 0) {
	  
          if (cur_active && node_id(x,y-1) != MAX_UINT) {
            maxflow_energy += add_term2(graph,node_id(x,y), node_id(x,y-1), 0.0, lambda*e_ab, lambda*e_ab, 0.0);
          }
          else if (cur_active) {
	    
            double ea = lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                  + fabs(cur_flow(x,y,1) - alpha_v) );
            double eb = lambda * (fabs(cur_flow(x,y,0) - beta_u)
                                  + fabs(cur_flow(x,y,1) - beta_v) );
            graph.add_tweights(node_id(x,y), eb, ea);
          }
          else if (node_id(x+1,y) != MAX_UINT) {

            double ea = lambda * (fabs(cur_flow(x,y-1,0) - alpha_u)
                                  + fabs(cur_flow(x,y-1,1) - alpha_v) );
            double eb = lambda * (fabs(cur_flow(x,y-1,0) - beta_u)
                                  + fabs(cur_flow(x,y-1,1) - beta_v) );
            graph.add_tweights(node_id(x,y), eb, ea);
          }
          else {

            maxflow_energy += lambda * (fabs(cur_flow(x,y,0) - cur_flow(x,y-1,0))
                                        + fabs(cur_flow(x,y,1) - cur_flow(x,y-1,1)) );;
          }

        }
	
        if (neighborhood >= 8) {
          if (x > 0 && y > 0) {

            if (cur_active && node_id(x-1,y-1) != MAX_UINT) {
              maxflow_energy += add_term2(graph,node_id(x,y), node_id(x-1,y-1), 0.0, diag_lambda*e_ab, diag_lambda*e_ab, 0.0);
            }
            else if (cur_active) {
	      
              double ea = diag_lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                         + fabs(cur_flow(x,y,1) - alpha_v) );
              double eb = diag_lambda * (fabs(cur_flow(x,y,0) - beta_u)
                                         + fabs(cur_flow(x,y,1) - beta_v) );
              graph.add_tweights(node_id(x,y), eb, ea);
            }
            else if (node_id(x+1,y) != MAX_UINT) {
	      
              double ea = diag_lambda * (fabs(cur_flow(x-1,y-1,0) - alpha_u)
                                         + fabs(cur_flow(x-1,y-1,1) - alpha_v) );
              double eb = diag_lambda * (fabs(cur_flow(x-1,y-1,0) - beta_u)
                                         + fabs(cur_flow(x-1,y-1,1) - beta_v) );
              graph.add_tweights(node_id(x,y), eb, ea);
            }
            else {
	      
              maxflow_energy += diag_lambda * (fabs(cur_flow(x,y,0) - cur_flow(x-1,y-1,0))
                                               + fabs(cur_flow(x,y,1) - cur_flow(x-1,y-1,1)) );
            }

          }
          if (x+1 < xDim && y > 0) {
	    
            if (cur_active && node_id(x+1,y-1) != MAX_UINT) {
              maxflow_energy += add_term2(graph,node_id(x,y), node_id(x+1,y-1), 0.0, diag_lambda*e_ab, diag_lambda*e_ab, 0.0);
            }
            else if (cur_active) {
	      
              double ea = diag_lambda * (fabs(cur_flow(x,y,0) - alpha_u)
                                         + fabs(cur_flow(x,y,1) - alpha_v) );
              double eb = diag_lambda * (fabs(cur_flow(x,y,0) - beta_u)
                                         + fabs(cur_flow(x,y,1) - beta_v) );
              graph.add_tweights(node_id(x,y), eb, ea);
            }
            else if (node_id(x+1,y) != MAX_UINT) {
	      
              double ea = diag_lambda * (fabs(cur_flow(x+1,y-1,0) - alpha_u)
                                         + fabs(cur_flow(x+1,y-1,1) - alpha_v) );
              double eb = diag_lambda * (fabs(cur_flow(x+1,y-1,0) - beta_u)
                                         + fabs(cur_flow(x+1,y-1,1) - beta_v) );
              graph.add_tweights(node_id(x,y), eb, ea);
            }
            else {
	      
              maxflow_energy += diag_lambda * (fabs(cur_flow(x,y,0) - cur_flow(x+1,y-1,0))
                                               + fabs(cur_flow(x,y,1) - cur_flow(x+1,y-1,1)) );
            }


          }
        }
      }
    }  

    timeval tStartMaxflow,tEndMaxflow;
    gettimeofday(&tStartMaxflow,0);
    maxflow_energy += graph.maxflow();
    std::cerr.precision(8);
    std::cerr << "maxflow energy: " << maxflow_energy << std::endl;
    gettimeofday(&tEndMaxflow,0);
    //std::cerr << "maxflow time: " << diff_seconds(tEndMaxflow,tStartMaxflow) << std::endl;
    
    for (uint i=0; i < yDim*xDim; i++) {
      const Graph<double,double,double>::termtype seg = graph.what_segment(i);
      
      if (seg == Graph<double,double,double>::SINK) {
        labeling.direct_access(i) = alpha;
        changes = true;
      }
    }
  }

  return changes;
}



template <typename T>
double discrete_motion_opt(const Math3D::Tensor<T>& label_cost, uint nHorLabels,
                           uint spacing, double lambda, uint neighborhood, Math2D::Matrix<uint>& labeling,
                           uint nIter) {

  uint nLabels = label_cost.zDim();

  double energy = 1e300;
  uint nIterWithoutChanges = 0;

  double save_energy = energy;

  for (uint iter=1; iter <= nIter; iter++) {  

#if 1
    for (uint alpha = 0; alpha < nLabels; alpha++) {

      bool changes = motion_expansion_move(label_cost, nHorLabels, spacing, lambda, neighborhood, 
                                           alpha, labeling);
  
      energy = motion_energy(label_cost, nHorLabels, spacing, lambda, neighborhood, labeling);

      std::cerr.precision(8);
      std::cerr << "energy: " << energy << std::endl;
    
      if (changes)
        nIterWithoutChanges = 0;
      else
        nIterWithoutChanges++;
     
      if (iter > 0 && nIterWithoutChanges >= (nLabels-1))
        break;
    }

    if ((iter == 1 && (nIterWithoutChanges == nLabels))
        || (nIterWithoutChanges >= nLabels-1) ) 
      break;

    if (fabs(save_energy - energy) < 0.05)
      break;

    save_energy = energy;
#else
    for (uint alpha = 0; alpha < nLabels; alpha++) {
      for (uint beta = 0; beta < nLabels; beta++) {

        if (alpha != beta) {
          bool changes = motion_swap_move(label_cost, nHorLabels, spacing, lambda, neighborhood, 
                                          alpha, beta, labeling);
	  
          energy = motion_energy(label_cost, nHorLabels, spacing, lambda, neighborhood, labeling);

          std::cerr.precision(8);
          std::cerr << "energy: " << energy << std::endl;
	  
          if (changes)
            nIterWithoutChanges = 0;
          else
            nIterWithoutChanges++;
        }
      }
    }

    if (nIterWithoutChanges > nLabels*(nLabels-1))
      break;
#endif
  }

  return energy;
}

#endif
