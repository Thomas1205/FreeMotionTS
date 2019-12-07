/*** written by Thomas Schoenemann as an employee of Lund University, August 2010 ***/


#include "mrf_energy.hh"
#include "submodular_energy_minimization.hh"
#include "graph.h"
#include "timing.hh" 
#include "sparse_matrix_description.hh"
#include "conv_lp_solving.hh"
#include "projection.hh"
#include "factorMPBP.hh"
#include "factorDualOpt.hh"

//inclusion of TRW-S:
#include "MRFEnergy.h"
#include "typeGeneral.h"

/*virtual*/ SmoothnessCost::~SmoothnessCost() {}

PottsSmoothness::PottsSmoothness(double lambda) : lambda_(lambda) {}

/*virtual*/ double PottsSmoothness::binary_cost(int label1, int label2) const {

  return (label1 != label2) ? lambda_ : 0.0;
}

double PottsSmoothness::lambda() const {
  return lambda_;
}

SqrtSmoothness::SqrtSmoothness(double lambda)  : lambda_(lambda) {}

/*virtual*/ double SqrtSmoothness::binary_cost(int label1, int label2) const {

  double diff = (label1-label2);

  return sqrt(fabs(diff));
}


TruncatedLinear::TruncatedLinear(double cutoff, double lambda) : cutoff_(cutoff), lambda_(lambda) {}

/*virtual*/ double TruncatedLinear::binary_cost(int label1, int label2) const {

  const double diff = std::min<double>(std::abs(label1-label2), cutoff_);
  return lambda_ * diff;
}


TruncatedQuadratic::TruncatedQuadratic(double cutoff, double lambda) : cutoff_(cutoff), lambda_(lambda) {}

/*virtual*/ double TruncatedQuadratic::binary_cost(int label1, int label2) const {

  
  double diff = (label1-label2);
  diff *= diff;
  diff = std::min(diff,cutoff_);

  return lambda_ * diff;
}


/********************************************************************/

BinaryMarkovRandomField::BinaryMarkovRandomField(const Math3D::Tensor<double>& data_term, const SmoothnessCost& binary_term,
						 uint nNeighbors) : data_term_(data_term), binary_term_(binary_term),
								    neighbors_(nNeighbors,2) {

  xDim_ = data_term.xDim();
  yDim_ = data_term.yDim();
  nNeighbors_ = 0;

  solution_.resize(xDim_,yDim_,0);
}

void BinaryMarkovRandomField::add_neighbor_pair(uint x1, uint y1, uint x2, uint y2) {

  add_neighbor_pair(y1*xDim_+x1, y2*xDim_+x2);
}

void BinaryMarkovRandomField::add_neighbor_pair(uint n1, uint n2) {

  if (nNeighbors_ >= neighbors_.size())
    neighbors_.resize(neighbors_.size() + (neighbors_.size()/2) + 1, 2);

  neighbors_(nNeighbors_,0) = n1;
  neighbors_(nNeighbors_,1) = n2;

  nNeighbors_++;
}

double BinaryMarkovRandomField::cur_energy() {

  double energy = 0.0;
  for (uint y=0; y < yDim_; y++)
    for (uint x=0; x < xDim_; x++)
      energy += data_term_(x,y,solution_(x,y));
  
  for (uint n=0; n < nNeighbors_; n++) {
    
    const uint n1 = neighbors_(n,0);
    const uint n2 = neighbors_(n,1);
    energy += binary_term_.binary_cost( solution_.direct_access(n1), solution_.direct_access(n2)  );
  }

  return energy;
}

const Math2D::Matrix<uint>& BinaryMarkovRandomField::solution() {

  return solution_;
}

void BinaryMarkovRandomField::solve_expansion_moves() {

  uint nLabels = data_term_.zDim();

  uint nIterWithoutChanges = 0;
  double energy = 1e300;

  double save_energy = 1e50;

  for (uint iter=1; iter <= 1000; iter++) {

    double new_energy = cur_energy();
    if (fabs(new_energy - save_energy) < 1e-3)
      break;
    save_energy = new_energy;

    for (uint alpha = 0; alpha < nLabels; alpha++) {

      bool changes = expansion_move(alpha);
  
      energy = cur_energy();
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
  }

  //return energy;
}

bool BinaryMarkovRandomField::expansion_move(uint alpha) {

  assert(alpha < data_term_.zDim());
  
  Graph<double,double,double> graph(xDim_*yDim_+2, 2*nNeighbors_);
  
  graph.add_node(xDim_*yDim_);

  timeval tStartData,tEndData;
  gettimeofday(&tStartData,0);

  /** construct nodes and set data terms **/
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      const int id = y*xDim_+x;
      const uint cur_seg = solution_(x,y);
      
      double cur_weight = (cur_seg == alpha) ? 1e50 : data_term_(x,y,cur_seg);
      graph.add_tweights(id, data_term_(x,y,alpha), cur_weight );

      //double alpha_weight = (cur_seg == alpha) ? 1e50 : data_term_(x,y,alpha);
      //graph.add_tweights(id, alpha_weight, data_term_(x,y,cur_seg) );
    }
  }

  gettimeofday(&tEndData,0);
  //std::cerr << "init of tweights took " << diff_seconds(tEndData,tStartData) << " seconds. " << std::endl;

  double energy = 0.0;

  /** include binary terms **/
  for (uint n=0; n < nNeighbors_; n++) {

    const uint n1 = neighbors_(n,0);
    const uint n2 = neighbors_(n,1);

    const uint l1 = solution_.direct_access(n1);
    const uint l2 = solution_.direct_access(n2);

    double e00 = binary_term_.binary_cost(l1,l2);
    double e01 = binary_term_.binary_cost(l1,alpha);
    double e10 = binary_term_.binary_cost(alpha,l2);

    energy += add_term2(graph,n1,n2, e00,e01,e10,0.0);
  }

  timeval tStartMaxflow,tEndMaxflow;
  gettimeofday(&tStartMaxflow,0);
  energy += graph.maxflow();
  std::cerr << "maxflow energy: " << energy << std::endl;
  gettimeofday(&tEndMaxflow,0);
  //std::cerr << "maxflow time: " << diff_seconds(tEndMaxflow,tStartMaxflow) << std::endl;

  bool changes = false;

  for (uint i=0; i < xDim_*yDim_; i++) {
    const Graph<double,double,double>::termtype seg = graph.what_segment(i);
    
    if (seg == Graph<double,double,double>::SINK) {
      solution_.direct_access(i) = alpha;
      changes = true;
    }
  }

  //std::cerr << "changes: " << changes << std::endl;
  return changes;

}

void BinaryMarkovRandomField::solve_lp_relaxation() {

  solve_lp_relaxation_implicit_representation();
}

void BinaryMarkovRandomField::solve_lp_relaxation_explicit_representation() {

  const uint nLabels = data_term_.zDim();

  Math1D::Vector<uint> simplex_starts(xDim_*yDim_+1);
  for (uint i=0; i <= xDim_*yDim_; i++) {
    simplex_starts[i] = i*nLabels;
  }

  const uint nVars = xDim_*yDim_*nLabels + nNeighbors_*nLabels*nLabels;
  const uint nConstraints = 2*nNeighbors_*nLabels;

  Math1D::NamedVector<double> conv_solution(nVars,0.0,MAKENAME(conv_solution));
  
  Math1D::NamedVector<double> cost(nVars,0.0, MAKENAME(cost));

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
      for (uint l=0; l < nLabels; l++) {
	cost[(y*xDim_+x)*nLabels + l] = data_term_(x,y,l);
	conv_solution[(y*xDim_+x)*nLabels + l] = 1.0 / nLabels;
      }
    }
  }

  const uint binary_offs = xDim_*yDim_*nLabels;

  for (uint n=0; n < nNeighbors_; n++) {
    
    //this could be written more efficiently
    for (uint l1=0; l1 < nLabels; l1++) {
      for (uint l2=0; l2 < nLabels; l2++) {
	cost[binary_offs + n*nLabels*nLabels + l1*nLabels+l2] = binary_term_.binary_cost(l1,l2);
	conv_solution[binary_offs + n*nLabels*nLabels + l1*nLabels+l2] = 1.0 / (nLabels*nLabels);
      }
    }
  }

  const uint nMatrixEntries = nConstraints * (nLabels+1);

  SparseMatrixDescription<double> lp_descr(nMatrixEntries, nConstraints, nVars);
  
  for (uint n=0; n < nNeighbors_; n++) {

    uint n1 = neighbors_(n,0);
    uint n2 = neighbors_(n,1);

    for (uint l1 = 0; l1 < nLabels; l1++) {
      
      const uint row = 2*n*nLabels+l1;

      lp_descr.add_entry( row, n1*nLabels + l1, -1.0);
      for (uint l2 = 0; l2 < nLabels; l2++) {
	lp_descr.add_entry( row, binary_offs + n*nLabels*nLabels + l1*nLabels+l2, 1.0);
      }
    }

    for (uint l2 = 0; l2 < nLabels; l2++) {

      const uint row = (2*n+1)*nLabels+l2;

      lp_descr.add_entry( row, n2*nLabels + l2, -1.0);
      for (uint l1 = 0; l1 < nLabels; l1++) {
	lp_descr.add_entry( row, binary_offs + n*nLabels*nLabels + l1*nLabels+l2, 1.0);
      }
    }

    assert(lp_descr.nEntries() == 2*(n+1)*nLabels*(nLabels+1));
  }

  std::cerr << "calling augmented lagrangian solver" << std::endl;

  //we use the standard upper and lower bounds (1.0 and 0.0) and the standard rhs of 0.0
  eq_and_simplex_constrained_lp_solve_auglagr_nesterov(nVars, nConstraints, cost.direct_access(), 0, 0,
						       lp_descr, 0, xDim_*yDim_, simplex_starts.direct_access(),
						       conv_solution.direct_access());

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
 
      uint argmax = MAX_UINT;
      double maxval = 0.0;

      for (uint l=0; l < nLabels; l++) {

	double val = conv_solution[(y*xDim_+x)*nLabels + l];

	if (val > maxval) {
	  maxval = val;
	  argmax = l;
	}
      }
	
      solution_(x,y) = argmax;
    }
  }

}


void BinaryMarkovRandomField::solve_lp_relaxation_implicit_representation() {

  std::cerr << "lp solving with implicit storage" << std::endl;

  const uint nLabels = data_term_.zDim();
  const uint nLabelVars = xDim_*yDim_*nLabels;
  const uint nVars = nLabelVars + nNeighbors_*nLabels*nLabels;
  const uint nConstraints = 2*nNeighbors_*nLabels;

  Math1D::NamedVector<double> solution(nVars,0.0,MAKENAME(solution));

  Math1D::NamedVector<double> aux_solution(nVars,MAKENAME(aux_solution));

  Math1D::NamedVector<double> grad(nVars,MAKENAME(grad));

  Math2D::NamedMatrix<double> smooth_cost(nLabels,nLabels,MAKENAME(smooth_cost));

  for (uint l1=0; l1 < nLabels; l1++)
    for (uint l2=0; l2 < nLabels; l2++)
      smooth_cost(l1,l2) = binary_term_.binary_cost(l1,l2);
  
  const double* data_cost = data_term_.direct_access();
  const double* smoothness_cost  = smooth_cost.direct_access();

  //Math1D::NamedVector<float> cost(nVars,0.0, MAKENAME(cost));

  Math1D::NamedVector<double> lagrangian_multiplier(nConstraints, 0.0, MAKENAME(lagrangian_multiplier));

  Math1D::NamedVector<double> ax(nConstraints, 0.0, MAKENAME(ax));

  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
      for (uint l=0; l < nLabels; l++) {
	//cost[(y*xDim_+x)*nLabels + l] = data_term_(x,y,l);
	solution[(y*xDim_+x)*nLabels + l] = 1.0 / nLabels;
      }
    }
  }

  const uint binary_offs = xDim_*yDim_*nLabels;

  for (uint n=0; n < nNeighbors_; n++) {
    
    //this could be written more efficiently
    for (uint l1=0; l1 < nLabels; l1++) {
      for (uint l2=0; l2 < nLabels; l2++) {
	//cost[binary_offs + n*nLabels*nLabels + l1*nLabels+l2] = binary_term_.binary_cost(l1,l2);
	solution[binary_offs + n*nLabels*nLabels + l1*nLabels+l2] = 1.0 / (nLabels*nLabels);
      }
    }
  }

  double penalty = 100.0;

  for (uint outer_iter = 1; outer_iter <= 12; outer_iter++) {
  
    if (outer_iter > 1)
      penalty *= 1.25;
    
    std::cerr << "######### penalty " << penalty << std::endl;

    double alpha = 1e-1 / penalty;
    
    double prev_t = 1.0;
    
    const uint nInnerIter = 650;

    aux_solution = solution;

    uint iter_since_restart = 0;

    double last_energy = 1e50;

    double save_energy = 1e50;

    for (uint inner_iter = 1; inner_iter <= nInnerIter; inner_iter++) {

      /**** 1. calculate energy ****/

      double energy = 0.0;
      // for (uint v=0; v < nVars; v++)
      // energy += cost[v] * solution[v];

      for (uint v=0; v < nLabelVars; v++) 
	energy += solution[v] * data_cost[v]; 

      for (uint n=0; n < nNeighbors_; n++) {
	
	for (uint l_square = 0; l_square < nLabels*nLabels; l_square++)
	  energy += solution[binary_offs + n*nLabels*nLabels + l_square] * smoothness_cost[l_square];
      }

      ax.set_constant(0.0);

      for (uint n=0; n < nNeighbors_; n++) {
      
	uint n1 = neighbors_(n,0);
	uint n2 = neighbors_(n,1);
	
	const uint offs = binary_offs + n*nLabels*nLabels;

	for (uint l1 = 0; l1 < nLabels; l1++) {
	  
	  const uint row = 2*n*nLabels+l1;

	  double sum = - solution[n1*nLabels + l1];

	  for (uint l2 = 0; l2 < nLabels; l2++) {
	    sum += solution[offs + l1*nLabels+l2];
	  }
	  ax[row] += sum;
	}
	
	for (uint l2 = 0; l2 < nLabels; l2++) {

	  const uint row = (2*n+1)*nLabels+l2;

	  double sum = - solution[n2*nLabels + l2];
	  for (uint l1 = 0; l1 < nLabels; l1++) {
	    sum += solution[offs + l1*nLabels+l2];
	  }
	  ax[row] += sum; 
	}
      }

      double penalty_energy = 0.0;
      
      for (uint c=0; c < nConstraints; c++) {

	const double temp = ax[c];

	energy += lagrangian_multiplier[c] * temp;
	penalty_energy += temp*temp;
      }

      energy += 0.5*penalty*penalty_energy;

      std::cerr.precision(8);
      std::cerr << "energy: " << energy << std::endl;


      if ((inner_iter % 15) == 0) {

	if (fabs(save_energy - energy) < 1e-4) {

	  std::cerr << "OSCILLATION OR (SLOW) CONVERGENCE DETECTED -> CUTOFF" << std::endl;
	  break;
	}	

	save_energy = energy;
      }
      

      if (iter_since_restart >= 5 && energy > last_energy + 1e-8) {

	alpha *= 0.65;

	std::cerr << "RESTART, new alpha: " << alpha << std::endl;

	iter_since_restart = 0;

	for (uint v=0; v < nVars; v++)
	  aux_solution[v] = solution[v];
      }
      else
	iter_since_restart++;

      last_energy = energy;

      /**** 2. calculate gradient ****/
      //for (uint v=0; v < nVars; v++)
      //  grad[v] = cost[v];

      for (uint v=0; v < nLabelVars; v++) 
	grad[v] = data_cost[v]; 
      for (uint n=0; n < nNeighbors_; n++) {
	
	for (uint l_square = 0; l_square < nLabels*nLabels; l_square++)
	  grad[binary_offs + n*nLabels*nLabels + l_square] = smoothness_cost[l_square];
      }

      ax.set_constant(0.0);
      
      /*** a) multiply with matrix ***/
      for (uint n=0; n < nNeighbors_; n++) {
      
	uint n1 = neighbors_(n,0);
	uint n2 = neighbors_(n,1);
	
	const uint offs = binary_offs + n*nLabels*nLabels;

	for (uint l1 = 0; l1 < nLabels; l1++) {
	  
	  const uint row = 2*n*nLabels+l1;

	  double sum = - aux_solution[n1*nLabels + l1];

	  const uint cur_offs = offs + l1*nLabels;

	  for (uint l2 = 0; l2 < nLabels; l2++) {
	    sum += aux_solution[cur_offs + l2];
	  }
	  ax[row] += sum;	  
	}
	
	for (uint l2 = 0; l2 < nLabels; l2++) {

	  const uint row = (2*n+1)*nLabels+l2;

	  double sum = - aux_solution[n2*nLabels + l2];
	  for (uint l1 = 0; l1 < nLabels; l1++) {
	    sum += aux_solution[offs + l1*nLabels+l2];
	  }
	  ax[row] += sum; 
	}
      }

      ax *= penalty;
      ax += lagrangian_multiplier;

      /*** b) multiply with transpose matrix ***/

      for (uint n=0; n < nNeighbors_; n++) {
      
	uint n1 = neighbors_(n,0);
	uint n2 = neighbors_(n,1);
	
	const uint offs = binary_offs + n*nLabels*nLabels;

	for (uint l1 = 0; l1 < nLabels; l1++) {
	  
	  const uint row = 2*n*nLabels+l1;
	  const double cur_ax = ax[row];

	  grad[n1*nLabels + l1] -= cur_ax;

	  const uint cur_offs = offs + l1*nLabels;

	  for (uint l2 = 0; l2 < nLabels; l2++) {
	    grad[cur_offs + l2] += cur_ax;
	  }
	}
	
	for (uint l2 = 0; l2 < nLabels; l2++) {

	  const uint row = (2*n+1)*nLabels+l2;
	  const double cur_ax = ax[row];

	  grad[n2*nLabels + l2] -= cur_ax;
	  for (uint l1 = 0; l1 < nLabels; l1++) {
	    grad[offs + l1*nLabels+l2] += cur_ax;
	  }
	}
      }


      /*** 3. go in the direction of the negative gradient ***/
      grad *= alpha;
      aux_solution -= grad;
      
      /*** 4. reproject on the simplices and variable bounds ***/
      for (uint i=0; i < xDim_*yDim_; i++) {
	const uint start = i*nLabels;
	projection_on_simplex(aux_solution.direct_access()+start, nLabels);
      }
      
      for (uint k=xDim_*yDim_*nLabels; k < nVars; k++) {

	if (aux_solution[k] < 0.0)
	  aux_solution[k] = 0.0;
	else if (aux_solution[k] > 1.0)
	  aux_solution[k] = 1.0;
      }
      
      /*** 5. update nesterov variables ****/
      const double new_t = 0.5 * (1 + sqrt(1+4*prev_t*prev_t));
      const double nesterov_fac = (prev_t - 1) / new_t;
      
      for (uint i=0; i < nVars; i++) {
	
	const double old_aux = aux_solution.direct_access(i);
	aux_solution.direct_access(i) = old_aux + nesterov_fac*(old_aux - solution[i]) ;
	solution[i] = old_aux;
      }
      
      prev_t = new_t;

    } //end of inner iterations
    
    /**** update the lagrangian multiplier ***/

    ax.set_constant(0.0);
    
    for (uint n=0; n < nNeighbors_; n++) {
      
      uint n1 = neighbors_(n,0);
      uint n2 = neighbors_(n,1);
      
      const uint offs = binary_offs + n*nLabels*nLabels;
      
      for (uint l1 = 0; l1 < nLabels; l1++) {
	
	const uint row = 2*n*nLabels+l1;
	
	double sum = - solution[n1*nLabels + l1];
	
	for (uint l2 = 0; l2 < nLabels; l2++) {
	  sum += solution[offs + l1*nLabels+l2];
	}
	ax[row] += sum;
      }
      
      for (uint l2 = 0; l2 < nLabels; l2++) {
	
	const uint row = (2*n+1)*nLabels+l2;
	
	double sum = - solution[n2*nLabels + l2];
	for (uint l1 = 0; l1 < nLabels; l1++) {
	  sum += solution[offs + l1*nLabels+l2];
	}
	ax[row] += sum; 
      }
    }

    for (uint c=0; c < nConstraints; c++) {

      lagrangian_multiplier[c] += penalty * ax[c];
    } 
  }

  //extract solution by thresholding
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {

      uint argmax = MAX_UINT;
      double max_val = 0.0;

      for (uint l=0; l < nLabels; l++) {

	double cur_val = solution[(y*xDim_+x)*nLabels+l];
	if (cur_val > max_val) {
	  max_val = cur_val;
	  argmax = l;
	}
      }

      solution_(x,y) = argmax;
    }
  }

  double lp_energy = 0.0;

  for (uint v=0; v < nLabelVars; v++) 
    lp_energy += solution[v] * data_cost[v]; 

  for (uint n=0; n < nNeighbors_; n++) {
    
    for (uint l_square = 0; l_square < nLabels*nLabels; l_square++)
      lp_energy += solution[binary_offs + n*nLabels*nLabels + l_square] * smoothness_cost[l_square];
  }

  std::cerr << "lp energy: " << lp_energy << std::endl;
    
}

void BinaryMarkovRandomField::solve_trws() {

  //TODO: exploit Potts-potentials etc.

  MRFEnergy<TypeGeneral>* mrf;
  MRFEnergy<TypeGeneral>::Options options;
  TypeGeneral::REAL energy, lowerBound;
  
  const uint K = data_term_.zDim(); // number of labels
  TypeGeneral::REAL* D = new TypeGeneral::REAL[K];
  
  mrf = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());

  Math2D::Matrix<MRFEnergy<TypeGeneral>::NodeId> node_adr(xDim_,yDim_);

  // construct energy
  for (uint y=0; y < yDim_; y++) {
    for (uint x=0; x < xDim_; x++) {
      for (uint l=0; l < K; l++) {
	D[l] = data_term_(x,y,l);
      }
      node_adr(x,y) = mrf->AddNode(TypeGeneral::LocalSize(K), TypeGeneral::NodeData(D));
    }
  }

  Math1D::Vector<TypeGeneral::REAL> edata(K*K);
  for (uint k1=0; k1 < K; k1++)
    for (uint k2=0; k2 < K; k2++)
      edata[k1 + K*k2] = binary_term_.binary_cost(k1,k2);

  // EdgeData(Type type, REAL* data); // type must be GENERAL. data = pointer to array of size Ki*Kj
  // such that V(ki,kj) = data[ki + Ki*kj]

  TypeGeneral::EdgeData edge_data(TypeGeneral::GENERAL, edata.direct_access());

  for (uint n=0; n < nNeighbors_; n++) {
      
    uint n1 = neighbors_(n,0);
    uint n2 = neighbors_(n,1);

    mrf->AddEdge(node_adr.direct_access(n1), node_adr.direct_access(n2), edge_data);
  }

  options.m_iterMax = 1000; // maximum number of iterations
  mrf->Minimize_TRW_S(options, lowerBound, energy);

  //TODO: extract solution

  // done
  delete mrf;
  delete[] D;

}

void BinaryMarkovRandomField::solve_own_bp() {

  const PottsSmoothness* potts = dynamic_cast<const PottsSmoothness*> (&binary_term_);
  //const PottsSmoothness* potts = 0; 

  double potts_lambda = -1.0;
  if (potts != 0)
    potts_lambda = potts->lambda();

  FactorMPBP facBP(xDim_*yDim_,nNeighbors_);

  Math1D::Vector<float> data_term(data_term_.zDim());
  Math2D::Matrix<float> bin_term(data_term_.zDim(),data_term_.zDim());

  uint nLabels = data_term_.zDim();

  for (uint k1=0; k1 < data_term_.zDim(); k1++)
    for (uint k2=0; k2 < data_term_.zDim(); k2++)
      bin_term(k1,k2) = binary_term_.binary_cost(k1,k2);

  for (uint y=0; y < yDim_; y++) {
    std::cerr << "y: " << y << std::endl;
    for (uint x=0; x < xDim_; x++) {
      for (uint l=0; l < data_term_.zDim(); l++) {
       data_term[l] = data_term_(x,y,l);
      }
      facBP.add_var(data_term);
    }
  }

  for (uint n=0; n < nNeighbors_; n++) {
      
    std::cerr << "n: " << n << " / " << nNeighbors_ << std::endl;

    uint n1 = neighbors_(n,0);
    uint n2 = neighbors_(n,1);

    if (potts != 0) {
      facBP.add_potts_factor(n1,n2,potts_lambda);
    }
    else {
      //facBP.add_generic_binary_factor(n1,n2,bin_term);

      Math1D::Vector<uint> vars(2);
      vars[0] = n1;
      vars[1] = n2;

      Math1D::Vector<size_t> pos(2);
      pos[0] = nLabels;
      pos[1] = nLabels;
      
      VarDimStorage<float> var_cost(pos);
      for (uint l1=0; l1 < nLabels; l1++) {
	for (uint l2=0; l2 < nLabels; l2++) {
	  pos[0] = l1;
	  pos[1] = l2;

	  var_cost(pos) = bin_term(l1,l2);
	}
      }

      facBP.add_generic_factor(vars,var_cost);
    }
  }
  
  facBP.mpbp(250);

  const Math1D::Vector<uint>& labeling = facBP.labeling();

  for (uint k=0; k < labeling.size(); k++)
    solution_.direct_access(k) = labeling[k];

  std::cerr << "energy: " << cur_energy() << std::endl;
}

void BinaryMarkovRandomField::solve_mplp() {

  const PottsSmoothness* potts = dynamic_cast<const PottsSmoothness*> (&binary_term_);
  //const PottsSmoothness* potts = 0;

  double potts_lambda = -1.0;
  if (potts != 0)
    potts_lambda = potts->lambda();
  
  uint nLabels = data_term_.zDim();

  FactorDualOpt facDO(xDim_*yDim_,nNeighbors_);

  Math1D::Vector<float> data_term(data_term_.zDim());
  Math2D::Matrix<float> bin_term(data_term_.zDim(),data_term_.zDim());

  for (uint k1=0; k1 < data_term_.zDim(); k1++)
    for (uint k2=0; k2 < data_term_.zDim(); k2++)
      bin_term(k1,k2) = binary_term_.binary_cost(k1,k2);

  for (uint y=0; y < yDim_; y++) {
    std::cerr << "y: " << y << std::endl;
    for (uint x=0; x < xDim_; x++) {
      for (uint l=0; l < data_term_.zDim(); l++) {
       data_term[l] = data_term_(x,y,l);
      }
      facDO.add_var(data_term);
    }
  }

  for (uint n=0; n < nNeighbors_; n++) {
      
    std::cerr << "n: " << n << " / " << nNeighbors_ << std::endl;

    uint n1 = neighbors_(n,0);
    uint n2 = neighbors_(n,1);

    if (potts != 0) {
      facDO.add_potts_factor(n1,n2,potts_lambda);
    }
    else {
      facDO.add_generic_binary_factor(n1,n2,bin_term);

      // Math1D::Vector<uint> vars(2);
      // vars[0] = n1;
      // vars[1] = n2;

      // Math1D::Vector<size_t> pos(2);
      // pos[0] = nLabels;
      // pos[1] = nLabels;
      
      // VarDimStorage<float> var_cost(pos);
      // for (uint l1=0; l1 < nLabels; l1++) {
      // 	for (uint l2=0; l2 < nLabels; l2++) {
      // 	  pos[0] = l1;
      // 	  pos[1] = l2;

      // 	  var_cost(pos) = bin_term(l1,l2);
      // 	}
      // }

      // facDO.add_generic_factor(vars,var_cost);
    }
  }
  
  facDO.dual_bca(250);

}

