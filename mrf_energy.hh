/*** written by Thomas Schoenemann as an employee of Lund University, August 2010 ***/

#ifndef MRF_ENERGY_HH
#define MRF_ENERGY_HH

#include "vector.hh"
#include "matrix.hh"
#include "tensor.hh"

class SmoothnessCost {
public:

  virtual ~SmoothnessCost();

  virtual double binary_cost(int label1, int label2) const = 0;
};

class PottsSmoothness : public SmoothnessCost {
public:

  PottsSmoothness(double lambda);

  virtual double binary_cost(int label1, int label2) const;

  double lambda() const;

protected:

  double lambda_;
};

class SqrtSmoothness : public SmoothnessCost {
public:

  SqrtSmoothness(double lambda);

  virtual double binary_cost(int label1, int label2) const;

protected:
  double lambda_;
};

class TruncatedLinear : public SmoothnessCost {
public:

  TruncatedLinear(double cutoff, double lambda);

  virtual double binary_cost(int label1, int label2) const;

protected:
  
  double cutoff_;
  double lambda_;
};

class TruncatedQuadratic : public SmoothnessCost {
public:

  TruncatedQuadratic(double cutoff, double lambda);

  virtual double binary_cost(int label1, int label2) const;

protected:
  
  double cutoff_;
  double lambda_;
};

/***************************************/

class BinaryMarkovRandomField {
public:

  BinaryMarkovRandomField(const Math3D::Tensor<double>& data_term, const SmoothnessCost& binary_term,
			  uint nNeighbors);

  void add_neighbor_pair(uint x1, uint y1, uint x2, uint y2);

  void add_neighbor_pair(uint n1, uint n2);

  //returns energy of current labeling
  double cur_energy();

  void solve_lp_relaxation();

  void solve_expansion_moves();

  void solve_trws();

  void solve_own_bp();

  void solve_mplp();

  const Math2D::Matrix<uint>& solution();

protected:

  void solve_lp_relaxation_explicit_representation();

  void solve_lp_relaxation_implicit_representation();


  //returns true if any labels changed
  bool expansion_move(uint alpha);
  
  uint xDim_;
  uint yDim_;

  uint nNeighbors_;

  const Math3D::Tensor<double>& data_term_;
  const SmoothnessCost& binary_term_;

  Math2D::Matrix<uint> neighbors_;

  Math2D::Matrix<uint> solution_;

};



#endif
