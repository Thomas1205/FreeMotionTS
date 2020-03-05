
/*** written by Thomas Schoenemann as a private person without employment, August 2011 ***/

//Assumption: every edge belongs to exactly one chain

#ifndef TRWS_HH
#define TRWS_HH

#include "vector.hh"
#include "matrix.hh"

class CumTRWSNode;

class CumTRWSEdge {
public:

  CumTRWSEdge(CumTRWSNode* from, CumTRWSNode* to, const Math2D::Matrix<float>& cost);

  double compute_forward_message(const Math1D::Vector<double>& init);

  double compute_backward_message(const Math1D::Vector<double>& init);

  void init_message();

  CumTRWSNode* from() const;

  CumTRWSNode* to() const;

  const Math1D::Vector<double>& message();

protected:

  CumTRWSNode* from_; //node with lower rank
  CumTRWSNode* to_; //node with lower rank

  Math2D::Matrix<float> cost_;

  Math1D::Vector<double> message_;
};

class CumTRWSNode {
public:

  CumTRWSNode(const Math1D::Vector<float>& cost, uint rank);

  void add_edge(CumTRWSEdge* edge);

  size_t nLabels() const;

  uint rank() const;

  uint nChains() const;

  Storage1D<CumTRWSEdge*>& outgoing_edges();

  Storage1D<CumTRWSEdge*>& incoming_edges();

  //returns an energy offset
  double average(Math1D::Vector<double>& cum_cost, uint& arg_min);

protected:
  Storage1D<CumTRWSEdge*> outgoing_edges_;
  Storage1D<CumTRWSEdge*> incoming_edges_;

  Math1D::Vector<float> cost_;
  uint rank_; //rank in the ordering
};

class CumTRWS {
public:

  CumTRWS(uint nNodes, uint nEdges);

  double optimize(uint nIter, bool quiet = false, bool terminate_when_no_progress = false);

  void add_node(const Math1D::Vector<float>& cost);

  void add_edge(uint from, uint to, const Math2D::Matrix<float>& cost);

  const Math1D::Vector<uint>& labeling() const;

protected:
  Storage1D<CumTRWSEdge*> edge_;
  Storage1D<CumTRWSNode*> node_;

  Math1D::Vector<uint> labeling_;

  uint nUsedNodes_;
  uint nUsedEdges_;
};

/*********************************************/

class TRWSNode;

class TRWSEdge {
public:

  TRWSEdge(TRWSNode* from, TRWSNode* to, const Math2D::Matrix<float>& cost);

  //returns the constant offset that was removed
  double compute_forward_message(Math1D::Vector<double>& msg);

  //returns the constant offset that was removed
  double compute_backward_message(Math1D::Vector<double>& msg);

  // TRWSEdge* prev() const;

  // TRWSEdge* next() const;

  TRWSNode* from() const;

  TRWSNode* to() const;

  // void set_prev(TRWSEdge* prev);

  // void set_next(TRWSEdge* next);

protected:

  TRWSNode* from_; //node with lower rank
  TRWSNode* to_; //node with higher rank

  //TRWSEdge* prev_; //preceeding edge in the chain (when moving forward)
  //TRWSEdge* next_; //succeeding edge in the chain (when moving forward)

  Math2D::Matrix<float> cost_;

  Math1D::Vector<double> fwd_reparameterization_;
  Math1D::Vector<double> bwd_reparameterization_;
};


class TRWSNode {
public:

  TRWSNode(const Math1D::Vector<double>& cost, uint rank);

  void add_edge(TRWSEdge* edge);

  size_t nLabels() const;

  uint rank() const;

  const Math1D::Vector<double>& cost() const;

  const Storage1D<TRWSEdge*>& outgoing_edges() const;

  const Storage1D<TRWSEdge*>& incoming_edges() const;

  //returns an energy offset
  double average_forward(double& bound);

  //returns an energy offset
  double average_backward(double& bound, uint& arg_min);

  void init();

protected:
  Storage1D<TRWSEdge*> outgoing_edges_; //needed??
  Storage1D<TRWSEdge*> incoming_edges_; //needed??

  Math1D::Vector<double> cost_;

  uint rank_; //rank in the ordering
};


class TRWS {
public:

  TRWS(uint nNodes, uint nEdges);

  //void create_chains();

  double optimize(uint nIter, bool quiet = false, bool terminate_when_no_progress = false);

  void add_node(const Math1D::Vector<double>& cost);

  void add_node(const Math1D::Vector<float>& cost);

  void add_edge(uint from, uint to, const Math2D::Matrix<float>& cost);

  const Math1D::Vector<uint>& labeling() const;

protected:
  Storage1D<TRWSEdge*> edge_;
  Storage1D<TRWSNode*> node_;

  uint nUsedNodes_;
  uint nUsedEdges_;

  Math1D::Vector<uint> labeling_;

  double constant_energy_;
};


/****************************************************/

class NaiveTRWSNode;

class NaiveTRWSEdge {
public:

  NaiveTRWSEdge(NaiveTRWSNode* from, NaiveTRWSNode* to, const Math2D::Matrix<double>& cost);

  const Math1D::Vector<double>& message();

  //returns the constant offset that was removed
  double compute_forward_message_and_reparameterize();

  //returns the constant offset that was removed
  double compute_backward_message_and_reparameterize();

  NaiveTRWSEdge* prev();

  NaiveTRWSEdge* next();

  NaiveTRWSNode* from();

  NaiveTRWSNode* to();

  void set_prev(NaiveTRWSEdge* prev);

  void set_next(NaiveTRWSEdge* next);

  //protected:

  NaiveTRWSNode* from_; //node with lower rank
  NaiveTRWSNode* to_; //node with lower rank

  NaiveTRWSEdge* prev_; //preceeding edge in the chain (when moving forward)
  NaiveTRWSEdge* next_; //succeeding edge in the chain (when moving forward)

  Math2D::Matrix<double> cost_;
  Math1D::Vector<double> message_;
};

struct ChainElement {

  NaiveTRWSEdge* incoming_;
  NaiveTRWSEdge* outgoing_;

  Math1D::Vector<double> node_parameters_;
};

class NaiveTRWSNode {
public:

  NaiveTRWSNode(const Math1D::Vector<double>& cost, uint rank);

  void add_edge(NaiveTRWSEdge* edge);

  size_t nLabels();

  uint rank();

  const Math1D::Vector<double>& cost();

  const Storage1D<NaiveTRWSEdge*>& outgoing_edges();

  const Storage1D<NaiveTRWSEdge*>& incoming_edges();

  double average();

  void set_up_chains();

  const Storage1D<ChainElement>& chains();

  double reparameterize_forward();

  double reparameterize_backward();

protected:
  Storage1D<NaiveTRWSEdge*> outgoing_edges_; //needed??
  Storage1D<NaiveTRWSEdge*> incoming_edges_; //needed??

  Math1D::Vector<double> cost_;

  Storage1D<ChainElement> chain_;

  uint rank_; //rank in the ordering
};

class NaiveTRWS {
public:

  NaiveTRWS(uint nNodes, uint nEdges);

  void create_chains();

  void optimize(uint nIter);

  void add_node(const Math1D::Vector<double>& cost);

  void add_edge(uint from, uint to, const Math2D::Matrix<double>& cost);

protected:

  double chain_energy(NaiveTRWSEdge* start_edge);

  double dual_energy();

  Storage1D<NaiveTRWSEdge*> edge_;
  Storage1D<NaiveTRWSNode*> node_;

  uint nUsedNodes_;
  uint nUsedEdges_;

  double constant_energy_;
};

#endif
