/*** written by Thomas Schoenemann as a private person without employment, August 2011 ***/

#include "trws.hh"
#include <set>


CumTRWSNode::CumTRWSNode(const Math1D::Vector<float>& cost, uint rank) :
  cost_(cost), rank_(rank)
{
}

void CumTRWSNode::add_edge(CumTRWSEdge* edge)
{

  if (edge->from() == this) {

    uint size = outgoing_edges_.size();

    outgoing_edges_.resize(size+1);
    outgoing_edges_[size] = edge;
  }
  else {
    assert(edge->to() == this);

    uint size = incoming_edges_.size();

    incoming_edges_.resize(size+1);
    incoming_edges_[size] = edge;
  }
}

size_t CumTRWSNode::nLabels() const
{
  return cost_.size();
}

uint CumTRWSNode::rank() const
{
  return rank_;
}

Storage1D<CumTRWSEdge*>& CumTRWSNode::outgoing_edges()
{
  return outgoing_edges_;
}

Storage1D<CumTRWSEdge*>& CumTRWSNode::incoming_edges()
{
  return incoming_edges_;
}

uint CumTRWSNode::nChains() const
{

  return std::max(incoming_edges_.size(),outgoing_edges_.size());
}

//returns an energy offset
double CumTRWSNode::average(Math1D::Vector<double>& cum_cost, uint& arg_min)
{

  cum_cost.resize(cost_.size());

  const uint nLabels = cum_cost.size();

  for (uint k=0; k < nLabels; k++)
    cum_cost[k] = cost_[k];

  for (uint in = 0; in < incoming_edges_.size(); in++) {
    cum_cost += incoming_edges_[in]->message();
  }

  for (uint out = 0; out < outgoing_edges_.size(); out++)
    cum_cost += outgoing_edges_[out]->message();

  double offs = 1e300;
  for (uint k=0; k < nLabels; k++) {
    if (cum_cost[k] < offs) {
      offs = cum_cost[k];
      arg_min = k;
    }
  }

  //double offs = cum_cost.min();

  //std::cerr << "cum sum: " << cum_cost_ << std::endl;

  int nChains = this->nChains();

  for (uint k=0; k < nLabels; k++) {
    cum_cost[k] = (cum_cost[k] - offs) / nChains;
  }

  return offs;
}

/***********************/

CumTRWSEdge::CumTRWSEdge(CumTRWSNode* from, CumTRWSNode* to, const Math2D::Matrix<float>& cost) :
  from_(from), to_(to), cost_(cost)
{
  from_->add_edge(this);
  to_->add_edge(this);
}

double CumTRWSEdge::compute_forward_message(const Math1D::Vector<double>& init)
{

  uint nLabels = to_->nLabels();
  uint nPrevLabels = from_->nLabels();

  Math1D::Vector<double> prev_potential = init;

  prev_potential -= message_;

  message_.resize(nLabels);

  for (uint k=0; k < nLabels; k++) {

    double best = 1e300;

    for (uint l=0; l < nPrevLabels; l++) {

      double hyp = cost_(l,k) + prev_potential[l];

      best = std::min(best,hyp);
    }

    message_[k] = best;
  }

  double offs = message_.min();

  for (uint k=0; k < nLabels; k++)
    message_[k] -= offs;

  return offs;
}

double CumTRWSEdge::compute_backward_message(const Math1D::Vector<double>& init)
{

  uint nLabels = from_->nLabels();
  uint nPrevLabels = to_->nLabels();

  Math1D::Vector<double> prev_potential = init;

  prev_potential -= message_;

  message_.resize(nLabels);

  for (uint k=0; k < nLabels; k++) {

    double best = 1e300;

    for (uint l=0; l < nPrevLabels; l++) {

      double hyp = cost_(k,l) + prev_potential[l];

      best = std::min(best,hyp);
    }

    message_[k] = best;
  }


  double offs = message_.min();

  for (uint k=0; k < nLabels; k++)
    message_[k] -= offs;

  return offs;
}

void CumTRWSEdge::init_message()
{

  message_.resize(from_->nLabels());
  message_.set_constant(0.0);
}

CumTRWSNode* CumTRWSEdge::from() const
{
  return from_;
}

CumTRWSNode* CumTRWSEdge::to() const
{
  return to_;
}

const Math1D::Vector<double>& CumTRWSEdge::message()
{
  return message_;
}

/*******************************************/

CumTRWS::CumTRWS(uint nNodes, uint nEdges) : nUsedNodes_(0), nUsedEdges_(0)
{

  node_.resize(nNodes,0);
  edge_.resize(nEdges,0);
}

const Math1D::Vector<uint>& CumTRWS::labeling() const
{
  return labeling_;
}

double CumTRWS::optimize(uint nIter, bool quiet, bool terminate_when_no_progress)
{

  double bound = 0.0;

  for (uint e=0; e < nUsedEdges_; e++)
    edge_[e]->init_message();

  if (!quiet)
    std::cerr << nUsedNodes_ << " nodes, " << nUsedEdges_ << " edges" << std::endl;

  labeling_.resize(nUsedNodes_,0);

  for (uint iter = 1; iter <= nIter; iter++) {

    //forward
    double fwd_bound = 0.0;
    for (uint v=0; v < nUsedNodes_; v++) {

      Math1D::Vector<double> cum_cost;

      fwd_bound += node_[v]->average(cum_cost,labeling_[v]);

      Storage1D<CumTRWSEdge*>& out_edges = node_[v]->outgoing_edges();

      for (uint k=0; k < out_edges.size(); k++)
        fwd_bound += out_edges[k]->compute_forward_message(cum_cost);
    }

    //std::cerr << "iteration " << iter << ", forward bound: " << fwd_bound << std::endl;

    //backward
    double bwd_bound = 0.0;
    for (int v=nUsedNodes_-1; v >= 0; v--) {

      Math1D::Vector<double> cum_cost;

      bwd_bound += node_[v]->average(cum_cost,labeling_[v]);

      Storage1D<CumTRWSEdge*>& in_edges = node_[v]->incoming_edges();

      for (uint k=0; k < in_edges.size(); k++)
        bwd_bound += in_edges[k]->compute_backward_message(cum_cost);
    }

    if (!quiet)
      std::cerr << "iteration " << iter << ", backward bound: " << bwd_bound << std::endl;

    if (terminate_when_no_progress && bound == bwd_bound) {
      break;
    }

    bound = bwd_bound;
  }
  return bound;
}

void CumTRWS::add_node(const Math1D::Vector<float>& cost)
{

  assert(nUsedNodes_+1 <= node_.size());
  node_[nUsedNodes_] = new CumTRWSNode(cost, nUsedNodes_);
  nUsedNodes_++;
}


void CumTRWS::add_edge(uint from, uint to, const Math2D::Matrix<float>& cost)
{

  assert(nUsedEdges_+1 <= edge_.size());

  if (from > to)
    edge_[nUsedEdges_] = new CumTRWSEdge(node_[to],node_[from],transpose(cost));
  else
    edge_[nUsedEdges_] = new CumTRWSEdge(node_[from],node_[to],cost);
  nUsedEdges_++;
}

/**************************************************************************************/


TRWSEdge::TRWSEdge(TRWSNode* from, TRWSNode* to, const Math2D::Matrix<float>& cost) :
  from_(from), to_(to), /*prev_(0), next_(0),*/ cost_(cost)
{

  assert(from->nLabels() == cost_.xDim());
  assert(to->nLabels() == cost_.yDim());

  assert(from_->rank() < to_->rank());

  from_->add_edge(this);
  to_->add_edge(this);

  fwd_reparameterization_.resize(cost_.xDim(),0.0);
  bwd_reparameterization_.resize(cost_.yDim(),0.0);
}

//returns the constant offset that was removed
double TRWSEdge::compute_forward_message(Math1D::Vector<double>& message)
{

  // std::cerr << "################### com. forward msg"
  // 	    << from_->rank() << " -> " << to_->rank() << std::endl;

  const uint size = cost_.yDim();

  Math1D::Vector<double> unary_cost  = from_->cost();

  //std::cerr << "initial unary cost: " << unary_cost << std::endl;
  //std::cerr << "fwd repar: " << fwd_reparameterization_
  //	    << ", last message: " << message_ << std::endl;


  unary_cost -= fwd_reparameterization_;

  //DEBUG
  // bool match = true;
  // for (uint k=0; k < fwd_reparameterization_.size(); k++) {
  //   if (fabs(fwd_reparameterization_[k] - message_[k]) > 1e-2)
  //     match = false;
  // }
  // if (!match) {
  //   std::cerr << "fwd repar: " << fwd_reparameterization_  << std::endl
  // 	      << ", last message: " << message_ << std::endl
  // 	      << ", difference: " << (fwd_reparameterization_ - message_) << std::endl;

  // }
  //END_DEBUG

  message.resize(size);

  //std::cerr << "unary: " << unary_cost << std::endl;
  //std::cerr << "cost matrix: " << cost_ << std::endl;

  //DEBUG
  // Math1D::Vector<double> prev_bwd = bwd_reparameterization_;
  // Math1D::Vector<double> best_vec(size,0.0);
  //END_DEBUG

  for (uint k=0; k < size; k++) {

    double best = 1e300;

    for (uint l=0; l < from_->nLabels(); l++) {

      double hyp = unary_cost[l] + cost_(l,k);
      best = std::min(best,hyp);
    }

    //DEBUG
    //best_vec[k] = best;
    //END_DEBUG

    message[k] = best - bwd_reparameterization_[k];
    bwd_reparameterization_[k] = best;
  }

  //std::cerr << "msg: " << message_ << std::endl;

  //DEBUG
  //Math1D::Vector<double> org_msg = message_;
  //END_DEBUG

  double cur_offs = message.min();
  for (uint k=0; k < size; k++) {
    message[k] -= cur_offs;
    bwd_reparameterization_[k] -= cur_offs;
  }

  //bwd_reparameterization_ += message_;

  //DEBUG
  // match = true;
  // for (uint k=0; k < bwd_reparameterization_.size(); k++) {
  //   if (fabs(bwd_reparameterization_[k] - message_[k]) > 1e-2)
  //     match = false;
  // }
  // if (!match) {
  //   std::cerr << "bwd repar: " << bwd_reparameterization_  << std::endl
  // 	      << ", new message: " << message_ << std::endl
  // 	      << ", difference: " << (bwd_reparameterization_ - message_) << std::endl;

  //   std::cerr << "previous repar: " << prev_bwd << std::endl;
  //   std::cerr << "best vector: " << best_vec << std::endl;
  //   std::cerr << "message without removing the constant: " << org_msg << std::endl;
  //   exit(1);
  // }
  //END_DEBUG

  return cur_offs;
}

//returns the constant offset that was removed
double TRWSEdge::compute_backward_message(Math1D::Vector<double>& message)
{

  const uint size = cost_.xDim();

  Math1D::Vector<double> unary_cost = to_->cost();

  unary_cost -= bwd_reparameterization_;

  message.resize(size);

  for (uint k=0; k < size; k++) {

    double best = 1e300;

    for (uint l=0; l < to_->nLabels(); l++) {

      double hyp = unary_cost[l] + cost_(k,l);
      best = std::min(best,hyp);
    }

    message[k] = best - fwd_reparameterization_[k];
    //fwd_reparameterization_[k] = best;
  }

  const double cur_offs = message.min();
  for (uint k=0; k < size; k++) {
    message[k] -= cur_offs;
    //fwd_reparameterization_[k] -= cur_offs;
  }

  fwd_reparameterization_ += message;

  return cur_offs;
}

// void TRWSEdge::set_prev(TRWSEdge* prev) {
//   prev_ = prev;
// }

// void TRWSEdge::set_next(TRWSEdge* next) {
//   next_ = next;
// }


TRWSNode* TRWSEdge::from() const
{
  return from_;
}

TRWSNode* TRWSEdge::to() const
{
  return to_;
}

// TRWSEdge* TRWSEdge::prev() const {
//   return prev_;
// }

// TRWSEdge* TRWSEdge::next() const {
//   return next_;
// }

/****************************************/

TRWSNode::TRWSNode(const Math1D::Vector<double>& cost, uint rank) :
  cost_(cost), rank_(rank)
{
}

void TRWSNode::add_edge(TRWSEdge* edge)
{

  if (edge->from() == this) {

    uint size = outgoing_edges_.size();

    outgoing_edges_.resize(size+1);
    outgoing_edges_[size] = edge;
  }
  else {
    assert(edge->to() == this);

    uint size = incoming_edges_.size();

    incoming_edges_.resize(size+1);
    incoming_edges_[size] = edge;
  }
}

size_t TRWSNode::nLabels() const
{
  return cost_.size();
}

uint TRWSNode::rank() const
{
  return rank_;
}

const Math1D::Vector<double>& TRWSNode::cost() const
{
  return cost_;
}

const Storage1D<TRWSEdge*>& TRWSNode::outgoing_edges() const
{
  return outgoing_edges_;
}

const Storage1D<TRWSEdge*>& TRWSNode::incoming_edges() const
{
  return incoming_edges_;
}

void TRWSNode::init()
{

  uint nChains = std::max(incoming_edges_.size(),outgoing_edges_.size());

  if (nChains >= 1) {
    for (uint k=0; k < cost_.size(); k++)
      cost_[k] *= 1.0 / nChains;
  }
}

double TRWSNode::average_forward(double& bound)
{

  //std::cerr << "############## avg. fwd" << std::endl;

  double total_offs = 0.0;

  const uint size = cost_.size();

  uint nChains = std::max(incoming_edges_.size(),outgoing_edges_.size());

  Math1D::Vector<double> sum(size,0.0);

  //std::cerr << "cost: " << cost_ << std::endl;

  Math1D::Vector<double> message;

  for (uint k=0; k < incoming_edges_.size(); k++) {

    bound += incoming_edges_[k]->compute_forward_message(message);
    sum += message;
  }

  for (uint l=0; l < size; l++)
    sum[l] += nChains * cost_[l];

  //std::cerr << "sum: " << sum << std::endl;

  double offs = sum.min();
  total_offs += offs;

  //std::cerr << "joint offs: " << offs << std::endl;

  for (uint l=0; l < size; l++)
    cost_[l] = (sum[l] - offs) / nChains;

  return total_offs;
}

double TRWSNode::average_backward(double& bound, uint& arg_min)
{

  double total_offs = 0.0;

  const uint size = cost_.size();

  //uint nChains = outgoing_edges_.size();
  uint nChains = std::max(incoming_edges_.size(),outgoing_edges_.size());

  Math1D::Vector<double> sum(cost_.size(),0.0);
  Math1D::Vector<double> message;

  for (uint k=0; k < outgoing_edges_.size(); k++) {

    bound += outgoing_edges_[k]->compute_backward_message(message);
    sum += message;
  }

  for (uint l=0; l < size; l++)
    sum[l] += nChains * cost_[l];

  double offs = 1e300;
  for (uint k=0; k < size; k++) {
    if (sum[k] < offs) {
      offs = sum[k];
      arg_min = k;
    }
  }

  //double offs = sum.min();
  total_offs += offs;

  for (uint l=0; l < size; l++) {
    cost_[l] = (sum[l]-offs) / nChains;
  }

  return total_offs;
}

/****************************************/

TRWS::TRWS(uint nNodes, uint nEdges) : nUsedNodes_(0), nUsedEdges_(0), constant_energy_(0.0)
{

  node_.resize(nNodes,0);
  edge_.resize(nEdges,0);
}

const Math1D::Vector<uint>& TRWS::labeling() const
{
  return labeling_;
}

// void TRWS::create_chains() {

//   for (uint k=0; k < nUsedEdges_; k++) {

//     if (edge_[k]->prev() == 0 && edge_[k]->next() == 0) {

//       std::set<TRWSNode*> current_nodes;
//       current_nodes.insert(edge_[k]->from());
//       current_nodes.insert(edge_[k]->to());

//       //extend prev

//       bool progress = true;

//       TRWSEdge* cur_edge = edge_[k];

//       while (progress) {

// 	progress = false;

// 	TRWSNode* prev_node = cur_edge->from();
// 	current_nodes.insert(prev_node);

// 	const Storage1D<TRWSEdge*>& in_edges = prev_node->incoming_edges();
// 	for (uint l=0; l < in_edges.size(); l++) {

// 	  TRWSEdge* hyp_edge = in_edges[l];

// 	  if (current_nodes.find(hyp_edge->from()) == current_nodes.end()
// 	      && (hyp_edge->prev() == 0) && (hyp_edge->next() == 0) ) {

// 	    progress = true;
// 	    cur_edge->set_prev(hyp_edge);
// 	    hyp_edge->set_next(cur_edge);

// 	    cur_edge = hyp_edge;

// 	    break;
// 	  }
// 	}
//       }

//       //extend next

//       progress = true;

//       cur_edge = edge_[k];

//       while (progress) {

// 	progress = false;

// 	TRWSNode* prev_node = cur_edge->to();
// 	current_nodes.insert(prev_node);

// 	const Storage1D<TRWSEdge*>& out_edges = prev_node->outgoing_edges();
// 	for (uint l=0; l < out_edges.size(); l++) {

// 	  TRWSEdge* hyp_edge = out_edges[l];

// 	  if (current_nodes.find(hyp_edge->to()) == current_nodes.end()
// 	      && (hyp_edge->prev() == 0) && (hyp_edge->next() == 0) ) {

// 	    progress = true;
// 	    cur_edge->set_next(hyp_edge);
// 	    hyp_edge->set_prev(cur_edge);

// 	    cur_edge = hyp_edge;

// 	    break;
// 	  }
// 	}
//       }
//     }
//   }

// }

double TRWS::optimize(uint nIter, bool quiet, bool terminate_when_no_progress)
{

  //create_chains();

  for (uint i=0; i < nUsedNodes_; i++)
    node_[i]->init();

  labeling_.resize(nUsedNodes_,0);

  double bound = 1e300;

  for (uint iter=1; iter <= nIter; iter++) {

    //forward
    double forward_lower = 0.0;

    //std::cerr << "prev constant: " << constant_energy_ << std::endl;

    for (uint n=0; n < nUsedNodes_; n++) {

      constant_energy_ += node_[n]->average_forward(forward_lower);
    }

    forward_lower += constant_energy_;

    //std::cerr << "iter " << iter << ", forward bound: " << forward_lower << std::endl;
    //std::cerr << "constant energy: " << constant_energy_ << std::endl;

    //backward
    double backward_lower = 0.0;

    for (int n=nUsedNodes_-1; n >= 0; n--) {

      constant_energy_ += node_[n]->average_backward(backward_lower, labeling_[n]);
    }

    backward_lower += constant_energy_;

    if (!quiet)
      std::cerr << "iter " << iter << ", backward bound: " << backward_lower << ", gain: " << (backward_lower - bound) << std::endl;

    if (terminate_when_no_progress && bound == backward_lower) {
      break;
    }

    bound = backward_lower;
  }

  return bound;
}

void TRWS::add_node(const Math1D::Vector<double>& cost)
{

  assert(nUsedNodes_+1 <= node_.size());
  node_[nUsedNodes_] = new TRWSNode(cost, nUsedNodes_);
  nUsedNodes_++;
}

void TRWS::add_node(const Math1D::Vector<float>& cost)
{

  Math1D::Vector<double> dcost(cost.size());
  for (uint k=0; k < cost.size(); k++)
    dcost[k] = cost[k];

  assert(nUsedNodes_+1 <= node_.size());
  node_[nUsedNodes_] = new TRWSNode(dcost, nUsedNodes_);
  nUsedNodes_++;
}

void TRWS::add_edge(uint from, uint to, const Math2D::Matrix<float>& cost)
{

  assert(nUsedEdges_+1 <= edge_.size());

  if (from > to)
    edge_[nUsedEdges_] = new TRWSEdge(node_[to],node_[from],transpose(cost));
  else
    edge_[nUsedEdges_] = new TRWSEdge(node_[from],node_[to],cost);
  nUsedEdges_++;
}

/*************************************************************************/


NaiveTRWSEdge::NaiveTRWSEdge(NaiveTRWSNode* from, NaiveTRWSNode* to, const Math2D::Matrix<double>& cost) :
  from_(from), to_(to), prev_(0), next_(0), cost_(cost)
{

  assert(from->nLabels() == cost_.xDim());
  assert(to->nLabels() == cost_.yDim());

  assert(from_->rank() < to_->rank());

  from_->add_edge(this);
  to_->add_edge(this);

  //NOTE: for debugging purposes we do not allocate message_ here
}

const Math1D::Vector<double>& NaiveTRWSEdge::message()
{
  return message_;
}

//returns the constant offset that was removed
double NaiveTRWSEdge::compute_forward_message_and_reparameterize()
{

  //std::cerr << "############# comp. fwd mesg. " << from_->rank() << " -> " << to_->rank() << std::endl;

  const uint size = cost_.yDim();

  message_.resize(size);

  Math1D::Vector<double> unary_cost;

  const Storage1D<ChainElement> chain = from_->chains();

  //std::cerr << "chain size: " << chain.size() << std::endl;

  for (uint k=0; k < chain.size(); k++) {

    if (chain[k].outgoing_ == this) {
      unary_cost = chain[k].node_parameters_;
      assert(chain[k].node_parameters_.size() > 0);
      break;
    }
  }

  //std::cerr << "unary_cost: " << unary_cost << std::endl;

  assert(unary_cost.size() > 0);

  for (uint k=0; k < size; k++) {

    double best = 1e300;

    for (uint l=0; l < from_->nLabels(); l++) {

      double hyp = unary_cost[l] + cost_(l,k);
      best = std::min(best,hyp);
    }

    message_[k] = best;
  }

  double offs = message_.min();
  for (uint k=0; k < size; k++)
    message_[k] -= offs;

  //std::cerr << "message: " << message_ << std::endl;

  for (uint k=0; k < size; k++) {
    for (uint l=0; l < from_->nLabels(); l++) {
      cost_(l,k) -= message_[k];
    }
  }

  return offs;
}

//returns the constant offset that was removed
double NaiveTRWSEdge::compute_backward_message_and_reparameterize()
{

  const uint size = cost_.xDim();

  message_.resize(size);

  Math1D::Vector<double> unary_cost;

  const Storage1D<ChainElement> chain = to_->chains();

  for (uint k=0; k < chain.size(); k++) {
    if (chain[k].incoming_ == this) {
      unary_cost = chain[k].node_parameters_;
      break;
    }
  }

  for (uint k=0; k < size; k++) {

    double best = 1e300;

    for (uint l=0; l < to_->nLabels(); l++) {

      double hyp = unary_cost[l] + cost_(k,l);
      best = std::min(best,hyp);
    }

    message_[k] = best;
  }

  const double offs = message_.min();
  for (uint k=0; k < size; k++)
    message_[k] -= offs;

  for (uint k=0; k < size; k++) {
    for (uint l=0; l < to_->nLabels(); l++) {
      cost_(k,l) -= message_[k];
    }
  }

  return offs;
}

void NaiveTRWSEdge::set_prev(NaiveTRWSEdge* prev)
{
  prev_ = prev;
}

void NaiveTRWSEdge::set_next(NaiveTRWSEdge* next)
{
  next_ = next;
}


NaiveTRWSNode* NaiveTRWSEdge::from()
{
  return from_;
}

NaiveTRWSNode* NaiveTRWSEdge::to()
{
  return to_;
}

NaiveTRWSEdge* NaiveTRWSEdge::prev()
{
  return prev_;
}

NaiveTRWSEdge* NaiveTRWSEdge::next()
{
  return next_;
}

/**************/

NaiveTRWSNode::NaiveTRWSNode(const Math1D::Vector<double>& cost, uint rank) :
  cost_(cost), rank_(rank)
{
}

void NaiveTRWSNode::add_edge(NaiveTRWSEdge* edge)
{

  if (edge->from() == this) {

    uint size = outgoing_edges_.size();

    outgoing_edges_.resize(size+1);
    outgoing_edges_[size] = edge;
  }
  else {
    assert(edge->to() == this);

    uint size = incoming_edges_.size();

    incoming_edges_.resize(size+1);
    incoming_edges_[size] = edge;
  }
}

size_t NaiveTRWSNode::nLabels()
{
  return cost_.size();
}

uint NaiveTRWSNode::rank()
{
  return rank_;
}

const Math1D::Vector<double>& NaiveTRWSNode::cost()
{
  return cost_;
}

const Storage1D<NaiveTRWSEdge*>& NaiveTRWSNode::outgoing_edges()
{
  return outgoing_edges_;
}

const Storage1D<NaiveTRWSEdge*>& NaiveTRWSNode::incoming_edges()
{
  return incoming_edges_;
}

const Storage1D<ChainElement>& NaiveTRWSNode::chains()
{

  return chain_;
}

void NaiveTRWSNode::set_up_chains()
{

  chain_.resize(incoming_edges_.size());

  for (uint k=0; k < incoming_edges_.size(); k++) {
    chain_[k].incoming_ = incoming_edges_[k];
    chain_[k].outgoing_ = incoming_edges_[k]->next();
  }

  for (uint k=0; k < outgoing_edges_.size(); k++) {
    if (outgoing_edges_[k]->prev() == 0) {

      uint size = chain_.size();
      chain_.resize(size+1);
      chain_[size].outgoing_ = outgoing_edges_[k];
      chain_[size].incoming_ = 0;
    }
    else {
      //DEBUG
      bool found = false;
      for (uint i=0; i < chain_.size(); i++) {

        if (chain_[i].outgoing_ == outgoing_edges_[k])
          found = true;
      }
      assert(found);
      //END_DEBUG
    }
  }

  assert(chain_.size() > 0); //the graph has to be connected

  //set initial parameters
  for (uint k=0; k < chain_.size(); k++) {

    chain_[k].node_parameters_ = cost_;
    chain_[k].node_parameters_ *= 1.0 / chain_.size();
  }
}

double NaiveTRWSNode::reparameterize_forward()
{

  double offs = 0.0;

  assert(chain_.size() > 0);

  for (uint k=0; k < chain_.size(); k++) {

    NaiveTRWSEdge* incoming = chain_[k].incoming_;

    if (incoming != 0) {

      for (uint l=0; l < cost_.size(); l++) {
        chain_[k].node_parameters_[l] += incoming->message()[l];
      }
    }

    double cur_offs = chain_[k].node_parameters_.min();

    for (uint l=0; l < cost_.size(); l++) {
      chain_[k].node_parameters_[l] -= cur_offs;
    }

    offs += cur_offs;
  }

  return offs;
}

double NaiveTRWSNode::reparameterize_backward()
{

  double offs = 0.0;

  for (uint k=0; k < chain_.size(); k++) {

    NaiveTRWSEdge* outgoing = chain_[k].outgoing_;

    if (outgoing != 0) {

      for (uint l=0; l < cost_.size(); l++) {
        chain_[k].node_parameters_[l] += outgoing->message()[l];
      }
    }

    double cur_offs = chain_[k].node_parameters_.min();

    for (uint l=0; l < cost_.size(); l++) {
      chain_[k].node_parameters_[l] -= cur_offs;
    }

    offs += cur_offs;
  }

  return offs;
}

double NaiveTRWSNode::average()
{

  double offs = 0.0;

  Math1D::Vector<double> sum (cost_.size(),0.0);

  for (uint k=0; k < chain_.size(); k++) {

    //std::cerr << "vector: " << chain_[k].node_parameters_ << std::endl;
    sum += chain_[k].node_parameters_;
  }

  //std::cerr << "sum: " << sum << std::endl;

  offs = sum.min();
  for (uint l=0; l < sum.size(); l++)
    sum[l] -= offs;

  sum *= 1.0 / chain_.size();

  for (uint k=0; k < chain_.size(); k++) {
    chain_[k].node_parameters_ = sum;
  }

  //std::cerr << "offset after averaging: " << offs << std::endl;

  return offs;
}


/**************/

NaiveTRWS::NaiveTRWS(uint nNodes, uint nEdges) : nUsedNodes_(0), nUsedEdges_(0), constant_energy_(0.0)
{

  node_.resize(nNodes,0);
  edge_.resize(nEdges,0);
}

void NaiveTRWS::create_chains()
{

  for (uint k=0; k < nUsedEdges_; k++) {

    if (edge_[k]->prev() == 0 && edge_[k]->next() == 0) {

      std::set<NaiveTRWSNode*> current_nodes;
      current_nodes.insert(edge_[k]->from());
      current_nodes.insert(edge_[k]->to());

      //extend prev

      bool progress = true;

      NaiveTRWSEdge* cur_edge = edge_[k];

      while (progress) {

        progress = false;

        NaiveTRWSNode* prev_node = cur_edge->from();
        current_nodes.insert(prev_node);

        const Storage1D<NaiveTRWSEdge*>& in_edges = prev_node->incoming_edges();
        for (uint k=0; k < in_edges.size(); k++) {

          NaiveTRWSEdge* hyp_edge = in_edges[k];

          if (current_nodes.find(hyp_edge->from()) == current_nodes.end()
              && (hyp_edge->prev() == 0) && (hyp_edge->next() == 0) ) {

            progress = true;
            cur_edge->set_prev(hyp_edge);
            hyp_edge->set_next(cur_edge);

            cur_edge = hyp_edge;

            break;
          }
        }
      }

      //extend next

      progress = true;

      cur_edge = edge_[k];

      while (progress) {

        progress = false;

        NaiveTRWSNode* prev_node = cur_edge->to();
        current_nodes.insert(prev_node);

        const Storage1D<NaiveTRWSEdge*>& out_edges = prev_node->outgoing_edges();
        for (uint k=0; k < out_edges.size(); k++) {

          NaiveTRWSEdge* hyp_edge = out_edges[k];

          if (current_nodes.find(hyp_edge->to()) == current_nodes.end()
              && (hyp_edge->prev() == 0) && (hyp_edge->next() == 0) ) {

            progress = true;
            cur_edge->set_next(hyp_edge);
            hyp_edge->set_prev(cur_edge);

            cur_edge = hyp_edge;

            break;
          }
        }
      }
    }
  }

  for (uint n=0; n < nUsedNodes_; n++)
    node_[n]->set_up_chains();
}

void NaiveTRWS::optimize(uint nIter)
{

  create_chains();

  for (uint iter=1; iter <= nIter; iter++) {

    //forward
    double forward_lower = 0.0;

    for (uint n=0; n < nUsedNodes_; n++) {

      //std::cerr << "n: " << n << std::endl;

      double offs = node_[n]->reparameterize_forward();
      constant_energy_ += offs;

      //if (iter > 1) {
      if (true) {
        constant_energy_ += node_[n]->average();
      }

      // if ((n%50) == 0) {
      // 	std::cerr << "after node " << n << ": " << constant_energy_ << std::endl;
      // }

      //std::cerr << "B" << std::endl;

      const Storage1D<NaiveTRWSEdge*>& outgoing = node_[n]->outgoing_edges();

      for (uint k=0; k < outgoing.size(); k++)
        forward_lower += outgoing[k]->compute_forward_message_and_reparameterize();
    }

    forward_lower += constant_energy_;

    std::cerr << "forward bound: " << forward_lower << std::endl;
    std::cerr << "constant energy: " << constant_energy_ << std::endl;

    double dual_bound = dual_energy();
    std::cerr << "independently computed dual energy: " << dual_bound << std::endl;

    //backward
    double backward_lower = 0.0;

    for (int n=nUsedNodes_-1; n >= 0; n--) {

      double offs = node_[n]->reparameterize_backward();
      backward_lower += offs;
      constant_energy_ += offs;

      //if (iter > 1) {
      if (true) {
        constant_energy_ += node_[n]->average();
      }

      const Storage1D<NaiveTRWSEdge*>& incoming = node_[n]->incoming_edges();

      for (uint k=0; k < incoming.size(); k++)
        incoming[k]->compute_backward_message_and_reparameterize();
    }

    backward_lower += constant_energy_;

    //std::cerr << "constant energy after backward: " << constant_energy_ << std::endl;
  }
}

void NaiveTRWS::add_node(const Math1D::Vector<double>& cost)
{

  assert(nUsedNodes_+1 <= node_.size());
  node_[nUsedNodes_] = new NaiveTRWSNode(cost, nUsedNodes_);
  nUsedNodes_++;
}

void NaiveTRWS::add_edge(uint from, uint to, const Math2D::Matrix<double>& cost)
{

  assert(from < to);
  assert(nUsedEdges_+1 <= edge_.size());

  edge_[nUsedEdges_] = new NaiveTRWSEdge(node_[from],node_[to],cost);
  nUsedEdges_++;
}


double NaiveTRWS::chain_energy(NaiveTRWSEdge* start_edge)
{

  Math1D::Vector<double> forward1;
  Math1D::Vector<double> forward2;

  assert(start_edge->prev() == 0);

  NaiveTRWSNode* from = start_edge->from();

  const Storage1D<ChainElement>& chain = from->chains();

  for (uint k=0; k < chain.size(); k++) {
    if (chain[k].outgoing_ == start_edge)
      forward1 = chain[k].node_parameters_;
  }

  NaiveTRWSEdge* cur_edge = start_edge;

  for (uint i=0; true; i++) {

    //std::cerr << "i: " << i << std::endl;

    Math1D::Vector<double>& last_cost = ((i % 2) == 0) ? forward1 : forward2;
    Math1D::Vector<double>& new_cost = ((i % 2) == 1) ? forward1 : forward2;

    NaiveTRWSNode* to = cur_edge->to();

    const Storage1D<ChainElement>& cur_chain = to->chains();

    Math1D::NamedVector<double> cur_params(MAKENAME(cur_params));
    for (uint k=0; k < cur_chain.size(); k++) {
      if (cur_chain[k].incoming_ == cur_edge)
        cur_params = cur_chain[k].node_parameters_;
    }

    uint prev_size = cur_edge->cost_.xDim();
    uint new_size = cur_edge->cost_.yDim();

    new_cost.resize(new_size);

    for (uint l_new = 0; l_new < new_size; l_new++) {

      double best = 1e300;

      for (uint l_prev = 0; l_prev < prev_size; l_prev++) {
        double hyp = cur_edge->cost_(l_prev,l_new) + last_cost[l_prev];
        best = std::min(best,hyp);
      }

      new_cost[l_new] = best + cur_params[l_new];
    }

    if (cur_edge->next() == 0)
      return new_cost.min();

    cur_edge = cur_edge->next();
  }

  assert(false);
  return 0.0; //should never be reached
}


double NaiveTRWS::dual_energy()
{

  double cost = constant_energy_;

  for (uint i=0; i < nUsedEdges_; i++) {
    if (edge_[i]->prev() == 0)
      cost += chain_energy(edge_[i]);
  }

  return cost;
}
