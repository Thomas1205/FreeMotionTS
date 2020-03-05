/******* written by Thomas Schoenemann as an employee of the University of Pisa, Italy, 2011 *****/
/******* continued at the University of Düsseldorf, Germany, 2012 ****/

#include "factorChainDualDecomp.hh"

#include <map>
#include <vector>
#include <set>
#include "stl_out.hh"
#include "stl_util.hh"
#include "storage_util.hh"

//#define PRIMAL_DUAL_STEPSIZE

ChainDDVar::ChainDDVar(const Math1D::Vector<float>& cost) : cost_(cost) {}

void ChainDDVar::add_factor(ChainDDFactor* factor) noexcept
{
  uint size = neighboring_factor_.size();

  neighboring_factor_.resize(size+1);
  neighboring_factor_[size] = factor;
}

void ChainDDVar::add_cost(const Math1D::Vector<float>& add_cost) noexcept
{
  if (add_cost.size() != cost_.size()) {
    INTERNAL_ERROR << "cannot add cost due to incompatible vector sizes: " << cost_.size() << " and " << add_cost.size() << std::endl;
    exit(1);
  }

  const float nC = nChains();

  for (uint i=0; i < cost_.size(); i++)
    cost_[i] += add_cost[i] / nC;
}

const Math1D::Vector<float>& ChainDDVar::cost() const noexcept
{
  return cost_;
}

uint ChainDDVar::nLabels() const noexcept
{
  return cost_.size();
}

const Storage1D<ChainDDFactor*>& ChainDDVar::neighboring_factor() const noexcept
{
  return neighboring_factor_;
}

uint ChainDDVar::nChains() const noexcept
{
  uint nChains = 0;

  for (uint k=0; k < neighboring_factor_.size(); k++) {
    if (neighboring_factor_[k]->prev_var() != this)
      nChains++;
  }

  return nChains;
}

void ChainDDVar::set_up_chains() noexcept
{
  uint nChains = 0;

  for (uint k=0; k < neighboring_factor_.size(); k++) {
    if (neighboring_factor_[k]->prev_var() != this)
      nChains++;
  }

  cost_ *= 1.0 / nChains;
}

double ChainDDVar::dual_value(uint& arg_min) noexcept
{
  Math1D::Vector<double> sum(cost_.size());
  for (uint l=0; l < cost_.size(); l++)
    sum[l] = cost_[l];

  for (uint f=0; f < neighboring_factor_.size(); f++) {

    sum += neighboring_factor_[f]->dual_vars(this);
  }

  double best = 1e300;

  for (uint l=0; l < cost_.size(); l++) {

    if (sum[l] < best) {
      best = sum[l];
      arg_min = l;
    }
  }

  return best;
}

/********************************************/

ChainDDFactor::ChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars) :
  prev_var_(0), next_var_(0), prev_factor_(0), next_factor_(0), involved_var_(involved_vars)
{
  dual_var_.resize(involved_var_.size());

  for (uint v=0; v < involved_var_.size(); v++) {
    involved_var_[v]->add_factor(this);
    dual_var_[v].resize(involved_var_[v]->nLabels(),0.0);
  }
}

/*virtual*/ ChainDDFactor::~ChainDDFactor() {}

Math1D::Vector<double>& ChainDDFactor::dual_vars(const ChainDDVar* var) noexcept
{
  for (uint k=0; k < involved_var_.size(); k++) {

    if (involved_var_[k] == var)
      return dual_var_[k];
  }

  assert(false);
  return dual_var_[0];
}

const Math1D::Vector<double>& ChainDDFactor::dual_vars(const ChainDDVar* var) const noexcept
{
  for (uint k=0; k < involved_var_.size(); k++) {

    if (involved_var_[k] == var)
      return dual_var_[k];
  }

  assert(false);
  return dual_var_[0];
}

Math1D::Vector<double>& ChainDDFactor::dual_vars(uint var) noexcept
{
  return dual_var_[var];
}

const Math1D::Vector<double>& ChainDDFactor::dual_vars(uint var) const noexcept
{
  return dual_var_[var];
}

uint ChainDDFactor::var_idx(const ChainDDVar* var) const noexcept
{
  for (uint k=0; k < involved_var_.size(); k++) {

    if (involved_var_[k] == var)
      return k;
  }

  return MAX_UINT;
}

/*virtual*/ double ChainDDFactor::compute_sum_forward(const ChainDDVar* /*incoming*/, const ChainDDVar* /*outgoing*/,
    const Math1D::Vector<double>& /*prev_forward*/,  Math1D::Vector<double>& /*forward*/, double /*mu*/) const noexcept
{
  TODO("sum-product");
}

/*virtual*/ double ChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* /*incoming*/, const ChainDDVar* /*outgoing*/,
    const Math1D::Vector<double>& /*prev_log_forward*/, Math1D::Vector<double>& /*log_forward*/, double /*mu*/) const noexcept
{
  TODO("sum-product in log-space");
}

/*virtual*/ void ChainDDFactor::compute_marginals(const ChainDDVar* /*target*/, const ChainDDVar* /*in_var1*/,
    const ChainDDVar* /*in_var2*/, const Math1D::Vector<double>& /*forward1*/,
    const Math1D::Vector<double>& /*forward2*/, double /*mu*/, Math1D::Vector<double>& /*marginals*/) const noexcept
{
  TODO("marginal computation");
}

/*virtual*/
void ChainDDFactor::compute_marginals_logspace(const ChainDDVar* /*target*/, const ChainDDVar* /*in_var1*/, const ChainDDVar* /*in_var2*/,
    const Math1D::Vector<double>& /*log_forward1*/, const Math1D::Vector<double>& /*log_forward2*/,
    double /*mu*/, Math1D::Vector<double>& /*marginals*/) const noexcept
{
  TODO("marginal computation in log-space");
}

/*virtual*/
void ChainDDFactor::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  const uint nVars = involved_var_.size();

  marginal.resize(nVars);

  for (uint v=0; v < nVars; v++) {
    compute_marginals_logspace(involved_var_[v],in_var1,in_var2,log_forward1,log_forward2,mu,marginal[v]);

    assert(marginal[v].sum() >= 0.99 && marginal[v].sum() <= 1.01);
  }
}

const Storage1D<ChainDDVar*>& ChainDDFactor::involved_vars() const noexcept
{
  return involved_var_;
}

ChainDDVar* ChainDDFactor::prev_var() const noexcept
{
  return prev_var_;
}

ChainDDVar* ChainDDFactor::next_var() const noexcept
{
  return next_var_;
}

ChainDDFactor* ChainDDFactor::prev_factor() const noexcept
{
  return prev_factor_;
}

ChainDDFactor* ChainDDFactor::next_factor() const noexcept
{
  return next_factor_;
}

void ChainDDFactor::set_prev_var(ChainDDVar* var) noexcept
{
  prev_var_ = var;
}

void ChainDDFactor::set_next_var(ChainDDVar* var) noexcept
{
  next_var_ = var;
}

void ChainDDFactor::set_prev_factor(ChainDDFactor* factor) noexcept
{
  prev_factor_ = factor;
}

void ChainDDFactor::set_next_factor(ChainDDFactor* factor) noexcept
{
  next_factor_ = factor;
}

/********************************************/

GenericChainDDFactor::GenericChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const VarDimStorage<float>& cost)
  : ChainDDFactor(involved_vars), cost_(cost)
{
  if (cost.nDims() != involved_vars.size()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
    exit(1);
  }

  for (uint v = 0; v < involved_vars.size(); v++) {
    if (cost.dim(v) < involved_vars[v]->nLabels()) {
      INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
      exit(1);
    }
  }
}

/*virtual*/ GenericChainDDFactor::~GenericChainDDFactor() {}

/*virtual*/
double GenericChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace) const noexcept
{
  const uint nVars = involved_var_.size();

  Math1D::NamedVector<uint> nLabels(nVars, MAKENAME(nLabels));

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    nLabels[k] = cur_param.size();

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  forward.set_constant(1e300);
  trace.resize(involved_var_.size(),forward.size(),MAX_UINT);

  Math1D::Vector<size_t> labeling(nVars,0);

  //std::cerr << "idx: " << idx << std::endl;

  while (true) {

    double cost = cost_(labeling);
    for (uint v=0; v < nVars; v++) {
      cost -= param[v][labeling[v]];
    }

    //std::cerr << "labeling: " << labeling << ", cost: " << cost << ", forward: " << forward << std::endl;

    if (cost < forward[labeling[idx]]) {
      forward[labeling[idx]] = cost;

      for (uint v=0; v < nVars; v++) {
        trace(v,labeling[idx]) = labeling[v];
      }
    }

    //increase labeling
    uint l;
    for (l=0; l < nVars; l++) {

      labeling[l] = (labeling[l] + 1) % nLabels[l];
      if (labeling[l] != 0)
        break;
    }

    if (l == nVars) //all zero after increase => cycle completed
      break;
  }

  //std::cerr << "trace: " << trace << std::endl;
  assert(trace.max() < MAX_UINT);

  return 0.0; //presently not removing an offset
}

/*virtual*/
double GenericChainDDFactor::cost(const Math1D::Vector<uint>& labels) const noexcept
{
  Math1D::Vector<size_t> size_t_labels(labels.size());

  for (uint k=0; k < labels.size(); k++)
    size_t_labels[k] = labels[k];

  return cost_(size_t_labels);
}

/********************************************/

BinaryChainDDFactorBase::BinaryChainDDFactorBase(const Storage1D<ChainDDVar*>& involved_vars)
  : ChainDDFactor(involved_vars)
{
  if (involved_vars.size() != 2 ) {
    INTERNAL_ERROR << "attempt to instantiate a binary factor with " << involved_vars.size() << " variables. Exiting." << std::endl;
    exit(1);
  }
}

double BinaryChainDDFactorBase::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace, const Math2D::Matrix<float>& cost) const noexcept
{
  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(2,forward.size());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double best = 1e300;
      uint arg_best = MAX_UINT;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        double hyp = cost(l1,l2) - param[1][l2];

        if (hyp < best) {
          best = hyp;
          arg_best = l2;
        }
      }

      forward[l1] = best - idx_param[l1];
      trace(0,l1) = l1;
      trace(1,l1) = arg_best;
    }

  }
  else {
    assert(idx == 1);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double best = 1e300;
      uint arg_best = MAX_UINT;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        double hyp = cost(l1,l2) - param[0][l1];

        if (hyp < best) {
          best = hyp;
          arg_best = l1;
        }
      }

      forward[l2] = best - idx_param[l2];
      trace(0,l2) = arg_best;
      trace(1,l2) = l2;
    }

  }

  // std::cerr << "prev_forward: " << prev_forward << std::endl;
  // std::cerr << "idx: " << idx << std::endl;
  // std::cerr << "cost: " << cost << std::endl;
  // std::cerr << "next forward: " << forward << std::endl;

  return 0.0; //currently not removing a constant offset
}

double BinaryChainDDFactorBase::compute_sum_forward(const ChainDDVar* in_var, const ChainDDVar* out_var, const Math1D::Vector<double>& prev_forward,
    Math1D::Vector<double>& forward, double mu, const Math2D::Matrix<float>& cost) const noexcept
{
  double log_offs = 0.0;

  //std::cerr << "binary sum-forward" << std::endl;

  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {
    Math1D::Vector<double>& cur_param = param[k];

    Math1D::Vector<double> temp(cur_param.size());

    if (involved_var_[k] == in_var) {

      for (uint l=0; l < cur_param.size(); l++) {
        temp[l] = cur_param[l] / mu;
      }
      double cur_offs = temp.max();

      for (uint l=0; l < cur_param.size(); l++) {
        cur_param[l] = std::exp(temp[l] - cur_offs); //note: signs cancel (we actually want -param for standard minimization)
        cur_param[l] *= prev_forward[l];
        cur_param[l] = std::max(1e-75,cur_param[l]); //TRIAL -- for numerical stability
      }

      log_offs += cur_offs;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++) {
        temp[l] = (cur_param[l] - cur_cost[l]) / mu;
      }
      double cur_offs = temp.max();

      for (uint l=0; l < param[k].size(); l++) {
        //param[k][l] -= involved_var_[k]->cost()[l];
        //param[k][l] = std::exp((long double) (param[k][l] / mu)); //note: signs cancel (we actually want -param for standard minimization)

        cur_param[l] = std::exp(temp[l] - cur_offs); //note: signs cancel (we actually want -param for standard minimization)
        cur_param[l] = std::max(1e-75,cur_param[l]); //TRIAL -- for numerical stability
      }

      log_offs += cur_offs;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  //std::cerr << "param: " << param << std::endl;
  //std::cerr << "idx: " << idx << std::endl;

  forward.resize_dirty(involved_var_[idx]->nLabels());

  const double cost_offs = cost.min();

  log_offs -= cost_offs / mu;

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        //std::cerr << "calculating exp(" << cost(l1,l2) << "/" << mu << ") = exp(" << (cost(l1,l2) / mu) << ") = "
        //	  << std::exp(cost(l1,l2) / mu) << std::endl;

        sum += param[1][l2] * std::exp((long double) (-(cost(l1,l2)-cost_offs) / mu));
      }

      forward[l1] = sum * param[0][l1];
    }
  }
  else {
    assert(idx == 1);

    for (uint l2 = 0; l2 < nLabels2; l2++) {
      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {
        sum += param[0][l1] * std::exp((long double) (-(cost(l1,l2)-cost_offs) / mu));
      }

      forward[l2] = sum * param[1][l2];
    }
  }

  //std::cerr << "forward: " << forward << std::endl;

  return log_offs;
}

double BinaryChainDDFactorBase::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var, const Math1D::Vector<double>& prev_log_forward, 
                                                             Math1D::Vector<double>& log_forward, double mu, const Math2D::Matrix<float>& cost) const noexcept
{
  double log_offs = 0.0;

  //std::cerr << "binary sum-forward" << std::endl;

  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {
    Math1D::Vector<double>& cur_param = param[k];

    Math1D::Vector<double> temp(cur_param.size());

    if (involved_var_[k] == in_var) {

      cur_param *= 1.0 / mu;

      cur_param += prev_log_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++)
        cur_param[l] -= cur_cost[l]; //negative cost since we maximize
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  //std::cerr << "param: " << param << std::endl;
  //std::cerr << "idx: " << idx << std::endl;

  log_forward.resize_dirty(involved_var_[idx]->nLabels());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math1D::Vector<double> temp(nLabels2);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        temp[l2] = param[1][l2] - cost(l1,l2) / mu;
      }
      const double offs = temp.max();

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        //sum += std::exp(temp[l2]-offs);

        //exp is expensive -> better use an if
        sum += (temp[l2] != offs) ? std::exp(temp[l2]-offs) : 1.0;
      }

      log_forward[l1] = offs + std::log(sum) + idx_param[l1];
    }
  }
  else {
    assert(idx == 1);

    Math1D::Vector<double> temp(nLabels1);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {
        temp[l1] = param[0][l1] - cost(l1,l2) / mu;
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        //sum += std::exp(temp[l1]-offs);

        //exp is expensive -> better use an if
        sum += (temp[l1] != offs) ? std::exp(temp[l1]-offs) : 1.0;
      }

      log_forward[l2] = offs + std::log(sum) + idx_param[l2];
    }
  }

  return log_offs;
}


void BinaryChainDDFactorBase::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, const Math2D::Matrix<float>& cost, Math1D::Vector<double>& marginal) const noexcept
{
  //pretty much like computing a sum-forward vector. Differences:
  // 1. we have two input vectors (usually forward and backward)
  // 2. we renormalize the computed vector

  //std::cerr << "binary marginals" << std::endl;

  //std::cerr << "forward 1: " << forward1 << std::endl;
  //std::cerr << "forward 2: " << forward2 << std::endl;

  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {

    Math1D::Vector<double> temp(param[k].size());

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      for (uint l=0; l < param[k].size(); l++) {
        temp[l] = (param[k][l] / mu);
      }
      double offs = temp.max();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(temp[l]-offs); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward1[l];
      }

      param[k] *= 1.0 / param[k].max();
    }
    else if (involved_var_[k] == in_var2) {

      for (uint l=0; l < param[k].size(); l++) {
        temp[l] = (param[k][l] / mu);
      }
      double offs = temp.max();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(temp[l]-offs); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward2[l];
      }

      param[k] *= 1.0 / param[k].max();
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        temp[l] = (param[k][l] - cur_cost[l]) / mu;
      }
      double offs = temp.max();

      for (uint l=0; l < param[k].size(); l++) {
        //param[k][l] -= involved_var_[k]->cost()[l];
        param[k][l] = std::exp(temp[l]-offs); //note: signs cancel (we actually want -param for standard minimization)

        param[k][l] = std::max(1e-75,param[k][l]); //TRIAL - for numerical stability
      }
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  marginal.resize_dirty(involved_var_[idx]->nLabels());

  const double cost_offs = cost.min();

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {
        sum += param[1][l2] * std::exp((long double) (-(cost(l1,l2)-cost_offs) / mu));
      }

      marginal[l1] = sum * param[0][l1];
    }
  }
  else {
    assert(idx == 1);

    for (uint l2 = 0; l2 < nLabels2; l2++) {
      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {
        sum += param[0][l1] * std::exp((long double) (-(cost(l1,l2)-cost_offs) / mu));
      }

      marginal[l2] = sum * param[1][l2];
    }
  }

  assert(marginal.sum() > 1e-305);
  assert(!isnan(marginal.sum()));

  marginal *= 1.0 / marginal.sum();
}

void BinaryChainDDFactorBase::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, const Math2D::Matrix<float>& cost, Math1D::Vector<double>& marginal) const noexcept
{
  //pretty much like computing a sum-forward vector. Differences:
  // 1. we have two input vectors (usually forward and backward)
  // 2. we renormalize the computed vector

  //std::cerr << "binary marginals log-space" << std::endl;

  //std::cerr << "forward 1: " << forward1 << std::endl;
  //std::cerr << "forward 2: " << forward2 << std::endl;

  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  marginal.resize_dirty(involved_var_[idx]->nLabels());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math1D::Vector<double> temp(nLabels2);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        temp[l2] = param[1][l2] - cost(l1,l2) / mu;
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        //sum += std::exp(temp[l2]-offs);

        //exp is expensive -> better use an if
        sum += (temp[l2] != offs) ? std::exp(temp[l2]-offs) : 1.0;
      }

      marginal[l1] = offs + std::log(sum) + idx_param[l1]; //store logs first
    }
  }
  else {
    assert(idx == 1);

    Math1D::Vector<double> temp(nLabels1);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        temp[l1] = param[0][l1] - cost(l1,l2) / mu;
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        //sum += std::exp(temp[l1]-offs);

        //exp is expensive -> better use an if
        sum += (temp[l1] != offs) ? std::exp(temp[l1]-offs) : 1.0;
      }

      marginal[l2] = offs + std::log(sum) + idx_param[l2]; //store logs first
    }
  }

  //now convert to exponential format
  const double offs = marginal.max();
  for (uint l=0; l < marginal.size(); l++) {
    //marginal[l] = std::exp(marginal[l]-offs);
    marginal[l] = (marginal[l] != offs) ? std::exp(marginal[l]-offs) : 1.0;
  }

  marginal *= 1.0 / marginal.sum(); //renormalize to get the actual marginals
}

void BinaryChainDDFactorBase::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, const Math2D::Matrix<float>& cost, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  //pretty much like computing a sum-forward vector. Differences:
  // 1. we have two input vectors (usually forward and backward)
  // 2. we renormalize the computed vector

  //std::cerr << "binary marginals log-space" << std::endl;

  //std::cerr << "forward 1: " << forward1 << std::endl;
  //std::cerr << "forward 2: " << forward2 << std::endl;

  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 2; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  for (uint idx=0; idx < 2; idx++) {
    marginal[idx].resize_dirty(involved_var_[idx]->nLabels());
  }

  //first var

  Math1D::Vector<double> temp(nLabels2);

  for (uint l1 = 0; l1 < nLabels1; l1++) {

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      temp[l2] = param[1][l2] - cost(l1,l2) / mu;
    }

    const double offs = temp.max();

    double sum = 0.0;

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      //sum += std::exp(temp[l2]-offs);

      //exp is expensive -> better use an if
      sum += (temp[l2] != offs) ? std::exp(temp[l2]-offs) : 1.0;
    }

    marginal[0][l1] = offs + std::log(sum); //store logs first, param[0] is added below
  }

  //second var

  temp.resize(nLabels1);

  for (uint l2 = 0; l2 < nLabels2; l2++) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      temp[l1] = param[0][l1] - cost(l1,l2) / mu;
    }

    const double offs = temp.max();

    double sum = 0.0;

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      //sum += std::exp(temp[l1]-offs);

      //exp is expensive -> better use an if
      sum += (temp[l1] != offs) ? std::exp(temp[l1]-offs) : 1.0;
    }

    marginal[1][l2] = offs + std::log(sum); //store logs first, param[1] is added below
  }

  //now convert to exponential format
  for (uint idx=0; idx < 2; idx++) {

    Math1D::Vector<double>& cur_marginal = marginal[idx];

    cur_marginal += param[idx];

    const double offs = cur_marginal.max();
    for (uint l=0; l < cur_marginal.size(); l++) {
      //cur_marginal[l] = std::exp(cur_marginal[l]-offs);

      cur_marginal[l] = (cur_marginal[l] != offs) ? std::exp(cur_marginal[l]-offs) : 1.0;
    }

    cur_marginal *= 1.0 / cur_marginal.sum(); //renormalize to get the actual marginals
  }
}


/******/

BinaryChainDDFactor::BinaryChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math2D::Matrix<float>& cost) :
  BinaryChainDDFactorBase(involved_vars), cost_(cost)
{
  if (cost_.xDim() < involved_vars[0]->nLabels() || cost_.yDim() < involved_vars[1]->nLabels()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
  }
}

/*virtual*/ BinaryChainDDFactor::~BinaryChainDDFactor() {}

/*virtual*/
double BinaryChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_(labeling[0],labeling[1]);
}


/*virtual */
double BinaryChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var, const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace) const noexcept
{
  return BinaryChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/*virtual*/
double BinaryChainDDFactor::compute_sum_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return BinaryChainDDFactorBase::compute_sum_forward(in_var,out_var,prev_forward,forward,mu,cost_);
}

/*virtual*/
double BinaryChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return BinaryChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}

/*virtual*/
void BinaryChainDDFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, Math1D::Vector<double>& marginals) const noexcept
{
  BinaryChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void BinaryChainDDFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  BinaryChainDDFactorBase::compute_marginals_logspace(target,in_var1, in_var2, log_forward1, log_forward2, mu, cost_, marginal);
}

/*virtual*/
void BinaryChainDDFactor::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  return BinaryChainDDFactorBase::compute_all_marginals_logspace(in_var1, in_var2, log_forward1, log_forward2, mu, cost_, marginal);
}

/*************/

BinaryChainDDRefFactor::BinaryChainDDRefFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math2D::Matrix<float>& cost) :
  BinaryChainDDFactorBase(involved_vars), cost_(cost)
{
}

/*virtual*/ BinaryChainDDRefFactor::~BinaryChainDDRefFactor() {}

/*virtual*/
double BinaryChainDDRefFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_(labeling[0],labeling[1]);
}

/*virtual */
double BinaryChainDDRefFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  assert(cost_.xDim() >= involved_var_[0]->nLabels());
  assert(cost_.yDim() >= involved_var_[1]->nLabels());

  return BinaryChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/*virtual*/
double BinaryChainDDRefFactor::compute_sum_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return BinaryChainDDFactorBase::compute_sum_forward(in_var,out_var,prev_forward,forward,mu,cost_);
}

/*virtual*/
double BinaryChainDDRefFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return BinaryChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}


/*virtual*/
void BinaryChainDDRefFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, Math1D::Vector<double>& marginals) const noexcept
{
  BinaryChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void BinaryChainDDRefFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  BinaryChainDDFactorBase::compute_marginals_logspace(target,in_var1, in_var2, log_forward1, log_forward2, mu, cost_, marginal);
}

/*virtual*/
void BinaryChainDDRefFactor::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  return BinaryChainDDFactorBase::compute_all_marginals_logspace(in_var1, in_var2, log_forward1, log_forward2, mu, cost_, marginal);
}

/********************************************/

TernaryChainDDFactorBase::TernaryChainDDFactorBase(const Storage1D<ChainDDVar*>& involved_vars)
  : ChainDDFactor(involved_vars)
{
  if (involved_vars.size() != 3 ) {
    INTERNAL_ERROR << "attempt to instantiate a ternary factor with " << involved_vars.size() << " variables. Exiting." << std::endl;
    exit(1);
  }
}

double TernaryChainDDFactorBase::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, const Math3D::Tensor<float>& cost,
    Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    forward.resize_dirty(nLabels1);
    trace.resize_dirty(3,nLabels1);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double best = 1e300;
      uint argbest2 = MAX_UINT;
      uint argbest3 = MAX_UINT;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        const double inter1 = param[1][l2];

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          double hyp = cost(l1,l2,l3) - inter1 - param[2][l3];

          if (hyp < best) {
            best = hyp;
            argbest2 = l2;
            argbest3 = l3;
          }
        }
      }

      forward[l1] = best - idx_param[l1];
      trace(0,l1) = l1;
      trace(1,l1) = argbest2;
      trace(2,l1) = argbest3;
    }
  }
  else if (idx == 1) {

    forward.resize_dirty(nLabels2);
    trace.resize_dirty(3,nLabels2);

    for (uint l2 = 0; l2 < nLabels1; l2++) {

      double best = 1e300;
      uint argbest1 = MAX_UINT;
      uint argbest3 = MAX_UINT;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        double inter1 = param[0][l1];

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          double hyp = cost(l1,l2,l3) - inter1 - param[2][l3];

          if (hyp < best) {
            best = hyp;
            argbest1 = l1;
            argbest3 = l3;
          }
        }
      }

      forward[l2] = best - idx_param[l2];
      trace(0,l2) = argbest1;
      trace(1,l2) = l2;
      trace(2,l2) = argbest3;
    }
  }
  else {
    assert(out_var == involved_var_[2]);

    forward.resize_dirty(nLabels3);
    trace.resize_dirty(3,nLabels3);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double best = 1e300;
      uint argbest1 = MAX_UINT;
      uint argbest2 = MAX_UINT;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        const double inter2 = param[1][l2];

        for (uint l1 = 0; l1 < nLabels1; l1++) {

          double hyp = cost(l1,l2,l3) - inter2 - param[0][l1];

          if (hyp < best) {
            best = hyp;
            argbest1 = l1;
            argbest2 = l2;
          }
        }
      }


      forward[l3] = best - idx_param[l3];
      trace(0,l3) = argbest1;
      trace(1,l3) = argbest2;
      trace(2,l3) = l3;
    }
  }

  return 0.0; //presently not subtracting an offset
}

double TernaryChainDDFactorBase::compute_sum_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu, const Math3D::Tensor<float>& cost) const noexcept
{
  double log_offs = 0.0;

  //std::cerr << "ternary sum-forward" << std::endl;

  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {
    if (involved_var_[k] == in_var) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= prev_forward[l];
      }
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] -= cur_cost[l];
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {
          sum += param[1][l2] * param[2][l3] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      forward[l1] = sum * param[0][l1];
    }
  }
  else if (idx == 1) {

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {
          sum += param[0][l1] * param[2][l3] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      forward[l2] = sum * param[1][l2];
    }
  }
  else {
    assert(idx == 2);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {
          sum += param[0][l1] * param[1][l2] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      forward[l3] = sum * param[2][l3];
    }
  }

  return log_offs;
}

double TernaryChainDDFactorBase::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward,
    double mu, const Math3D::Tensor<float>& cost) const noexcept
{
  double log_offs = 0.0;

  assert(in_var != out_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == out_var) {
      idx = k;
    }

    if (involved_var_[k] == in_var) {

      cur_param *= 1.0 / mu;
      cur_param += prev_log_forward;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  log_forward.resize_dirty(involved_var_[idx]->nLabels());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math2D::Matrix<double> temp(nLabels2,nLabels3);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {
          temp(l2,l3) = param[1][l2] + param[2][l3] - (cost(l1,l2,l3) / mu);
        }
      }

      double offs = temp.max();

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          //sum += std::exp(temp(l2,l3)-offs);

          //exp is expensive -> better use an if
          sum += (temp(l2,l3) != offs) ? std::exp(temp(l2,l3)-offs) : 1.0;
        }
      }

      log_forward[l1] = offs + idx_param[l1] + std::log(sum);
    }
  }
  else if (idx == 1) {

    Math2D::Matrix<double> temp(nLabels1,nLabels3);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          temp(l1,l3) = param[0][l1] + param[2][l3] - (cost(l1,l2,l3) / mu);
        }
      }

      double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          //sum += std::exp(temp(l1,l3)-offs);

          //exp is expensive -> better use an if
          sum += (temp(l1,l3) != offs) ? std::exp(temp(l1,l3)-offs) : 1.0;
        }
      }

      log_forward[l2] = offs + idx_param[l2] + std::log(sum);
    }
  }
  else {
    assert(idx == 2);

    Math2D::Matrix<double> temp(nLabels1,nLabels2);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          temp(l1,l2) = param[0][l1] + param[1][l2] - (cost(l1,l2,l3) / mu);
        }
      }

      double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          //sum += std::exp(temp(l1,l2) - offs);

          //exp is expensive -> better use an if
          sum += (temp(l1,l2) != offs) ? std::exp(temp(l1,l2)-offs) : 1.0;
        }
      }

      log_forward[l3] = offs + idx_param[l3] + std::log(sum);
    }
  }

  return log_offs;
}


void TernaryChainDDFactorBase::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, const Math3D::Tensor<float>& cost, Math1D::Vector<double>& marginal) const noexcept
{
  //pretty much like computing a sum-forward vector. Differences:
  // 1. we have two input vectors (usually forward and backward)
  // 2. we renormalize the computed vector

  //std::cerr << "ternary marginals" << std::endl;

  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward1[l];
      }
    }
    else if (involved_var_[k] == in_var2) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward2[l];
      }
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] -= cur_cost[l];
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
      }
    }
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  marginal.resize_dirty(involved_var_[idx]->nLabels());

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {
          sum += param[1][l2] * param[2][l3] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      marginal[l1] = sum * param[0][l1];
    }
  }
  else if (idx == 1) {

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {
          sum += param[0][l1] * param[2][l3] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      marginal[l2] = sum * param[1][l2];
    }
  }
  else {
    assert(idx == 2);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {
          sum += param[0][l1] * param[1][l2] * std::exp(-cost(l1,l2,l3) / mu);
        }
      }

      marginal[l3] = sum * param[2][l3];
    }
  }

  marginal *= 1.0 / marginal.sum();
}

void TernaryChainDDFactorBase::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, const Math3D::Tensor<float>& cost, Math1D::Vector<double>& marginal) const noexcept
{
  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  marginal.resize_dirty(param[idx].size());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math2D::Matrix<double> temp(nLabels2,nLabels3);

    for (uint l1=0; l1 < nLabels1; l1++) {

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          temp(l2,l3) = param[1][l2] + param[2][l3] - (cost(l1,l2,l3) / mu);
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          //sum += std::exp(temp(l2,l3)-offs);

          //exp is expensive -> better use an if
          sum += (temp(l2,l3) != offs) ? std::exp(temp(l2,l3)-offs) : 1.0;
        }
      }

      marginal[l1] = offs + idx_param[l1] + std::log(sum); //store logs for now
    }
  }
  else if (idx == 1) {

    Math2D::Matrix<double> temp(nLabels1,nLabels3);

    for (uint l2=0; l2 < nLabels2; l2++) {

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          temp(l1,l3) = param[0][l1] + param[2][l3] - (cost(l1,l2,l3) / mu);
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          //sum += std::exp(temp(l1,l3)-offs);

          //exp is expensive -> better use an if
          sum += (temp(l1,l3) != offs) ? std::exp(temp(l1,l3)-offs) : 1.0;
        }
      }

      marginal[l2] = offs + idx_param[l2] + std::log(sum); //store logs for now
    }
  }
  else {
    assert(idx == 2);

    Math2D::Matrix<double> temp(nLabels1,nLabels2);

    for (uint l3=0; l3 < nLabels3; l3++) {

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l2=0; l2 < nLabels2; l2++) {

          temp(l1,l2) = param[0][l1] + param[1][l2] - (cost(l1,l2,l3) / mu);
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l2=0; l2 < nLabels2; l2++) {

          //sum += std::exp(temp(l1,l2)-offs);

          //exp is expensive -> better use an if
          sum += (temp(l1,l2) != offs) ? std::exp(temp(l1,l2)-offs) : 1.0;
        }
      }

      marginal[l3] = offs + idx_param[l3] + std::log(sum); //store logs for now
    }
  }

  //now convert to exponential format
  const double offs = marginal.max();
  for (uint l=0; l < marginal.size(); l++) {

    //marginal[l] = std::exp(marginal[l]-offs);
    marginal[l] = (marginal[l] != offs) ? std::exp(marginal[l]-offs) : 1.0;
  }

  marginal *= 1.0 / marginal.sum(); //renormalize to get the actual marginals
}

/*********/

TernaryChainDDFactor::TernaryChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math3D::Tensor<float>& cost)
  : TernaryChainDDFactorBase(involved_vars), cost_(cost)
{
  if (cost_.xDim() < involved_vars[0]->nLabels() || cost_.yDim() < involved_vars[1]->nLabels()
      || cost_.zDim() < involved_vars[2]->nLabels()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
  }
}

/*virtual*/ TernaryChainDDFactor::~TernaryChainDDFactor() {}

/*virtual*/ double TernaryChainDDFactor::compute_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace) const noexcept
{
  return TernaryChainDDFactorBase::compute_forward(incoming,outgoing,prev_forward,cost_,forward,trace);
}

/*virtual*/ double TernaryChainDDFactor::compute_sum_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return TernaryChainDDFactorBase::compute_sum_forward(incoming,outgoing,prev_forward,forward,mu,cost_);
}

/*virtual*/
double TernaryChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return TernaryChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}

/*virtual*/
void TernaryChainDDFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, Math1D::Vector<double>& marginals) const noexcept
{
  TernaryChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void TernaryChainDDFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  TernaryChainDDFactorBase::compute_marginals_logspace(target,in_var1,in_var2,log_forward1,log_forward2,mu,cost_,marginal);
}

/*virtual*/
double TernaryChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_(labeling[0],labeling[1],labeling[2]);
}

/*********/

TernaryChainDDRefFactor::TernaryChainDDRefFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math3D::Tensor<float>& cost)
  : TernaryChainDDFactorBase(involved_vars), cost_(cost) {}

/*virtual*/ TernaryChainDDRefFactor::~TernaryChainDDRefFactor() {}

/*virtual*/ double TernaryChainDDRefFactor::compute_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  assert(cost_.xDim() >= involved_var_[0]->nLabels());
  assert(cost_.yDim() >= involved_var_[1]->nLabels());
  assert(cost_.zDim() >= involved_var_[2]->nLabels());

  return TernaryChainDDFactorBase::compute_forward(incoming,outgoing,prev_forward,cost_,forward,trace);
}

/*virtual*/
double TernaryChainDDRefFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_(labeling[0],labeling[1],labeling[2]);
}

/*virtual*/ double TernaryChainDDRefFactor::compute_sum_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return TernaryChainDDFactorBase::compute_sum_forward(incoming,outgoing,prev_forward,forward,mu,cost_);
}

/*virtual*/
double TernaryChainDDRefFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return TernaryChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}

/*virtual*/
void TernaryChainDDRefFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, Math1D::Vector<double>& marginals) const noexcept
{
  TernaryChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void TernaryChainDDRefFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  TernaryChainDDFactorBase::compute_marginals_logspace(target,in_var1,in_var2,log_forward1,log_forward2,mu,cost_,marginal);
}

/***************/

SecondDiffChainDDFactor::SecondDiffChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, float lambda)
  : ChainDDFactor(involved_vars), lambda_(lambda)
{
  if (involved_vars.size() != 3 ) {
    INTERNAL_ERROR << "attempt to instantiate a second difference factor with "
                   << involved_vars.size() << " variables. Exiting." << std::endl;
    exit(1);
  }
}

/*virtual*/ SecondDiffChainDDFactor::~SecondDiffChainDDFactor() {}

/*virtual*/
double SecondDiffChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  int diff1 = int(labeling[1]) - int(labeling[0]);
  int diff2 = int(labeling[2]) - int(labeling[1]);

  int so_diff = diff2 - diff1;

  if (abs(diff1) <= 1 && abs(diff2) <= 1 && so_diff == 0)
    return 0.0; //no cost
  else if (abs(diff1) <= 1 && abs(diff2) <= 1 && abs(so_diff) == 1)
    return lambda_;

  return 3*lambda_;
}


/*virtual*/
double SecondDiffChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace) const noexcept
{
  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 3; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    forward.resize_dirty(nLabels1);
    trace.resize_dirty(3,nLabels1);

    uint default_base2 = MAX_UINT;
    double best2 = 1e300;

    for (uint l=0; l < nLabels2; l++) {

      if (-param[1][l] < best2) {
        best2 = -param[1][l];
        default_base2 = l;
      }
    }

    uint default_base3 = MAX_UINT;
    double best3 = 1e300;

    for (uint l=0; l < nLabels3; l++) {

      if (-param[2][l] < best3) {
        best3 = -param[2][l];
        default_base3 = l;
      }
    }

    double base_cost = best2 + best3 + 3*lambda_;

    for (int l1=0; l1 < int(nLabels1); l1++) {

      double best = base_cost;

      uint best2 = default_base2;
      uint best3 = default_base3;

      for (int l2 = std::max(0,l1-1); l2 <= std::min<int>(nLabels2-1,l1+1); l2++) {

        const double p1 = -param[1][l2];

        for (int l3 = std::max(0,l2-1); l3 <= std::min<int>(nLabels3-1,l2+1); l3++) {

          assert(abs(l2-l1) <= 1);
          assert(abs(l3-l2) <= 1);

          const int so_diff = l3 - 2*l2 + l1;

          double hyp = 1e300;

          if (so_diff == 0) {
            //hyp = -param[1][l2] - param[2][l3];
            hyp = p1 - param[2][l3];
          }
          else if (abs(so_diff) <= 1) {
            //hyp = -param[1][l2] - param[2][l3] + lambda_;
            hyp = p1 - param[2][l3] + lambda_;
          }

          if (hyp < best) {
            best = hyp;
            best2 = l2;
            best3 = l3;
          }
        }
      }

      forward[l1] = best - idx_param[l1];
      trace(0,l1) = l1;
      trace(1,l1) = best2;
      trace(2,l1) = best3;
    }
  }
  else if (idx == 1) {

    forward.resize_dirty(nLabels2);
    trace.resize_dirty(3,nLabels2);


    uint default_base1 = MAX_UINT;
    double best1 = 1e300;

    for (uint l=0; l < nLabels1; l++) {

      if (-param[0][l] < best1) {
        best1 = -param[0][l];
        default_base1 = l;
      }
    }

    uint default_base3 = MAX_UINT;
    double best3 = 1e300;

    for (uint l=0; l < nLabels3; l++) {

      if (-param[2][l] < best3) {
        best3 = -param[2][l];
        default_base3 = l;
      }
    }

    double base_cost = best1 + best3 + 3*lambda_;

    for (int l2=0; l2 < int(nLabels2); l2++) {

      double best = base_cost;
      uint best1 = default_base1;
      uint best3 = default_base3;

      for (int l1 = std::max(0,l2-1); l1 <= std::min<int>(nLabels1-1,l2+1); l1++) {

        const double p1 = -param[0][l1];

        for (int l3 = std::max(0,l2-1); l3 <= std::min<int>(nLabels3-1,l2+1); l3++) {

          assert(abs(l2-l1) <= 1);
          assert(abs(l3-l2) <= 1);

          const int so_diff = l3 - 2*l2 + l1;

          double hyp = 1e300;

          if (so_diff == 0) {
            //hyp = -param[0][l1] - param[2][l3];
            hyp = p1 - param[2][l3];
          }
          else if (abs(so_diff) <= 1) {
            //hyp = -param[0][l1] - param[2][l3] + lambda_;
            hyp = p1 - param[2][l3] + lambda_;
          }

          if (hyp < best) {
            best = hyp;
            best1 = l1;
            best3 = l3;
          }
        }
      }

      forward[l2] = best - idx_param[l2];
      trace(0,l2) = best1;
      trace(1,l2) = l2;
      trace(2,l2) = best3;
    }
  }
  else {

    forward.resize_dirty(nLabels3);
    trace.resize_dirty(3,nLabels3);

    uint default_base1 = MAX_UINT;
    double best1 = 1e300;

    for (uint l=0; l < nLabels1; l++) {

      if (-param[0][l] < best1) {
        best1 = -param[0][l];
        default_base1 = l;
      }
    }

    uint default_base2 = MAX_UINT;
    double best2 = 1e300;

    for (uint l=0; l < nLabels2; l++) {

      if (-param[1][l] < best2) {
        best2 = -param[1][l];
        default_base2 = l;
      }
    }

    double base_cost = best1 + best2 + 3*lambda_;

    for (int l3=0; l3 < int(nLabels3); l3++) {

      double best = base_cost;
      uint best1 = default_base1;
      uint best2 = default_base2;

      for (int l2 = std::max(0,l3-1); l2 <= std::min<int>(nLabels2-1,l3+1); l2++) {

        const double p2 = - param[1][l2];

        for (int l1 = std::max(0,l2-1); l1 <= std::min<int>(nLabels1-1,l2+1); l1++) {

          assert(abs(l2-l1) <= 1);
          assert(abs(l3-l2) <= 1);

          const int so_diff = l3 - 2*l2 + l1;

          double hyp = 1e300;

          if (so_diff == 0) {
            //hyp = -param[0][l1] - param[1][l2];
            hyp = -param[0][l1] + p2;
          }
          else if (abs(so_diff) <= 1) {
            //hyp = -param[0][l1] - param[1][l2] + lambda_;
            hyp = -param[0][l1] + p2 + lambda_;
          }

          if (hyp < best) {
            best = hyp;
            best1 = l1;
            best2 = l2;
          }
        }
      }

      forward[l3] = best - idx_param[l3];
      trace(0,l3) = best1;
      trace(1,l3) = best2;
      trace(2,l3) = l3;
    }
  }

  return 0.0;
}


/********************************************/

FourthOrderChainDDFactorBase::FourthOrderChainDDFactorBase(const Storage1D<ChainDDVar*>& involved_vars)
  : ChainDDFactor(involved_vars)
{
  if (involved_vars.size() != 4) {
    INTERNAL_ERROR << "attempt to instantiate a 4th order factor with "
                   << involved_vars.size() << " variables. Exiting." << std::endl;
    exit(1);
  }
}

double FourthOrderChainDDFactorBase::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace, const Storage1D<Math3D::Tensor<float> >& cost) const noexcept
{
  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();
  const uint nLabels4 = involved_var_[3]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 4; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    forward.resize_dirty(nLabels1);
    trace.resize_dirty(4,nLabels1);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double best = 1e300;
      uint argbest2 = MAX_UINT;
      uint argbest3 = MAX_UINT;
      uint argbest4 = MAX_UINT;

      const Math3D::Tensor<float>& cur_cost = cost[l1];

      for (uint l3 = 0; l3 < nLabels3; l3++) {

        const double inter1 =  param[2][l3];

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          const double inter2 = inter1 + param[1][l2];

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            double hyp = cur_cost(l2,l3,l4) - inter2 - param[3][l4];

            if (hyp < best) {
              best = hyp;
              argbest2 = l2;
              argbest3 = l3;
              argbest4 = l4;
            }
          }
        }
      }


      forward[l1] = best - idx_param[l1];
      trace(0,l1) = l1;
      trace(1,l1) = argbest2;
      trace(2,l1) = argbest3;
      trace(3,l1) = argbest4;
    }
  }
  else if (idx == 1) {

    forward.resize_dirty(nLabels2);
    trace.resize_dirty(4,nLabels2);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double best = 1e300;
      uint argbest1 = MAX_UINT;
      uint argbest3 = MAX_UINT;
      uint argbest4 = MAX_UINT;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        const double inter1 = param[0][l1];

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          const double inter2 = inter1 + param[2][l3];

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            double hyp = cur_cost(l2,l3,l4) - inter2 - param[3][l4];

            if (hyp < best) {
              best = hyp;
              argbest1 = l1;
              argbest3 = l3;
              argbest4 = l4;
            }
          }
        }
      }

      forward[l2] = best - idx_param[l2];
      trace(0,l2) = argbest1;
      trace(1,l2) = l2;
      trace(2,l2) = argbest3;
      trace(3,l2) = argbest4;
    }
  }
  else if (idx == 2) {

    forward.resize_dirty(nLabels2);
    trace.resize_dirty(4,nLabels2);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double best = 1e300;
      uint argbest1 = MAX_UINT;
      uint argbest2 = MAX_UINT;
      uint argbest4 = MAX_UINT;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        const double inter1 = param[0][l1];

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          const double inter2 = inter1 + param[1][l2];

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            double hyp = cur_cost(l2,l3,l4) - inter2 - param[3][l4];

            if (hyp < best) {
              best = hyp;
              argbest1 = l1;
              argbest2 = l2;
              argbest4 = l4;
            }
          }
        }
      }

      forward[l3] = best - idx_param[l3];
      trace(0,l3) = argbest1;
      trace(1,l3) = argbest2;
      trace(2,l3) = l3;
      trace(3,l3) = argbest4;
    }
  }
  else {

    assert(idx == 3);

    forward.resize_dirty(nLabels3);
    trace.resize_dirty(4,nLabels3);

    for (uint l4 = 0; l4 < nLabels4; l4++) {

      double best = 1e300;
      uint argbest1 = MAX_UINT;
      uint argbest2 = MAX_UINT;
      uint argbest3 = MAX_UINT;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        const double inter1 = param[0][l1];

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          const double inter2 = inter1 + param[2][l3];

          for (uint l2 = 0; l2 < nLabels2; l2++) {

            double hyp = cur_cost(l2,l3,l4) - inter2 - param[1][l2];

            if (hyp < best) {
              best = hyp;
              argbest1 = l1;
              argbest2 = l2;
              argbest3 = l3;
            }
          }
        }

      }

      forward[l4] = best - idx_param[l4];
      trace(0,l4) = argbest1;
      trace(1,l4) = argbest2;
      trace(2,l4) = argbest3;
      trace(3,l4) = l4;
    }
  }

  return 0.0; //presently not subtracting an offset
}

double FourthOrderChainDDFactorBase::compute_sum_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    double mu, const Storage1D<Math3D::Tensor<float> >& cost) const noexcept
{
  double log_offs = 0.0;

  //std::cerr << "4th order sum-forward" << std::endl;

  assert(out_var != in_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();
  const uint nLabels4 = involved_var_[3]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 4; k++) {
    if (involved_var_[k] == in_var) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= prev_forward[l];
      }
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] -= cur_cost[l];
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());

  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[1][l2] * param[2][l3] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      forward[l1] = sum * param[0][l1];
    }
  }
  else if (idx == 1) {

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[0][l1] * param[2][l3] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      forward[l2] = sum * param[1][l2];
    }
  }
  else if (idx == 2) {

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[0][l1] * param[1][l2] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      forward[l3] = sum * param[2][l3];
    }
  }
  else {
    assert(idx == 3);

    for (uint l4 = 0; l4 < nLabels4; l4++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l3 = 0; l3 < nLabels3; l3++) {

            sum += param[0][l1] * param[1][l2] * param[2][l3] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      forward[l4] = sum * param[3][l4];
    }
  }

  return log_offs;
}

double FourthOrderChainDDFactorBase::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward,
    double mu, const Storage1D<Math3D::Tensor<float> >& cost) const noexcept
{
  double log_offs = 0.0;

  assert(in_var != out_var);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();
  const uint nLabels4 = involved_var_[3]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 4; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == out_var) {
      idx = k;
    }

    if (involved_var_[k] == in_var) {

      cur_param *= 1.0 / mu;
      cur_param += prev_log_forward;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/

  log_forward.resize_dirty(involved_var_[idx]->nLabels());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math3D::Tensor<double> temp(nLabels2,nLabels3,nLabels4);

    for (uint l1=0; l1 < nLabels1; l1++) {

      const Math3D::Tensor<float>& cur_cost = cost[l1];

      for (uint l2=0; l2 < nLabels2; l2++) {

        for (uint l3=0; l3 < nLabels3; l3++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            temp(l2,l3,l4) = param[1][l2] + param[2][l3] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l2=0; l2 < nLabels2; l2++) {

        for (uint l3=0; l3 < nLabels3; l3++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            //sum += std::exp(temp(l2,l3,l4) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l2,l3,l4) != offs) ? std::exp(temp(l2,l3,l4) - offs) : 1.0;
          }
        }
      }

      log_forward[l1] = offs + idx_param[l1] + std::log(sum);
    }
  }
  else if (idx == 1) {

    Math3D::Tensor<double> temp(nLabels1,nLabels3,nLabels4);

    for (uint l2=0; l2 < nLabels2; l2++) {

      for (uint l1=0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l3=0; l3 < nLabels3; l3++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            temp(l1,l3,l4) = param[0][l1] + param[2][l3] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l3=0; l3 < nLabels3; l3++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            sum += std::exp(temp(l1,l3,l4) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l3,l4) != offs) ? std::exp(temp(l1,l3,l4) - offs) : 1.0;
          }
        }
      }

      log_forward[l2] = offs + idx_param[l2] + std::log(sum);
    }
  }
  else if (idx == 2) {

    Math3D::Tensor<double> temp(nLabels1,nLabels2,nLabels4);

    for (uint l3=0; l3 < nLabels3; l3++) {

      for (uint l1=0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l2=0; l2 < nLabels2; l2++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            temp(l1,l2,l4) = param[0][l1] + param[1][l2] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l2=0; l2 < nLabels2; l2++) {

          for (uint l4=0; l4 < nLabels4; l4++) {

            //sum += std::exp(temp(l1,l2,l4) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l2,l4) != offs) ? std::exp(temp(l1,l2,l4) - offs) : 1.0;
          }
        }
      }

      log_forward[l3] = offs + idx_param[l3] + std::log(sum);
    }
  }
  else {

    assert(idx == 3);

    Math3D::Tensor<double> temp(nLabels1,nLabels2,nLabels3);

    for (uint l4=0; l4 < nLabels4; l4++) {

      for (uint l1=0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l2=0; l2 < nLabels2; l2++) {

          for (uint l3=0; l3 < nLabels3; l3++) {

            temp(l1,l2,l3) = param[0][l1] + param[1][l2] + param[2][l3] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1=0; l1 < nLabels1; l1++) {

        for (uint l2=0; l2 < nLabels2; l2++) {

          for (uint l3=0; l3 < nLabels3; l3++) {

            //sum += std::exp(temp(l1,l2,l3) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l2,l3) != offs) ? std::exp(temp(l1,l2,l3) - offs) : 1.0;
          }
        }
      }

      log_forward[l4] = offs + idx_param[l4] + std::log(sum);
    }
  }

  return log_offs;
}

void FourthOrderChainDDFactorBase::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, const Storage1D<Math3D::Tensor<float> >& cost, Math1D::Vector<double>& marginals) const noexcept
{
  //pretty much like computing a sum-forward vector. Differences:
  // 1. we have two input vectors (usually forward and backward)
  // 2. we renormalize the computed vector

  //std::cerr << "4th order marginals" << std::endl;

  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();
  const uint nLabels4 = involved_var_[3]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 4; k++) {

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward1[l];
      }
    }
    else if (involved_var_[k] == in_var2) {
      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
        param[k][l] *= forward2[l];
      }
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < param[k].size(); l++) {
        param[k][l] -= cur_cost[l];
        param[k][l] = std::exp(param[k][l] / mu); //note: signs cancel (we actually want -param for standard minimization)
      }
    }
  }

  marginals.resize_dirty(involved_var_[idx]->nLabels());

  /*** now that we have param, the rest is just like sending a forward vector and renormalizing to get marginals ***/
  if (idx == 0) {

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[1][l2] * param[2][l3] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      marginals[l1] = sum * param[0][l1];
    }
  }
  else if (idx == 1) {

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[0][l1] * param[2][l3] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      marginals[l2] = sum * param[1][l2];
    }
  }
  else if (idx == 2) {

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            sum += param[0][l1] * param[1][l2] * param[3][l4] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      marginals[l3] = sum * param[2][l3];
    }
  }
  else {
    assert(idx == 3);

    for (uint l4 = 0; l4 < nLabels4; l4++) {

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l3 = 0; l3 < nLabels3; l3++) {

            sum += param[0][l1] * param[1][l2] * param[2][l3] * std::exp(-cost[l1](l2,l3,l4) / mu);
          }
        }
      }

      marginals[l4] = sum * param[3][l4];
    }
  }

  marginals *= 1.0 / marginals.sum();
}


void FourthOrderChainDDFactorBase::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, const Storage1D<Math3D::Tensor<float> >& cost, Math1D::Vector<double>& marginal) const noexcept
{
  assert(in_var1 == 0 || in_var2 == 0 || in_var1 != in_var2);

  const uint nLabels1 = involved_var_[0]->nLabels();
  const uint nLabels2 = involved_var_[1]->nLabels();
  const uint nLabels3 = involved_var_[2]->nLabels();
  const uint nLabels4 = involved_var_[3]->nLabels();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < 4; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  marginal.resize_dirty(involved_var_[idx]->nLabels());

  const Math1D::Vector<double>& idx_param = param[idx];

  if (idx == 0) {

    Math3D::Tensor<double> temp(nLabels2,nLabels3,nLabels4);

    for (uint l1 = 0; l1 < nLabels1; l1++) {

      const Math3D::Tensor<float>& cur_cost = cost[l1];

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            temp(l2,l3,l4) = param[1][l2] + param[2][l3] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l2 = 0; l2 < nLabels2; l2++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            //sum += std::exp(temp(l2,l3,l4)-offs);

            //exp is expensive -> better use an if
            sum += (temp(l2,l3,l4) != offs) ? std::exp(temp(l2,l3,l4) - offs) : 1.0;
          }
        }
      }

      marginal[l1] = offs + idx_param[l1] + std::log(sum); //store logs for now
    }
  }
  else if (idx == 1) {

    Math3D::Tensor<double> temp(nLabels1,nLabels3,nLabels4);

    for (uint l2 = 0; l2 < nLabels2; l2++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            temp(l1,l3,l4) = param[0][l1] + param[2][l3] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l3 = 0; l3 < nLabels3; l3++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            //sum += std::exp(temp(l1,l3,l4)-offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l3,l4) != offs) ? std::exp(temp(l1,l3,l4) - offs) : 1.0;
          }
        }
      }

      marginal[l2] = offs + idx_param[l2] + std::log(sum); //store logs for now
    }
  }
  else if (idx == 2) {

    Math3D::Tensor<double> temp(nLabels1,nLabels2,nLabels4);

    for (uint l3 = 0; l3 < nLabels3; l3++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            temp(l1,l2,l4) = param[0][l1] + param[1][l2] + param[3][l4] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l4 = 0; l4 < nLabels4; l4++) {

            //sum += std::exp(temp(l1,l2,l4) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l2,l4) != offs) ? std::exp(temp(l1,l2,l4) - offs) : 1.0;
          }
        }
      }

      marginal[l3] = offs + idx_param[l3] + std::log(sum); //store logs for now
    }
  }
  else {
    assert(idx == 3);

    Math3D::Tensor<double> temp(nLabels1,nLabels2,nLabels3);

    for (uint l4 = 0; l4 < nLabels4; l4++) {

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        const Math3D::Tensor<float>& cur_cost = cost[l1];

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l3 = 0; l3 < nLabels3; l3++) {

            temp(l1,l2,l3) = param[0][l1] + param[1][l2] + param[2][l3] - (cur_cost(l2,l3,l4) / mu);
          }
        }
      }

      const double offs = temp.max();

      double sum = 0.0;

      for (uint l1 = 0; l1 < nLabels1; l1++) {

        for (uint l2 = 0; l2 < nLabels2; l2++) {

          for (uint l3 = 0; l3 < nLabels3; l3++) {

            //sum += std::exp(temp(l1,l2,l3) - offs);

            //exp is expensive -> better use an if
            sum += (temp(l1,l2,l3) != offs) ? std::exp(temp(l1,l2,l3) - offs) : 1.0;
          }
        }
      }

      marginal[l4] = offs + idx_param[l4] + std::log(sum); //store logs for now
    }
  }

  //now convert to exponential format
  const double offs = marginal.max();
  for (uint l=0; l < marginal.size(); l++) {
    //marginal[l] = std::exp(marginal[l]-offs);

    marginal[l] = (marginal[l] != offs) ? std::exp(marginal[l]-offs) : 1.0;
  }


  marginal *= 1.0 / marginal.sum(); //renormalize to get the actual marginals
}


/*****/

FourthOrderChainDDFactor::FourthOrderChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const Storage1D<Math3D::Tensor<float> >& cost)
  : FourthOrderChainDDFactorBase(involved_vars), cost_(cost)
{
  if (cost_.size() < involved_vars[0]->nLabels() || cost_[0].xDim() < involved_vars[1]->nLabels()
      || cost_[0].yDim() < involved_vars[2]->nLabels() || cost_[0].zDim() < involved_vars[3]->nLabels()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
  }
}

/*virtual*/ FourthOrderChainDDFactor::~FourthOrderChainDDFactor() {}

/*virtual*/
double FourthOrderChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_[labeling[0]](labeling[1],labeling[2],labeling[3]);
}

/*virtual*/ double FourthOrderChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  return FourthOrderChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/*virtual*/
double FourthOrderChainDDFactor::compute_sum_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return FourthOrderChainDDFactorBase::compute_sum_forward(incoming,outgoing,prev_forward,forward,mu,cost_);
}

/*virtual*/
double FourthOrderChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return FourthOrderChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}

/*virtual*/
void FourthOrderChainDDFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2,
    double mu, Math1D::Vector<double>& marginals) const noexcept
{
  FourthOrderChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void FourthOrderChainDDFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  FourthOrderChainDDFactorBase::compute_marginals_logspace(target,in_var1,in_var2,log_forward1,log_forward2,mu,cost_,marginal);
}

/***/

FourthOrderChainDDRefFactor::FourthOrderChainDDRefFactor(const Storage1D<ChainDDVar*>& involved_vars, const Storage1D<Math3D::Tensor<float> >& cost)
  : FourthOrderChainDDFactorBase(involved_vars), cost_(cost)
{
}

/*virtual*/ FourthOrderChainDDRefFactor::~FourthOrderChainDDRefFactor() {}

/*virtual*/
double FourthOrderChainDDRefFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_[labeling[0]](labeling[1],labeling[2],labeling[3]);
}

/*virtual*/ double FourthOrderChainDDRefFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  assert(cost_.size() >= involved_var_[0]->nLabels());
  assert(cost_[0].xDim() >= involved_var_[1]->nLabels());
  assert(cost_[0].yDim() >= involved_var_[2]->nLabels());
  assert(cost_[0].zDim() >= involved_var_[3]->nLabels());

  return FourthOrderChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/*virtual*/
double FourthOrderChainDDRefFactor::compute_sum_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, double mu) const noexcept
{
  return FourthOrderChainDDFactorBase::compute_sum_forward(incoming,outgoing,prev_forward,forward,mu,cost_);
}

/*virtual*/
double FourthOrderChainDDRefFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  return FourthOrderChainDDFactorBase::compute_sum_forward_logspace(in_var,out_var,prev_log_forward,log_forward,mu,cost_);
}


/*virtual*/
void FourthOrderChainDDRefFactor::compute_marginals(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& forward1, const Math1D::Vector<double>& forward2, double mu, Math1D::Vector<double>& marginals) const noexcept
{
  FourthOrderChainDDFactorBase::compute_marginals(target,in_var1,in_var2,forward1,forward2,mu,cost_,marginals);
}

/*virtual*/
void FourthOrderChainDDRefFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  FourthOrderChainDDFactorBase::compute_marginals_logspace(target,in_var1,in_var2,log_forward1,log_forward2,mu,cost_,marginal);
}

/****/

GeneralizedPottsChainDDFactor::GeneralizedPottsChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, float lambda)
  : ChainDDFactor(involved_vars), lambda_(lambda)
{
  uint nLabels = involved_vars[0]->nLabels();
  for (uint k=1; k < involved_vars.size(); k++) {

    if (involved_vars[k]->nLabels() != nLabels) {
      INTERNAL_ERROR << "variables for GenPotts must all have the same number of labels. Exiting." << std::endl;
      exit(1);
    }
  }
}

/*virtual*/
double GeneralizedPottsChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  uint nVars = involved_var_.size();

  uint nLabels = involved_var_[0]->nLabels();

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());

  Math1D::Vector<uint> best_label(nVars);
  double best_sum = lambda_;
  for (uint k=0; k < nVars; k++) {

    if (k != idx) {
      double cur_best = 1e30;
      for (uint l=0; l < nLabels; l++) {

        double hyp = - param[k][l];
        if (hyp < cur_best) {
          cur_best = hyp;
          best_label[k] = l;
        }
      }
      best_sum += cur_best;
    }
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  for (uint l=0; l < nLabels; l++) {

    double hyp = 0.0;
    for (uint k=0; k < nVars; k++) {

      if (k != idx)
        hyp -= param[k][l];
    }

    if (hyp < best_sum) {
      forward[l] = hyp - idx_param[l];
      for (uint k=0; k < nVars; k++)
        trace(k,l) = l;
    }
    else {
      forward[l] = best_sum - idx_param[l];
      for (uint k=0; k < nVars; k++)
        trace(k,l) = best_label[k];
    }

    trace(idx,l) = l;
  }

  return 0.0; //presently not subtracting an offset
}

/*virtual*/
double GeneralizedPottsChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  uint l1 = labeling[0];

  for (uint k=1; k < labeling.size(); k++) {
    if (labeling[k] != l1)
      return lambda_;
  }

  return 0.0;
}

/****/

OneOfNChainDDFactor::OneOfNChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars) :
  ChainDDFactor(involved_vars)
{
  for (uint v=0; v < involved_vars.size(); v++) {
    if (involved_vars[v]->nLabels() != 2) {
      INTERNAL_ERROR << "instantiation of a 1-of-N factor with non-binary variables. Exiting." << std::endl;
      exit(1);
    }
  }
}

/*virtual*/ OneOfNChainDDFactor::~OneOfNChainDDFactor() {}

/*virtual*/
double OneOfNChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  if (labeling.sum() == 1)
    return 0.0;
  return 1e30;
}

uint OneOfNChainDDFactor::best_of_n() const noexcept
{
  uint nVars = involved_var_.size();

  double best = 1e300;
  uint arg_best = MAX_UINT;

  for (uint k=0; k < nVars; k++) {

    const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

    const double hyp = cur_cost[1] - dual_var_[k][1]
                       - cur_cost[0] + dual_var_[k][0];

    if (hyp < best) {

      best = hyp;
      arg_best = k;
    }
  }

  return arg_best;
}

/*virtual*/
double OneOfNChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  uint nVars = involved_var_.size();

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());
  trace.set_constant(0);

  double best_gain = 1e300;
  uint argmin = MAX_UINT;

  double sum = 0.0;

  for (uint i=0; i < nVars; i++) {

    if (involved_var_[i] == out_var) {
      forward[0] = -param[i][0];
      forward[1] = -param[i][1];
      trace(i,1) = 1;
    }
    else {
      double hyp = -param[i][1] + param[i][0];

      if (hyp < best_gain) {
        best_gain = hyp;
        argmin = i;
      }

      sum -= param[i][0];
    }
  }

  trace(argmin,0) = 1;
  forward[0] += sum + param[argmin][0] - param[argmin][1];
  forward[1] += sum;

  return 0.0; //presently not subtracting an offset
}

/**********************/

CardinalityChainDDFactorBase::CardinalityChainDDFactorBase(const Storage1D<ChainDDVar*>& involved_vars)
  : ChainDDFactor(involved_vars)
{
  for (uint v=0; v < involved_vars.size(); v++) {
    if (involved_vars[v]->nLabels() != 2) {
      INTERNAL_ERROR << "instantiation of a cardinality factor with non-binary variables. Exiting." << std::endl;
      exit(1);
    }
  }
}

double CardinalityChainDDFactorBase::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,
    Math2D::Matrix<uint>& trace, const Math1D::Vector<float>& cost) const noexcept
{
  uint nVars = involved_var_.size();

  assert(nVars >= 2);

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());
  trace.set_constant(0);
  trace(idx,1) = 1;

  Storage1D<std::pair<double,uint> > rel_msg(nVars-1);

  double offs = 0.0;

  uint next = 0;
  for (uint k=0; k < nVars; k++) {

    if (k != idx) {
      const Math1D::Vector<double>& cur_param = param[k];

      const double val0 = - cur_param[0];
      const double val1 = - cur_param[1];

      rel_msg[next] = std::make_pair(val1 - val0,k);
      offs += val0;

      next++;
    }
  }

  std::sort(rel_msg.direct_access(), rel_msg.direct_access() + nVars-1);

  forward.set_constant(1e300);

  int best_c0 = 0;
  int best_c1 = 0;

  double cum_sum = 0.0;

  for (uint c=0; c < nVars; c++) {

    double hyp0 = cum_sum + cost[c];
    if (hyp0 < forward[0]) {
      forward[0] = hyp0;
      best_c0 = c;
    }

    double hyp1 = cum_sum + cost[c+1];
    if (hyp1 < forward[1]) {
      forward[1] = hyp1;
      best_c1 = c;
    }

    if (c+1 < nVars)
      cum_sum += rel_msg[c].first;
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  for (uint l=0; l < 2; l++) {
    forward[l] += offs - idx_param[l];
  }

  for (int c = 0; c < best_c0; c++)
    trace(rel_msg[c].second, 0) = 1;

  for (int c = 0; c < best_c1; c++)
    trace(rel_msg[c].second, 1) = 1;

  return 0.0; //presently not subtracting an offset
}

/**********************/


CardinalityChainDDFactor::CardinalityChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math1D::Vector<float>& cost) :
  CardinalityChainDDFactorBase(involved_vars), cost_(cost)
{
  if (cost_.size() < involved_vars.size()+1) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
    exit(1);
  }
}

/*virtual*/ CardinalityChainDDFactor::~CardinalityChainDDFactor() {}

/*virtual*/
double CardinalityChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_[labeling.sum()];
}

/*virtual*/
double CardinalityChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  return CardinalityChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/*****************/

CardinalityChainDDRefFactor::CardinalityChainDDRefFactor(const Storage1D<ChainDDVar*>& involved_vars, const Math1D::Vector<float>& cost) :
  CardinalityChainDDFactorBase(involved_vars), cost_(cost)
{
}

/*virtual*/ CardinalityChainDDRefFactor::~CardinalityChainDDRefFactor() {}

/*virtual*/
double CardinalityChainDDRefFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  return cost_[labeling.sum()];
}

/*virtual*/
double CardinalityChainDDRefFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  assert(cost_.size() >= involved_var_.size()+1);

  return CardinalityChainDDFactorBase::compute_forward(in_var,out_var,prev_forward,forward,trace,cost_);
}

/********************/

NonbinaryCardinalityChainDDFactorBase::NonbinaryCardinalityChainDDFactorBase(const Storage1D<ChainDDVar*>& vars)
  : ChainDDFactor(vars) {}

double NonbinaryCardinalityChainDDFactorBase::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward,  Math2D::Matrix<uint>& trace,
    const Math1D::Vector<float>& cost, const Math1D::Vector<uint>& level) const noexcept
{
  uint nVars = involved_var_.size();

  assert(nVars >= 2);

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  Storage1D<Math1D::Vector<double> > bin_param(param.size());
  Math1D::Vector<uint> best_label(param.size(),MAX_UINT);

  for (uint k=0; k < nVars; k++) {
    bin_param[k].resize(2,1e300);

    for (uint l=0; l < param[k].size(); l++) {

      const double val = -param[k][l];

      if (l == level[k])
        bin_param[k][1] = val;
      else if (val < bin_param[k][0]) {
        bin_param[k][0] = val;
        best_label[k] = l;
      }
    }
  }

  Storage1D<std::pair<double,uint> > rel_msg(nVars-1);

  double offs = 0.0;

  uint next = 0;
  for (uint k=0; k < nVars; k++) {

    if (k != idx) {
      const Math1D::Vector<double>& cur_bin_param = bin_param[k];

      const double val0 = cur_bin_param[0];
      const double val1 = cur_bin_param[1];

      rel_msg[next] = std::make_pair(val1 - val0,k);
      offs += val0;

      next++;
    }
  }

  std::sort(rel_msg.direct_access(), rel_msg.direct_access() + nVars-1);

  Math1D::Vector<double> bin_forward(2,1e300);
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());

  for (uint v=0; v < nVars; v++) {
    for (uint l=0; l < trace.yDim(); l++)
      trace(v,l) = best_label[v];
  }

  trace(idx,level[idx]) = level[idx];

  int best_c0 = 0;
  int best_c1 = 0;

  double cum_sum = 0.0;

  for (uint c=0; c < nVars; c++) {

    double hyp0 = cum_sum + cost[c];
    if (hyp0 < bin_forward[0]) {
      bin_forward[0] = hyp0;
      best_c0 = c;
    }

    double hyp1 = cum_sum + cost[c+1];
    if (hyp1 < bin_forward[1]) {
      bin_forward[1] = hyp1;
      best_c1 = c;
    }

    if (c+1 < nVars)
      cum_sum += rel_msg[c].first;
  }

  for (uint l=0; l < 2; l++) {
    bin_forward[l] += offs;
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  forward.set_constant(bin_forward[0]);
  forward[level[idx]] = bin_forward[1];
  forward -= param[idx];

  for (int c = 0; c < best_c0; c++) {
    for (uint l=0; l < trace.yDim(); l++) {
      if (l != level[idx])
        trace(rel_msg[c].second,l) = level[rel_msg[c].second];
    }
  }

  for (int c = 0; c < best_c1; c++)
    trace(rel_msg[c].second,level[idx]) = level[rel_msg[c].second];

  return 0.0; //presently not subtracting an offset
}

/********************/

NonbinaryCardinalityChainDDFactor::NonbinaryCardinalityChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars,
    const Math1D::Vector<float>& cost, const Math1D::Vector<uint>& level)
  : NonbinaryCardinalityChainDDFactorBase(involved_vars), cost_(cost), level_(level) {}

/*virtual*/
double NonbinaryCardinalityChainDDFactor::compute_forward(const ChainDDVar* incoming, const ChainDDVar* outgoing,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  return NonbinaryCardinalityChainDDFactorBase::compute_forward(incoming,outgoing,prev_forward,forward, trace,cost_,level_);
}

/*virtual*/ double NonbinaryCardinalityChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  assert(labeling.size() == involved_var_.size());

  uint card = 0;
  for (uint k=0; k < involved_var_.size(); k++)  {
    if (labeling[k] == level_[k])
      card++;
  }

  return cost_[card];
}

/********************/

AllPosBILPChainDDFactor::AllPosBILPChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars,
    int rhs_lower, int rhs_upper)
  : ChainDDFactor(involved_vars), rhs_lower_(std::max(0,rhs_lower)), rhs_upper_(std::min<int>(involved_vars.size(),rhs_upper))
{
  if (rhs_lower_ > rhs_upper_ || rhs_upper_ < 0) {
    INTERNAL_ERROR << "constraint is unsatisfiable, so inference is pointless. Exiting." << std::endl;
    exit(1);
  }

  for (uint v=0; v < involved_vars.size(); v++) {
    if (involved_vars[v]->nLabels() != 2) {
      INTERNAL_ERROR << "instantiation of an AllPosBILP factor with non-binary variables. Exiting." << std::endl;
      exit(1);
    }
  }
}

/*virtual*/ AllPosBILPChainDDFactor::~AllPosBILPChainDDFactor() {}

/*virtual*/
double AllPosBILPChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward, Math2D::Matrix<uint>& trace) const noexcept
{
  uint nVars = involved_var_.size();

  assert(nVars >= 2);

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());
  trace.set_constant(0);
  trace(idx,1) = 1;

  Storage1D<std::pair<double,uint> > rel_msg(nVars-1);

  double offs = 0.0;

  uint next = 0;
  for (uint k=0; k < nVars; k++) {

    if (k != idx) {
      const Math1D::Vector<double>& cur_param = param[k];

      const double val0 = - cur_param[0];
      const double val1 = - cur_param[1];

      rel_msg[next] = std::make_pair(val1 - val0,k);
      offs += -cur_param[0];

      next++;
    }
  }

  std::sort(rel_msg.direct_access(), rel_msg.direct_access() + nVars-1);

  forward.set_constant(1e300);

  int best_c0 = 0;
  int best_c1 = 0;

  double cum_sum = 0.0;

  for (int c=0; c < int(nVars); c++) {

    if (c >= rhs_lower_ && c <= rhs_upper_) {
      double hyp0 = cum_sum;
      if (hyp0 < forward[0]) {
        forward[0] = hyp0;
        best_c0 = c;
      }
    }

    if (c+1 >= rhs_lower_ && c+1 <= rhs_upper_) {
      double hyp1 = cum_sum;
      if (hyp1 < forward[1]) {
        forward[1] = hyp1;
        best_c1 = c;
      }
    }

    if (c+1 < int(nVars))
      cum_sum += rel_msg[c].first;
  }

  const Math1D::Vector<double>& idx_param = param[idx];

  for (uint l=0; l < 2; l++) {
    forward[l] += offs - idx_param[l];
  }

  for (int c = 0; c < best_c0; c++)
    trace(rel_msg[c].second,0) = 1;

  for (int c = 0; c < best_c1; c++)
    trace(rel_msg[c].second,1) = 1;

  return 0.0; //presently not subtracting an offset
}

/*virtual*/
double AllPosBILPChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  int sum = labeling.sum();
  return (sum >= rhs_lower_ && sum <= rhs_upper_) ? 0.0 : 1e20;
}

/*virtual*/
double AllPosBILPChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  double log_offs = 0.0;

  const uint nVars = involved_var_.size();

  assert(out_var != in_var);

  Storage1D<Math1D::Vector<double> > param = dual_var_;

  uint idx = MAX_UINT;

  for (uint k=0; k < nVars; k++) {
    Math1D::Vector<double>& cur_param = param[k];

    Math1D::Vector<double> temp(cur_param.size());

    if (involved_var_[k] == in_var) {

      cur_param *= 1.0 / mu;

      cur_param += prev_log_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++)
        cur_param[l] -= cur_cost[l]; //negative cost since we maximize
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  //std::cerr << "param: " << param << std::endl;
  //std::cerr << "idx: " << idx << std::endl;

  log_forward.resize_dirty(2);

  assert(rhs_upper_ >= 1);

  Math1D::Vector<double> log_forward_vec[2];
  log_forward_vec[0].resize(rhs_upper_+1,-1e100);
  log_forward_vec[1].resize(rhs_upper_+1,-1e100); //init is needed because the loops below do not visit all entries (to reduce run-time)

  const uint start_idx = (idx != 0) ? 0 : 1;

  uint cur_idx = 0;
  Math1D::Vector<double>& start_log_forward_vec = log_forward_vec[0];

  start_log_forward_vec[0] = param[start_idx][0];
  start_log_forward_vec[1] = param[start_idx][1];

  Math1D::Vector<double> temp(2);

  //proceed for positive vars
  uint k=0;
  for (uint v= start_idx + 1; v < nVars; v++) {

    if (v != idx) {

      k++;

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward_vec = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];

      for (int sum=0; sum < std::min<int>(rhs_upper_+1,k+2); sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum - l;
          if (dest >= 0) {
            temp[l] = prev_log_forward_vec[dest] + cur_param[l];
          }
          else {
            temp[l] = -1e100;
          }
        }

        if (temp[0] >= temp[1]) {
          cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        }
        else {
          cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        }

      }
    }
  }

  //now final computation for idx
  const Math1D::Vector<double>& last_log_forward_vec = log_forward_vec[cur_idx];

  const Math1D::Vector<double>& idx_param = param[idx];

  const int lower_limit = int(rhs_lower_);
  const int upper_limit = int(rhs_upper_);

  Math1D::Vector<double> cur_temp(1+upper_limit-lower_limit);

  for (uint l=0; l < 2; l++) {

    //std::cerr << "l: " << l << std::endl;

    for (int s=lower_limit; s <= upper_limit; s++) {

      const int dest = s - l;
      if (dest >= 0 && dest <= upper_limit) {

        cur_temp[s-lower_limit] = last_log_forward_vec[dest];
      }
      else
        cur_temp[s-lower_limit] = -1e100;
    }

    double offs = cur_temp.max();
    double sum = 0.0;

    for (uint k=0; k < cur_temp.size(); k++) {
      sum += std::exp(cur_temp[k] - offs);
    }

    assert(sum > 0.0); //should be the case when the variable can have this label

    log_forward[l] = offs + std::log(sum) + idx_param[l];
  }


  return log_offs;
}

/*virtual*/
void AllPosBILPChainDDFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2, double mu, Math1D::Vector<double>& marginal) const noexcept
{
  const uint nVars = involved_var_.size();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  marginal.resize_dirty(2);

  Math1D::Vector<double> log_forward_vec[2];
  log_forward_vec[0].resize(rhs_upper_+1,-1e100);
  log_forward_vec[1].resize(rhs_upper_+1,-1e100); //init is needed because the loops below do not visit all entries (to reduce run-time)

  const uint start_idx = (idx != 0) ? 0 : 1;

  uint cur_idx = 0;
  Math1D::Vector<double>& start_log_forward_vec = log_forward_vec[0];

  assert(rhs_upper_ >= 1);

  start_log_forward_vec[0] = param[start_idx][0];
  start_log_forward_vec[1] = param[start_idx][1];

  Math1D::Vector<double> temp(2);

  //proceed for positive vars
  for (uint v= start_idx + 1; v < nVars; v++) {

    if (v != idx) {

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward_vec = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];

      for (int sum=0; sum < std::min<int>(rhs_upper_+1,v+2); sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum - l;
          if (dest >= 0) {
            temp[l] = prev_log_forward_vec[dest] + cur_param[l];
          }
          else {
            temp[l] = -1e100;
          }
        }

        if (temp[0] >= temp[1]) {
          cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        }
        else {
          cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        }

      }
    }
  }

  //now final computation for idx
  const Math1D::Vector<double>& last_log_forward_vec = log_forward_vec[cur_idx];

  const Math1D::Vector<double>& idx_param = param[idx];

  const int lower_limit = int(rhs_lower_);
  const int upper_limit = int(rhs_upper_);

  Math1D::Vector<double> cur_temp(1+upper_limit-lower_limit);

  for (uint l=0; l < 2; l++) {

    //std::cerr << "l: " << l << std::endl;

    for (int s=lower_limit; s <= upper_limit; s++) {

      const int dest = s - l;
      if (dest >= 0 && dest <= upper_limit) {

        cur_temp[s-lower_limit] = last_log_forward_vec[dest];
      }
      else
        cur_temp[s-lower_limit] = -1e100;
    }

    double offs = cur_temp.max();
    double sum = 0.0;

    for (uint k=0; k < cur_temp.size(); k++) {
      sum += std::exp(cur_temp[k] - offs);
    }

    assert(sum > 0.0); //should be the case when the variable can have this label

    marginal[l] = offs + std::log(sum) + idx_param[l];
  }


  //now convert to exponential format
  const double offs = marginal.max();
  for (uint l=0; l < marginal.size(); l++) {
    //marginal[l] = std::exp(marginal[l]-offs);

    marginal[l] = (marginal[l] != offs) ? std::exp(marginal[l]-offs) : 1.0;
  }

  marginal *= 1.0 / marginal.sum(); //renormalize to get the actual marginals
}

/*virtual*/
void AllPosBILPChainDDFactor::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  const uint nVars = involved_var_.size();
  assert(nVars >= 2);

  marginal.resize(nVars);

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  Math1D::Vector<double> temp(2);

  //a) forward

  Math2D::Matrix<double> forward(rhs_upper_+1,nVars-1,-1e100);

  forward(0,0) = param[0][0];
  forward(1,0) = param[0][1];

  for (uint v=1; v < nVars-1; v++) {

    const Math1D::Vector<double>& cur_param = param[v];

    const int limit = std::min<int>(rhs_upper_+1,v+2);

    for (int sum=0; sum < limit; sum++) {

      for (int l=0; l < 2; l++) {

        const int dest = sum - l;
        if (dest >= 0) {
          temp[l] = forward(dest,v-1) + cur_param[l];
        }
        else {
          temp[l] = -1e100;
        }
      }

      if (temp[0] >= temp[1]) {
        forward(sum,v) = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
      }
      else {
        forward(sum,v) = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
      }
    }
  }

  //b) backward
  Math2D::Matrix<double> backward(rhs_upper_+1,nVars,-1e100);

  backward(0,nVars-1) = param[nVars-1][0];
  backward(1,nVars-1) = param[nVars-1][1];

  for (int v=nVars-2; v >= 1; v--) {

    const Math1D::Vector<double>& cur_param = param[v];

    const int limit = std::min<int>(rhs_upper_+1,nVars-v+1);

    //for (int sum=0; sum <= rhs_upper_; sum++) {
    for (int sum=0; sum < limit; sum++) {

      for (int l=0; l < 2; l++) {

        const int dest = sum - l;
        if (dest >= 0) {
          temp[l] = backward(dest,v+1) + cur_param[l];
        }
        else {
          temp[l] = -1e100;
        }
      }

      if (temp[0] >= temp[1]) {
        backward(sum,v) = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
      }
      else {
        backward(sum,v) = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
      }
    }
  }

  //std::cerr << "c: " << rhs_lower_ << " <= c x <= " << rhs_upper_ << ", nVars: " << nVars << std::endl;

  //c) finally, compute the marginals
  for (uint v=0; v < nVars; v++) {

    //std::cerr << "v: " << v << std::endl;

    const Math1D::Vector<double>& cur_param = param[v];

    Math1D::Vector<double>& cur_marginal = marginal[v];

    cur_marginal.resize_dirty(2);

    for (int l=0; l < 2; l++) {

      std::vector<double> cur_temp;

      //std::cerr << "l: " << l << std::endl;

      if (v == 0) {

        for (int sum=rhs_lower_; sum <= rhs_upper_; sum++) {

          //std::cerr << "sum: " << sum << std::endl;

          if (sum-l >= 0) {
            //std::cerr << "push_back of " << backward(sum-l,1) << std::endl;
            cur_temp.push_back(backward(sum-l,1));
          }
        }
      }
      else if (v == nVars-1) {

        for (int sum=rhs_lower_; sum <= rhs_upper_; sum++) {

          if (sum-l >= 0) {
            cur_temp.push_back(forward(sum-l,nVars-2));
          }
        }
      }
      else {

        for (int sum=rhs_lower_; sum <= rhs_upper_; sum++) {

          for (int s_prev=0; s_prev <= sum-l; s_prev++) {

            double cand = forward(s_prev,v-1) + backward(sum-l-s_prev,v+1);

            if (cand > -1e98) {
              cur_temp.push_back(cand);
            }
          }
        }
      }

      //std::cerr << "common" << std::endl;

      if (cur_temp.size() == 0)
        cur_marginal[l] = -1e100;
      else {

        const double offs = vec_max(cur_temp);
        double sum = 0.0;
        for (uint k=0; k < cur_temp.size(); k++) {
          sum += (cur_temp[k] != offs) ? std::exp(cur_temp[k] - offs) : 1.0;
        }
        cur_marginal[l] = offs + std::log(sum) + cur_param[l]; //store logs for now
      }
    }

    //now convert to exponential format
    const double offs = cur_marginal.max();
    for (uint l=0; l < cur_marginal.size(); l++) {
      //cur_marginal[l] = std::exp(cur_marginal[l]-offs);

      cur_marginal[l] = (cur_marginal[l] != offs) ? std::exp(cur_marginal[l]-offs) : 1.0;
    }


    cur_marginal *= 1.0 / cur_marginal.sum(); //renormalize to get the actual marginals
  }
}

/*****************/

BILPChainDDFactor::BILPChainDDFactor(const Storage1D<ChainDDVar*>& involved_vars, const Storage1D<bool>& positive,
                                     int rhs_lower, int rhs_upper) :
  ChainDDFactor(involved_vars), rhs_lower_(rhs_lower), rhs_upper_(rhs_upper)
{
  for (uint v=0; v < involved_vars.size(); v++) {
    if (involved_vars[v]->nLabels() != 2) {
      INTERNAL_ERROR << "instantiation of an BILP factor with non-binary variables. Exiting." << std::endl;
      exit(1);
    }
  }

  if (positive.size() < involved_vars.size()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
    exit(1);
  }

  if (rhs_lower_ > rhs_upper_) {
    INTERNAL_ERROR << "constraint is unsatisfiable, so inference is pointless. Exiting." << std::endl;
    exit(1);
  }


  Storage1D<ChainDDVar*> sorted_involved_vars(involved_vars.size());
  uint next_pos = 0;

  //pass 1 - find all positive
  for (uint v=0; v < involved_vars.size(); v++) {
    if (positive[v]) {
      sorted_involved_vars[next_pos] = involved_vars[v];
      next_pos++;
    }
  }

  nPos_ = next_pos;

  //pass 2 - find all negative
  for (uint v=0; v < involved_vars.size(); v++) {
    if (!positive[v]) {
      sorted_involved_vars[next_pos] = involved_vars[v];
      next_pos++;
    }
  }

  involved_var_ = sorted_involved_vars;

  int nPositive = nPos_;
  int nNegative = involved_var_.size()-nPositive;

  int lower_bound = -nNegative;

  //lower_bound = std::max(lower_bound, rhs_lower_ - nPositive);
  /*** since we process the positive vars first, we need not compute entries below rhs_lower_-1 ***/
  /*** the offset of -1 is because we always leave one variable (the target of the message) out
       in the forward computation, and this variable can have a positive sign ***/
  lower_bound = std::max(lower_bound, rhs_lower_ - 1);

  int upper_bound = nPositive;
  upper_bound = std::min(upper_bound, rhs_upper_ + nNegative);

  const int range = upper_bound - lower_bound + 1;
  const int zero_offset = -lower_bound;

  if (rhs_upper_ + zero_offset < 0 || rhs_lower + zero_offset >= range) {
    INTERNAL_ERROR << "constraint is unsatisfiable. Exiting..." << std::endl;
    exit(1);
  }

  //adjust the bounds to the actually possible range of values
  if (rhs_lower_ + zero_offset < 0) {
    rhs_lower_ -= (rhs_lower_ + zero_offset);
  }
  if (rhs_upper_ + zero_offset >= range) {
    rhs_upper_ -= (rhs_upper_ + zero_offset - range +1);
  }

  range_ = range;
  zero_offset_ = zero_offset;
}

/*virtual*/ BILPChainDDFactor::~BILPChainDDFactor() {}

/*virtual*/
double BILPChainDDFactor::cost(const Math1D::Vector<uint>& labeling) const noexcept
{
  int sum = 0;
  const uint nVars = involved_var_.size();

  for (uint k=0; k < nPos_; k++)
    sum += labeling[k];
  for (uint k=nPos_; k < nVars; k++)
    sum -= labeling[k];

  return (sum >= rhs_lower_ && sum <= rhs_upper_) ? 0.0 : 1e30;
}

/*virtual*/
double BILPChainDDFactor::compute_forward(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_forward, Math1D::Vector<double>& forward_msg, Math2D::Matrix<uint>& trace) const noexcept
{
  //based on [Potetz & Lee CVIU 2007]

  //std::cerr << "forward comp" << std::endl;

  const uint nVars = involved_var_.size();

  assert(out_var != in_var);

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var) {

      cur_param -= prev_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
    }
  }

  forward_msg.resize_dirty(involved_var_[idx]->nLabels());
  trace.resize_dirty(nVars,involved_var_[idx]->nLabels());

  if (false) {

    //DONT PUBLISH THIS!!!!!
    //NOTE: once we permanently switch to this version, we can get rid of the members zero_offset_ and range_

    double msg_offs = 0.0;

    int nPos = nPos_;
    int nNeg = nVars-nPos;

    if (idx < nPos_)
      nPos--;
    else
      nNeg--;


    // FlexibleStorage1D<std::pair<double,uint> > pos(nPos);
    // FlexibleStorage1D<std::pair<double,uint> > neg(nNeg);

    // for (uint v=0; v < nVars; v++) {
    //   if (v != idx) {
    //     if (positive_[v]) {
    //       pos.append(std::make_pair(-param[v][1]+param[v][0],v));
    //     }
    //     else {
    //       neg.append(std::make_pair(-param[v][1]+param[v][0],v));
    //     }
    //     msg_offs -= param[v][0];
    //   }
    // }

    // std::sort(pos.direct_access(),pos.direct_access()+nPos);
    // std::sort(neg.direct_access(),neg.direct_access()+nNeg);

    std::vector<std::pair<double,uint> > pos;
    pos.reserve(nPos);
    std::vector<std::pair<double,uint> > neg;
    neg.reserve(nNeg);

    for (uint v=0; v < nPos_; v++) {

      if (v != idx) {
        pos.push_back(std::make_pair(-param[v][1]+param[v][0],v));
        msg_offs -= param[v][0];
      }
    }
    for (uint v=nPos_; v < nVars; v++) {

      if (v != idx) {
        neg.push_back(std::make_pair(-param[v][1]+param[v][0],v));
        msg_offs -= param[v][0];
      }
    }

    std::sort(pos.begin(),pos.end());
    std::sort(neg.begin(),neg.end());

    for (uint k=1; k < pos.size(); k++)
      pos[k].first += pos[k-1].first;
    for (uint k=1; k < neg.size(); k++)
      neg[k].first += neg[k-1].first;

    int nBestPos0 = 0;
    int nBestPos1 = 0;
    int nBestNeg0 = 0;
    int nBestNeg1 = 0;


    forward_msg.set_constant(1e20);
    if (0 >= rhs_lower_ && 0 <= rhs_upper_) {
      forward_msg[0] = 0.0;
      nBestPos0 = -1;
      nBestNeg0 = -1;
    }

    const int cmp = (idx < nPos_) ? 1 : -1;
    if (cmp >= rhs_lower_ && cmp <= rhs_upper_) {
      forward_msg[1] = 0.0;
      nBestPos1 = -1;
      nBestNeg1 = -1;
    }

    for (int rhs = rhs_lower_; rhs <= rhs_upper_; rhs++) {

      //std::cerr << "rhs: " << rhs << std::endl;

      /**** a) compute message[0] ****/

      //now handled in the message init:
      // if (rhs == 0) { //neither pos nor neg
      //   if (forward_msg[0] > 0.0) {
      //     forward_msg[0] = 0.0;
      //     nBestPos0 = -1;
      //     nBestNeg0 = -1;
      //   }
      // }
      if ((-rhs) > 0 && (-rhs) <= nNeg ) { //only neg
        double hyp = neg[-rhs-1].first;
        if (hyp < forward_msg[0]) {
          forward_msg[0] = hyp;
          nBestPos0 = -1;
          nBestNeg0 = -rhs-1;
        }
      }
      if (rhs > 0 && rhs <= nPos ) { //only pos
        double hyp = pos[rhs-1].first;
        if (hyp < forward_msg[0]) {
          forward_msg[0] = hyp;
          nBestPos0 = rhs-1;
          nBestNeg0 = -1;
        }
      }
      //both pos and neg
      for (int k=std::max(0,rhs); k < std::min<int>(nPos,nNeg+rhs); k++) {
        double hyp = pos[k].first + neg[k-rhs].first;
        if (hyp < forward_msg[0]) {
          forward_msg[0] = hyp;
          nBestPos0 = k;
          nBestNeg0 = k-rhs;
        }
      }


      /**** b) compute message[1] ****/

      //std::cerr << "b), positive: " << positive_[idx] << std::endl;

      if (idx < nPos_) {

        //now handled in the message init:
        // if (rhs == 1) { //neither pos nor neg
        //   if (forward_msg[1] > 0.0) {
        //     forward_msg[1] = 0.0;
        //     nBestPos1 = -1;
        //     nBestNeg1 = -1;
        //   }
        // }

        if (1-rhs > 0 && nNeg >= (1-rhs)) { //only neg
          double hyp = neg[-rhs].first; //remember: offsets start at 0
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = -1;
            nBestNeg1 = -rhs;
          }
        }
        if (rhs-1 > 0 && (rhs-1) <= nPos) { //only pos
          double hyp = pos[rhs-2].first;
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = rhs-2;
            nBestNeg1 = -1;
          }
        }

        //both pos and neg
        for (int k=std::max(0,rhs-1); k < std::min(nPos,nNeg-1+rhs); k++) {
          double hyp = pos[k].first + neg[k+1-rhs].first;
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = k;
            nBestNeg1 = k+1-rhs;
          }
        }
      }
      else {

        //now handled in the message init:
        // if (rhs == -1) { //neither pos nor neg
        //   if (forward_msg[1] > 0.0) {
        //     forward_msg[1] = 0.0;
        //     nBestPos1 = -1;
        //     nBestNeg1 = -1;
        //   }
        // }
        if (rhs+1 > 0 && rhs < nPos) { //only pos
          double hyp = pos[rhs].first;
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = rhs;
            nBestNeg1 = -1;
          }
        }
        if ((-rhs-1) > 0 && (-rhs-2) < nNeg) { // only neg
          double hyp = neg[-rhs-2].first;
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = -1;
            nBestNeg1 = -rhs-2;
          }
        }

        //both pos and neg
        for (int k=std::max(0,-rhs-1); k < std::min(nPos-rhs-1,nNeg); k++) {
          double hyp = pos[rhs+k+1].first + neg[k].first;
          if (hyp < forward_msg[1]) {
            forward_msg[1] = hyp;
            nBestPos1 = rhs+k+1;
            nBestNeg1 = k;
          }
        }
      }
    }

    trace.set_constant(0);
    for (int k=0; k <= nBestPos0; k++)
      trace(pos[k].second,0) = 1;
    for (int k=0; k <= nBestNeg0; k++)
      trace(neg[k].second,0) = 1;
    for (int k=0; k <= nBestPos1; k++)
      trace(pos[k].second,1) = 1;
    for (int k=0; k <= nBestNeg1; k++)
      trace(neg[k].second,1) = 1;


    forward_msg[0] += msg_offs - param[idx][0];
    forward_msg[1] += msg_offs - param[idx][1];
  }
  else {

    //std::cerr << "zero offs: " << zero_offset_ << ", range: " << range_ << ", idx: " << idx << std::endl;

    Math2D::Matrix<uchar> forward_light_trace(range_,nVars-1/*,255*/);

    Math1D::Vector<double> forward_vec[2];
    forward_vec[0].resize(range_,1e100);
    forward_vec[1].resize(range_,1e100); //init is needed because the loops below do not visit all entries (to reduce run-time)

    const uint start_idx = (idx != 0) ? 0 : 1;

    uint cur_idx = 0;
    Math1D::Vector<double>& start_forward_vec = forward_vec[0];

    start_forward_vec[zero_offset_] = -param[start_idx][0];
    forward_light_trace(zero_offset_,0) = 0;

    const int init_mul = (start_idx < nPos_) ? 1 : -1;
    if (int(zero_offset_)+init_mul >= 0
        && int(zero_offset_)+init_mul < range_) {
      start_forward_vec[zero_offset_+init_mul] = -param[start_idx][1];
      forward_light_trace(zero_offset_+init_mul,0) = 1;
    }

    uint k=0;

    //proceed
    //a) positive vars
    for (uint v= start_idx + 1; v < nPos_; v++) {

      //std::cerr << "v: " << v << std::endl;

      if (v != idx) {

        const Math1D::Vector<double>& cur_param = param[v];

        k++;
#ifndef NDEBUG
        uint k_check = v;
        if (v > idx)
          k_check--;
        assert(k==k_check);
#endif

        const Math1D::Vector<double>& prev_forward_vec = forward_vec[cur_idx];

        cur_idx = 1 - cur_idx;

        Math1D::Vector<double>& cur_forward_vec = forward_vec[cur_idx];

        const int cur_limit = std::min<int>(range_,zero_offset_+k+2);

        //for (int sum=0; sum < range_; sum++) {
        for (int sum=zero_offset_; sum < cur_limit; sum++) {

          double best = 1e300;
          uint arg_best = MAX_UINT;

          for (int l=0; l < 2; l++) {

            const int dest = sum - l;
            if (dest >= 0) {

              double hyp = prev_forward_vec[dest] - cur_param[l];
              if (hyp < best) {
                best = hyp;
                arg_best = l;
              }
            }
          }
          cur_forward_vec[sum] = best;
          forward_light_trace(sum,k) = arg_best;
        }
      }
    }

    //b) negative vars
    const int first_neg = std::max<uint>((idx <= 1) ? 2 : 1,nPos_);
    for (int v=first_neg; v < int(nVars); v++) {


      if (v != idx) {

        const Math1D::Vector<double>& cur_param = param[v];

        k++;
#ifndef NDEBUG
        uint k_check = v;
        if (v > idx)
          k_check--;
        assert(k==k_check);
#endif

        const Math1D::Vector<double>& prev_forward = forward_vec[cur_idx];

        cur_idx = 1 - cur_idx;

        Math1D::Vector<double>& cur_forward = forward_vec[cur_idx];

        //for (int sum=0; sum < range_; sum++) {
        for (int sum=std::max(0,zero_offset_-(v-first_neg)-1); sum < range_; sum++) {

          double best = 1e300;
          uint arg_best = MAX_UINT;

          for (int l=0; l < 2; l++) {

            const int dest = sum + l;
            if (dest < range_) {

              double hyp = prev_forward[dest] - cur_param[l];
              if (hyp < best) {
                best = hyp;
                arg_best = l;
              }
            }
          }

          cur_forward[sum] = best;
          forward_light_trace(sum,k) = arg_best;
        }
      }
    }

    //std::cerr << "trace back" << std::endl;

    const Math1D::Vector<double>& last_forward_vec = forward_vec[cur_idx];

    const Math1D::Vector<double>& idx_param = param[idx];

    for (uint l=0; l < 2; l++) {

      //std::cerr << "l: " << l << std::endl;

      double min_msg = 1e300;
      uint best_s = MAX_UINT;

      for (int s=int(rhs_lower_ + zero_offset_); s <= int(rhs_upper_ + zero_offset_); s++) {

        int move = l;
        if (idx < nPos_)
          move = -move;

        const int dest = s + move;
        if (dest >= 0 && dest < range_) {

          double hyp = last_forward_vec[dest] - idx_param[l];

          assert(!isinf(hyp));

          if (hyp < min_msg) {
            min_msg = hyp;
            best_s = s;
          }
        }
      }

      forward_msg[l] = min_msg;
      trace(idx, l) = l;

      if (min_msg >= 1e50) //the variable cannot have this label, traceback might violate some bounds
        continue;

      //std::cerr << "best_s: " << best_s << ", msg: " << min_msg << std::endl;

      if (idx < nPos_)
        best_s -= l;
      else
        best_s += l;

      for (int k=nVars-2; k >= 0; k--) {

        //std::cerr << "k: " << k << std::endl;

        uint v=k;
        if (k >= int(idx))
          v++;

        uint cur_l = forward_light_trace(best_s,k);
        trace(v,l) = cur_l;

        if (v < nPos_)
          best_s -= cur_l;
        else
          best_s += cur_l;
      }
    }
  }

  return 0.0; //currently not subtracting an offest
}

/*virtual*/
double BILPChainDDFactor::compute_sum_forward_logspace(const ChainDDVar* in_var, const ChainDDVar* out_var,
    const Math1D::Vector<double>& prev_log_forward, Math1D::Vector<double>& log_forward, double mu) const noexcept
{
  double log_offs = 0.0;

  const uint nVars = involved_var_.size();

  assert(out_var != in_var);

  Storage1D<Math1D::Vector<double> > param = dual_var_;

  uint idx = MAX_UINT;

  for (uint k=0; k < nVars; k++) {
    Math1D::Vector<double>& cur_param = param[k];

    Math1D::Vector<double> temp(cur_param.size());

    if (involved_var_[k] == in_var) {

      cur_param *= 1.0 / mu;

      cur_param += prev_log_forward;
    }
    else {
      if (involved_var_[k] == out_var) {
        idx = k;
      }

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_param.size(); l++)
        cur_param[l] -= cur_cost[l]; //negative cost since we maximize
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  //std::cerr << "param: " << param << std::endl;
  //std::cerr << "idx: " << idx << std::endl;

  log_forward.resize_dirty(2);

  Math1D::Vector<double> log_forward_vec[2];
  log_forward_vec[0].resize(range_,-1e100);
  log_forward_vec[1].resize(range_,-1e100); //init is needed because the loops below do not visit all entries (to reduce run-time)

  const uint start_idx = (idx != 0) ? 0 : 1;

  uint cur_idx = 0;
  Math1D::Vector<double>& start_log_forward_vec = log_forward_vec[0];

  start_log_forward_vec[zero_offset_] = param[start_idx][0];

  const int init_mul = (start_idx < nPos_) ? 1 : -1;
  if (int(zero_offset_)+init_mul >= 0
      && int(zero_offset_)+init_mul < range_) {
    start_log_forward_vec[zero_offset_+init_mul] = param[start_idx][1];
  }

  Math1D::Vector<double> temp(2);

  //proceed for positive vars
  uint k=0;
  for (uint v= start_idx + 1; v < nPos_; v++) {

    if (v != idx) {

      k++;

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward_vec = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];

      const int cur_limit = std::min<int>(range_,zero_offset_+k+2);

      //for (int sum=0; sum < range_; sum++) {
      for (int sum=zero_offset_; sum < cur_limit; sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum - l;
          if (dest >= 0) {

            temp[l] = prev_log_forward_vec[dest] + cur_param[l];
          }
          else {
            temp[l] = -1e100;
          }
        }

        assert(temp.max() > -1e100);

        if (temp[0] >= temp[1]) {
          cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        }
        else {
          cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        }

      }

    }
  }

  //proceed for negative vars
  for (uint v=std::max<uint>((idx <= 1) ? 2 : 1,nPos_); v < nVars; v++) {

    if (v != idx) {

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];

      for (int sum=0; sum < range_; sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum + l;
          if (dest < range_) {

            temp[l] = prev_log_forward[dest] + cur_param[l];
          }
          else
            temp[l] = -1e100;
        }

        const double offs = std::max(temp[0],temp[1]);

        if (offs == -1e100)
          cur_log_forward_vec[sum] = -1e100;
        else {
          double tsum = 0.0;
          for (uint k=0; k < 2; k++) {
            tsum += (temp[k] != offs) ? std::exp(temp[k]-offs) : 1.0;
          }
          cur_log_forward_vec[sum] = offs + std::log(tsum);
        }

        if (temp[0] >= temp[1]) {
          cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        }
        else {
          cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        }
      }
    }
  }

  //now final computation for idx
  const Math1D::Vector<double>& last_log_forward_vec = log_forward_vec[cur_idx];

  const Math1D::Vector<double>& idx_param = param[idx];

  const int lower_limit = int(rhs_lower_ + zero_offset_);
  const int upper_limit = int(rhs_upper_ + zero_offset_);

  Math1D::Vector<double> cur_temp(1+upper_limit-lower_limit);

  for (uint l=0; l < 2; l++) {

    //std::cerr << "l: " << l << std::endl;

    for (int s=lower_limit; s <= upper_limit; s++) {

      int move = l;
      if (idx < nPos_)
        move = -move;

      const int dest = s + move;
      if (dest >= 0 && dest < range_) {

        cur_temp[s-lower_limit] = last_log_forward_vec[dest];
      }
      else
        cur_temp[s-lower_limit] = -1e100;
    }

    double offs = cur_temp.max();
    double sum = 0.0;

    for (uint k=0; k < cur_temp.size(); k++) {
      sum += (cur_temp[k] != offs) ? std::exp(cur_temp[k] - offs) : 1.0;
    }

    assert(sum > 0.0); //should be the case when the variable can have this label

    log_forward[l] = offs + std::log(sum) + idx_param[l];
  }

  return log_offs;
}

/*virtual*/
void BILPChainDDFactor::compute_marginals_logspace(const ChainDDVar* target, const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Math1D::Vector<double>& marginal) const noexcept
{
  const uint nVars = involved_var_.size();

  uint idx = MAX_UINT;

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == target) {
      idx = k;
    }

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  marginal.resize_dirty(2);

  Math1D::Vector<double> log_forward_vec[2];
  log_forward_vec[0].resize(range_,-1e100);
  log_forward_vec[1].resize(range_,-1e100); //init is needed because the loops below do not visit all entries (to reduce run-time)

  const uint start_idx = (idx != 0) ? 0 : 1;

  uint cur_idx = 0;
  Math1D::Vector<double>& start_log_forward_vec = log_forward_vec[0];

  start_log_forward_vec[zero_offset_] = param[start_idx][0];

  const int init_mul = (start_idx < nPos_) ? 1 : -1;
  if (int(zero_offset_)+init_mul >= 0
      && int(zero_offset_)+init_mul < range_) {
    start_log_forward_vec[zero_offset_+init_mul] = param[start_idx][1];
  }

  Math1D::Vector<double> temp(2);

  //proceed for positive vars
  uint k=0;

  for (uint v= start_idx + 1; v < nPos_; v++) {

    if (v != idx) {

      k++;

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward_vec = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];


      //for (int sum=0; sum < range_; sum++) {
      for (int sum=zero_offset_; sum < std::min<int>(range_,zero_offset_+k+2); sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum - l;
          if (dest >= 0) {

            temp[l] = prev_log_forward_vec[dest] + cur_param[l];
          }
          else {
            temp[l] = -1e100;
          }
        }

        assert(temp.max() > -1e100);

        if (temp[0] >= temp[1]) {
          cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        }
        else {
          cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        }
      }
    }
  }

  //proceed for negative vars
  for (uint v=std::max<uint>((idx <= 1) ? 2 : 1,nPos_); v < nVars; v++) {

    if (v != idx) {

      //k++; //not needed here

      const Math1D::Vector<double>& cur_param = param[v];

      const Math1D::Vector<double>& prev_log_forward = log_forward_vec[cur_idx];

      cur_idx = 1 - cur_idx;

      Math1D::Vector<double>& cur_log_forward_vec = log_forward_vec[cur_idx];

      for (int sum=0; sum < range_; sum++) {

        for (int l=0; l < 2; l++) {

          const int dest = sum + l;
          if (dest < range_) {

            temp[l] = prev_log_forward[dest] + cur_param[l];
          }
          else
            temp[l] = -1e100;
        }

        const double offs = std::max(temp[0],temp[1]);

        if (offs == -1e100)
          cur_log_forward_vec[sum] = -1e100;
        else {
          double tsum = 0.0;
          for (uint i=0; i < 2; i++) {
            tsum += (temp[i] != offs) ? std::exp(temp[i]-offs) : 1.0;
          }
          cur_log_forward_vec[sum] = offs + std::log(tsum);
        }

        // if (temp[0] >= temp[1]) {
        //   cur_log_forward_vec[sum] = temp[0] + std::log(1.0 + std::exp(temp[1]-temp[0]));
        // }
        // else {
        //   cur_log_forward_vec[sum] = temp[1] + std::log(1.0 + std::exp(temp[0]-temp[1]));
        // }


      }
    }
  }

  //now final computation for idx
  const Math1D::Vector<double>& last_log_forward_vec = log_forward_vec[cur_idx];

  const Math1D::Vector<double>& idx_param = param[idx];

  const int lower_limit = int(rhs_lower_ + zero_offset_);
  const int upper_limit = int(rhs_upper_ + zero_offset_);

  Math1D::Vector<double> cur_temp(1+upper_limit-lower_limit);

  for (uint l=0; l < 2; l++) {

    //std::cerr << "l: " << l << std::endl;

    for (int s=lower_limit; s <= upper_limit; s++) {

      int move = l;
      if (idx < nPos_)
        move = -move;

      const int dest = s + move;
      if (dest >= 0 && dest < range_) {

        cur_temp[s-lower_limit] = last_log_forward_vec[dest];
      }
      else
        cur_temp[s-lower_limit] = -1e100;
    }

    const double offs = cur_temp.max();
    double sum = 0.0;

    for (uint k=0; k < cur_temp.size(); k++) {
      sum += (cur_temp[k] != offs) ? std::exp(cur_temp[k] - offs) : 1.0;
    }

    assert(sum > 0.0); //should be the case when the variable can have this label

    marginal[l] = offs + std::log(sum) + idx_param[l]; //store logs for now
  }


  //now convert to exponential format
  const double offs = marginal.max();
  for (uint l=0; l < marginal.size(); l++) {
    //marginal[l] = std::exp(marginal[l]-offs);

    marginal[l] = (marginal[l] != offs) ? std::exp(marginal[l]-offs) : 1.0;
  }

  marginal *= 1.0 / marginal.sum(); //renormalize to get the actual marginals
}

#if 1
/*virtual*/
void BILPChainDDFactor::compute_all_marginals_logspace(const ChainDDVar* in_var1, const ChainDDVar* in_var2,
    const Math1D::Vector<double>& log_forward1, const Math1D::Vector<double>& log_forward2,
    double mu, Storage1D<Math1D::Vector<double> >& marginal) const noexcept
{
  const uint nVars = involved_var_.size();

  //for backward, we need different bounds than stored in the range_ and zero_offset_
  int nPositive = nPos_;
  int nNegative = involved_var_.size()-nPositive;

  int lower_bound = -nNegative;
  lower_bound = std::max(lower_bound, rhs_lower_ - nPositive);

  int upper_bound = nPositive;
  upper_bound = std::min(upper_bound, rhs_upper_ + nNegative);

  const int range = upper_bound - lower_bound + 1;
  const int zero_offset = -lower_bound;

  marginal.resize(nVars);

  Storage1D<Math1D::Vector<double> > param = dual_var_;
  for (uint k=0; k < nVars; k++) {

    Math1D::Vector<double>& cur_param = param[k];

    if (involved_var_[k] == in_var1) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward1;
    }
    else if (involved_var_[k] == in_var2) {

      cur_param *= 1.0 / mu;
      cur_param += log_forward2;
    }
    else {

      const Math1D::Vector<float>& cur_cost = involved_var_[k]->cost();

      for (uint l=0; l < cur_cost.size(); l++) {
        cur_param[l] -= cur_cost[l];
      }
      cur_param *= 1.0 / mu;
    }

    //std::cerr << "param[" << k << "]: " << param[k] << std::endl;
  }

  Math1D::Vector<double> temp(2);

  //a) forward

  //std::cerr << "forward" << std::endl;

  Math2D::Matrix<double> log_forward(range,nVars-1,-1e100);

  log_forward(zero_offset,0) = param[0][0];

  const int init_mul = (0 < nPos_) ? 1 : -1;
  if (int(zero_offset)+init_mul >= 0 && int(zero_offset)+init_mul < range) {
    log_forward(zero_offset+init_mul,0) = param[0][1];
  }

  for (uint v=1; v < nVars-1; v++) { //for now just one loop, can split into positive and negative later

    const Math1D::Vector<double>& cur_param = param[v];

    for (uint sum=0; sum < range; sum++) {

      for (int l=0; l < 2; l++) {

        const int dest = (v < nPos_) ? (sum - l) : (sum + l);
        if (dest >= 0 && dest < range) {

          temp[l] = log_forward(dest,v-1) + cur_param[l];
        }
        else {
          temp[l] = -1e100;
        }
      }

      const double offs = std::max(temp[0],temp[1]);

      if (offs == -1e100)
        log_forward(sum,v) = -1e100;
      else {
        double tsum = 0.0;
        for (uint i=0; i < 2; i++) {
          tsum += (temp[i] != offs) ? std::exp(temp[i]-offs) : 1.0;
        }
        log_forward(sum,v) = offs + std::log(tsum);
      }

    }
  }

  // b) backward

  //std::cerr << "backward" << std::endl;

  Math2D::Matrix<double> log_backward(range,nVars,-1e100);

  log_backward(zero_offset,nVars-1) = param[nVars-1][0];

  const int bwd_init_mul = (nPos_ == nVars) ? 1 : -1;
  if (int(zero_offset)+bwd_init_mul >= 0
      && int(zero_offset)+bwd_init_mul < range) {
    log_backward(zero_offset+bwd_init_mul,nVars-1) = param[nVars-1][1];
  }

  for (int v=nVars-2; v >= 1; v--) {

    const Math1D::Vector<double>& cur_param = param[v];

    for (uint sum=0; sum < range; sum++) {

      for (int l=0; l < 2; l++) {

        const int dest = (v < nPos_) ? (sum - l) : (sum + l);
        if (dest >= 0 && dest < range) {

          temp[l] = log_backward(dest,v+1) + cur_param[l];
        }
        else {
          temp[l] = -1e100;
        }
      }

      const double offs = std::max(temp[0],temp[1]);

      if (offs == -1e100)
        log_backward(sum,v) = -1e100;
      else {
        double tsum = 0.0;
        for (uint i=0; i < 2; i++) {
          tsum += (temp[i] != offs) ? std::exp(temp[i]-offs) : 1.0;
        }
        log_backward(sum,v) = offs + std::log(tsum);
      }

    }
  }

  // c) now finally compute the marginals
  //std::cerr << "final comp." << std::endl;

  //std::cerr << "nPos: " << nPos_ << ", zero_offs: " << zero_offset << ", " << rhs_lower_ << " <= cx <= " << rhs_upper_ << std::endl;

  //std::cerr << "forward: " << std::endl << log_forward << std::endl;
  //std::cerr << "backward: " << std::endl << log_backward << std::endl;

  for (uint v=0; v < nVars; v++) {

    //std::cerr << "v: " << v << "/" << nVars << std::endl;

    Math1D::Vector<double>& cur_marginal = marginal[v];

    const Math1D::Vector<double>& cur_param = param[v];

    //std::cerr << "param: " << cur_param << std::endl;

    cur_marginal.resize_dirty(2);

    for (int l=0; l < 2; l++) {

      std::vector<double> cur_temp;

      //std::cerr << "l: " << l << std::endl;

      if (v == 0) {

        for (int sum=rhs_lower_; sum <= rhs_upper_; sum++) {

          //std::cerr << "sum: " << sum << std::endl;

          const int dest = int(zero_offset) + ((v < nPos_) ? (sum-l) : (sum+l));

          if (dest >= 0 && dest < range) {
            cur_temp.push_back(log_backward(dest,1));
          }
        }
      }
      else if (v == nVars-1) {

        for (int sum=rhs_lower_; sum <= rhs_upper_; sum++) {

          const int dest = int(zero_offset) + ((v < nPos_) ? (sum-l) : (sum+l));

          if (dest >= 0 && dest < range) {
            cur_temp.push_back(log_forward(dest,nVars-2));
          }
        }
      }
      else {

        for (int sum=0; sum < range; sum++) {

          if (log_forward(sum,v-1) == -1e100)
            continue;

          int diff = (sum - zero_offset);
          if (v < nPos_)
            diff += l;
          else
            diff -= l;

          for (int r=rhs_lower_; r <= rhs_upper_; r++) {

            const int other = r + zero_offset - diff;

            if (other >= 0 && other < (int) range) {

              const double cand = log_forward(sum,v-1) + log_backward(other,v+1);

              if (cand > -1e98) {
                cur_temp.push_back(cand);
              }
            }
          }

        }
      }

      //std::cerr << "common" << std::endl;

      if (cur_temp.size() == 0)
        cur_marginal[l] = -1e100;
      else {

        const double offs = vec_max(cur_temp);
        double sum = 0.0;
        for (uint k=0; k < cur_temp.size(); k++) {
          sum += (cur_temp[k] != offs) ? std::exp(cur_temp[k] - offs) : 1.0;
        }
        cur_marginal[l] = offs + std::log(sum) + cur_param[l]; //store logs for now
      }
    }

    //std::cerr << "intermediate result: " << cur_marginal << std::endl;

    //now convert to exponential format
    const double offs = cur_marginal.max();
    for (uint l=0; l < cur_marginal.size(); l++) {
      //cur_marginal[l] = std::exp(cur_marginal[l]-offs);

      cur_marginal[l] = (cur_marginal[l] != offs) ? std::exp(cur_marginal[l]-offs) : 1.0;
    }

    cur_marginal *= 1.0 / cur_marginal.sum(); //renormalize to get the actual marginals

    //DEBUG
    // Math1D::Vector<double> check_marginal;
    // compute_marginals_logspace(involved_var_[v], in_var1, in_var2, log_forward1, log_forward2, mu, check_marginal);

    // //std::cerr << "computed: " << cur_marginal << ", should be: " << check_marginal << std::endl;
    // check_marginal -= cur_marginal;
    // assert(check_marginal.sqr_norm() <= 0.001);
    //END_DEBUG
  }
}
#endif


/********************************************/

FactorChainDualDecomposition::FactorChainDualDecomposition(uint nVars, uint nFactors) :
  nUsedVars_(0), nUsedFactors_(0), optimize_called_(false)
{
  var_.resize(nVars);
  factor_.resize(nFactors);
}

FactorChainDualDecomposition::~FactorChainDualDecomposition()
{
  for (uint v=0; v < nUsedVars_; v++)
    delete var_[v];
  for (uint f=0; f < nUsedFactors_; f++)
    delete factor_[f];
}

uint FactorChainDualDecomposition::add_var(const Math1D::Vector<float>& cost)
{
  assert(!optimize_called_);

  if (nUsedVars_ == var_.size())
    var_.resize(uint(1.2*nUsedVars_)+4);

  assert(nUsedVars_ < var_.size());
  var_[nUsedVars_] = new ChainDDVar(cost);

  nUsedVars_++;

  return nUsedVars_-1;
}

uint FactorChainDualDecomposition::add_factor(ChainDDFactor* fac)
{
  assert(!optimize_called_);

  if (nUsedFactors_ == factor_.size())
    factor_.resize(uint(1.2*nUsedFactors_)+4);

  factor_[nUsedFactors_] = fac;
  nUsedFactors_++;

  return nUsedFactors_-1;
}

uint FactorChainDualDecomposition::add_generic_factor(const Math1D::Vector<uint> var, const VarDimStorage<float>& cost)
{
  Storage1D<ChainDDVar*> vars(var.size());

  for (uint v=0; v < var.size(); v++) {
    if (var[v] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    vars[v] = var_[var[v]];
  }

  return add_factor(new GenericChainDDFactor(vars,cost));
}

uint FactorChainDualDecomposition::add_binary_factor(uint var1, uint var2, const Math2D::Matrix<float>& cost, bool ref)
{
  if (var1 >= nUsedVars_ || var2 >= nUsedVars_) {
    INTERNAL_ERROR << "out of range. Exiting." << std::endl;
    exit(1);
  }

  Storage1D<ChainDDVar*> vars(2);
  vars[0] = var_[var1];
  vars[1] = var_[var2];

  ChainDDFactor* newFac;

  if (ref)
    newFac = new BinaryChainDDRefFactor(vars,cost);
  else
    newFac = new BinaryChainDDFactor(vars,cost);

  return add_factor(newFac);
}

uint FactorChainDualDecomposition::add_ternary_factor(uint var1, uint var2, uint var3,
    const Math3D::Tensor<float>& cost, bool ref)
{
  if (var1 >= nUsedVars_ || var2 >= nUsedVars_ || var3 >= nUsedVars_) {
    INTERNAL_ERROR << "out of range. Exiting." << std::endl;
    exit(1);
  }

  Storage1D<ChainDDVar*> vars(3);
  vars[0] = var_[var1];
  vars[1] = var_[var2];
  vars[2] = var_[var3];

  ChainDDFactor* new_fac = 0;

  if (!ref)
    new_fac = new TernaryChainDDFactor(vars,cost);
  else
    new_fac = new TernaryChainDDRefFactor(vars,cost);

  return add_factor(new_fac);
}

uint FactorChainDualDecomposition::add_fourth_order_factor(uint var1, uint var2, uint var3, uint var4,
    const Storage1D<Math3D::Tensor<float> >& cost, bool ref)
{
  if (var1 >= nUsedVars_ || var2 >= nUsedVars_ || var3 >= nUsedVars_ || var4 >= nUsedVars_) {
    INTERNAL_ERROR << "out of range. Exiting." << std::endl;
    exit(1);
  }

  Storage1D<ChainDDVar*> vars(4);
  vars[0] = var_[var1];
  vars[1] = var_[var2];
  vars[2] = var_[var3];
  vars[3] = var_[var4];

  ChainDDFactor* new_fac;

  if (ref)
    new_fac = new FourthOrderChainDDRefFactor(vars,cost);
  else
    new_fac = new FourthOrderChainDDFactor(vars,cost);

  return add_factor(new_fac);
}

uint FactorChainDualDecomposition::add_second_diff_factor(uint var1, uint var2, uint var3, float lambda)
{
  if (var1 >= nUsedVars_ || var2 >= nUsedVars_ || var3 >= nUsedVars_) {
    INTERNAL_ERROR << "out of range. Exiting." << std::endl;
    exit(1);
  }

  Storage1D<ChainDDVar*> vars(3);
  vars[0] = var_[var1];
  vars[1] = var_[var2];
  vars[2] = var_[var3];

  return add_factor(new SecondDiffChainDDFactor(vars,lambda));
}

uint FactorChainDualDecomposition::add_generalized_potts_factor(const Math1D::Vector<uint>& var, float lambda)
{
  Storage1D<ChainDDVar*> vars(var.size());

  for (uint k=0; k < var.size(); k++) {

    if (var[k] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    vars[k] = var_[var[k]];
  }

  return add_factor(new GeneralizedPottsChainDDFactor(vars,lambda));
}

uint FactorChainDualDecomposition::add_one_of_n_factor(const Math1D::Vector<uint>& var)
{
  Storage1D<ChainDDVar*> vars(var.size());

  for (uint k=0; k < var.size(); k++) {

    if (var[k] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    vars[k] = var_[var[k]];

    if (vars[k]->nLabels() != 2) {
      INTERNAL_ERROR << " variables of 1-of-N nodes must be binary. Exiting..." << std::endl;
      exit(1);
    }
  }

  if (var.size() == 1) {

    Math1D::Vector<float> cost(2);
    cost[0] = 1e20;
    cost[1] = 0.0;

    vars[0]->add_cost(cost);

    return MAX_UINT;
  }
  else
    return add_factor(new OneOfNChainDDFactor(vars));
}

uint FactorChainDualDecomposition::add_cardinality_factor(const Math1D::Vector<uint>& var,
    const Math1D::Vector<float>& cost, bool ref)
{
  if (cost.size() <= var.size()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
    exit(1);
  }

  if (var.size() == 1) {
    if (var[0] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    if (var_[var[0]]->nLabels() != 2) {
      INTERNAL_ERROR << " variables of cardinality nodes must be binary. Exiting..." << std::endl;
      exit(1);
    }
    var_[var[0]]->add_cost(cost);

    return MAX_UINT;
  }
  else {

    Storage1D<ChainDDVar*> vars(var.size());

    for (uint k=0; k < var.size(); k++) {

      if (var[k] >= nUsedVars_) {
        INTERNAL_ERROR << "out of range. Exiting." << std::endl;
        exit(1);
      }

      vars[k] = var_[var[k]];

      if (vars[k]->nLabels() != 2) {
        INTERNAL_ERROR << " variables of cardinality nodes must be binary. Exiting..." << std::endl;
        exit(1);
      }
    }

    if (!ref)
      return add_factor(new CardinalityChainDDFactor(vars,cost));
    else
      return add_factor(new CardinalityChainDDRefFactor(vars,cost));
  }
}

uint FactorChainDualDecomposition::add_nonbinary_cardinality_factor(const Math1D::Vector<uint>& var,
    const Math1D::Vector<float>& cost,
    const Math1D::Vector<uint>& level)
{
  if (cost.size() <= var.size()) {
    INTERNAL_ERROR << "dimension mismatch. Exiting." << std::endl;
    exit(1);
  }

  if (var.size() == 1) {
    if (var[0] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    if (var_[var[0]]->nLabels() <= level[0]) {
      INTERNAL_ERROR << " dimension mismatch. Exiting..." << std::endl;
      exit(1);
    }

    Math1D::Vector<float> addon(var_[var[0]]->nLabels(),cost[0]);
    addon[level[0]] = cost[1];
    var_[var[0]]->add_cost(cost);

    return MAX_UINT;
  }
  else {

    Storage1D<ChainDDVar*> vars(var.size());

    for (uint k=0; k < var.size(); k++) {

      if (var[k] >= nUsedVars_) {
        INTERNAL_ERROR << "out of range. Exiting." << std::endl;
        exit(1);
      }

      vars[k] = var_[var[k]];

      if (vars[k]->nLabels() <= level[k]) {
        INTERNAL_ERROR << " dimension mismatch. Exiting..." << std::endl;
        exit(1);
      }
    }

    return add_factor(new NonbinaryCardinalityChainDDFactor(vars,cost,level));
  }

}

uint FactorChainDualDecomposition::add_binary_ilp_factor(const Math1D::Vector<uint>& var, const Storage1D<bool>& positive,
    int rhs_lower, int rhs_upper)
{
  if (rhs_lower > rhs_upper) {
    INTERNAL_ERROR << " INFEASIBLE CONSTRAINT" << std::endl;
    exit(1);
  }

  uint nUseful = 0;
  int nPos = 0;
  int nNeg = 0;
  for (uint k=0; k < var.size(); k++) {

    if (var[k] >= nUsedVars_) {
      INTERNAL_ERROR << "out of range. Exiting." << std::endl;
      exit(1);
    }

    if (var_[var[k]]->nLabels() != 2) {
      INTERNAL_ERROR << "attempt to instantiate a BILP factor with non-binary variables. Exiting." << std::endl;
      exit(1);
    }

    if (fabs(var_[var[k]]->cost()[0] - var_[var[k]]->cost()[1]) < 1e10) {
      nUseful++;
      if (positive[k])
        nPos++;
      else
        nNeg++;
    }
    else {
      if (var_[var[k]]->cost()[0] > var_[var[k]]->cost()[1]) {
        if (positive[k]) {
          rhs_lower--;
          rhs_upper--;
        }
        else {
          rhs_lower++;
          rhs_upper++;
        }
      }
    }
  }

  if (nUseful != 0 && rhs_lower <= -nNeg && rhs_upper >= nPos)
    nUseful = 0; //constraint is always true


  if (nUseful == 0) {

    std::cerr << "WARNING: removed superfluous constraint factor" << std::endl;

    return MAX_UINT;
  }

  if (rhs_upper < -nNeg || rhs_lower > nPos) {

    INTERNAL_ERROR << " INFEASIBLE CONSTRAINT" << std::endl;
    exit(1);
  }

  // if (nUseful < 2) {
  //   std::cerr << "only " << nUseful << " out of " << var.size() << " variables are actually not fixed" << std::endl;

  //   for (uint k=0; k < var.size(); k++)
  //     std::cerr << "cost: " << var_[var[k]]->cost() << std::endl;

  //   std::cerr << "var: " << var << std::endl;
  // }

  Storage1D<ChainDDVar*> vars(nUseful);
  Storage1D<bool> reduced_positive(nUseful);

  uint next = 0;

  for (uint k=0; k < var.size(); k++) {

    if (fabs(var_[var[k]]->cost()[0] - var_[var[k]]->cost()[1]) < 1e10) {
      vars[next] = var_[var[k]];

      if (vars[next]->nLabels() != 2) {
        INTERNAL_ERROR << " variables of BILP nodes must be binary. Exiting..." << std::endl;
        exit(1);
      }

      reduced_positive[next] = positive[k];
      next++;
    }
  }

  assert(next == nUseful);

  if (nUseful == 1) {

    std::cerr << "happens" << std::endl;

    if (nPos == 0) {

      //invert the constraint

      double temp_lower = rhs_lower;
      rhs_lower = -rhs_upper;
      rhs_upper = -temp_lower;
    }

    Math1D::Vector<float> add_cost(2,0.0);

    if (rhs_lower == 1) {
      add_cost[0] = 1e30;
    }
    else if (rhs_upper == 0) {
      add_cost[1] = 1e30;
    }
    else {
      INTERNAL_ERROR << "STRANGE CONSTRAINT. exiting" << std::endl;
      exit(1);
    }

    vars[0]->add_cost(add_cost);
    return MAX_UINT;
  }
  else {

    if (nNeg == 0)
      // if (false)
      return add_factor(new AllPosBILPChainDDFactor(vars,rhs_lower,rhs_upper));
    else
      return add_factor(new BILPChainDDFactor(vars,reduced_positive,rhs_lower,rhs_upper));
  }
}

uint FactorChainDualDecomposition::pass_in_factor(ChainDDFactor* fac)
{
  return add_factor(fac);
}

ChainDDVar* FactorChainDualDecomposition::get_variable(uint v)
{
  if (v >= nUsedVars_) {
    INTERNAL_ERROR << "variable index out of bounds. Exiting." << std::endl;
    exit(1);
  }

  return var_[v];
}

ChainDDFactor* FactorChainDualDecomposition::get_factor(uint f)
{
  if (f >= nUsedFactors_) {
    INTERNAL_ERROR << "factor index out of bounds. Exiting." << std::endl;
    exit(1);
  }

  return factor_[f];
}

const Math1D::Vector<uint>& FactorChainDualDecomposition::labeling()
{
  return labeling_;
}

void FactorChainDualDecomposition::set_up_chains()
{
  //use non-monotonic chains

  uint nChains = 0;
  uint nAtLeast5 = 0;
  uint nAtLeast10 = 0;
  uint nAtLeast25 = 0;

  for (uint f=0; f < nUsedFactors_; f++) {

    //std::cerr << "factor #" << f << "/" << nUsedFactors_ << std::endl;

    if (factor_[f]->prev_var() == 0 && factor_[f]->next_var() == 0) {

      uint length = 1;

      std::set<ChainDDVar*> current_vars;

      //std::cerr << "extend lower" << std::endl;

      bool extension_found = true;

      ChainDDFactor* cur_factor = factor_[f];

#if 1
      //extend lower end
      while (extension_found) {

        extension_found = false;

        const Storage1D<ChainDDVar*>& involved_vars = cur_factor->involved_vars();

        for (uint k=0; k < involved_vars.size(); k++) {
          current_vars.insert(involved_vars[k]);
        }

        for (double k=0; k < involved_vars.size(); k++) {

          ChainDDVar* var = involved_vars[k];

          if (var == cur_factor->next_var())
            continue;

          const Storage1D<ChainDDFactor*>& adjacent_factor = var->neighboring_factor();

          for (uint l=0; l < adjacent_factor.size(); l++) {

            ChainDDFactor* hyp_factor = adjacent_factor[l];
            const Storage1D<ChainDDVar*>& hyp_involved_vars = hyp_factor->involved_vars();

            bool is_valid_extension = false;

            if (hyp_factor != cur_factor
                && hyp_factor->prev_var() == 0 && hyp_factor->next_var() == 0) {

              is_valid_extension = true;

              for (uint v=0; v < hyp_involved_vars.size(); v++) {

                if (hyp_involved_vars[v] != var && current_vars.find(hyp_involved_vars[v]) != current_vars.end())
                  is_valid_extension = false;
              }

              if (is_valid_extension) {

                extension_found = true;
                cur_factor->set_prev_var(var);
                cur_factor->set_prev_factor(hyp_factor);

                hyp_factor->set_next_var(var);
                hyp_factor->set_next_factor(cur_factor);

                cur_factor = hyp_factor;

                length++;

                break;
              }
            }
          }

          if (extension_found)
            break;
        }
      }
#endif

#if 1
      //extend upper end

      //std::cerr << "extend upper" << std::endl;

      cur_factor = factor_[f];
      extension_found = true;

      while (extension_found) {

        extension_found = false;

        const Storage1D<ChainDDVar*>& involved_vars = cur_factor->involved_vars();

        for (uint k=0; k < involved_vars.size(); k++) {
          current_vars.insert(involved_vars[k]);
        }

        for (double k=0; k < involved_vars.size(); k++) {

          ChainDDVar* var = involved_vars[k];

          if (var == cur_factor->prev_var())
            continue;

          const Storage1D<ChainDDFactor*>& adjacent_factor = var->neighboring_factor();

          for (uint l=0; l < adjacent_factor.size(); l++) {

            ChainDDFactor* hyp_factor = adjacent_factor[l];

            bool is_valid_extension = false;

            if (hyp_factor != cur_factor
                && hyp_factor->prev_var() == 0 && hyp_factor->next_var() == 0) {

              const Storage1D<ChainDDVar*>& hyp_involved_vars = hyp_factor->involved_vars();

              is_valid_extension = true;

              for (uint v=0; v < hyp_involved_vars.size(); v++) {

                if (hyp_involved_vars[v] != var && current_vars.find(hyp_involved_vars[v]) != current_vars.end())
                  is_valid_extension = false;
              }

              if (is_valid_extension) {

                extension_found = true;
                cur_factor->set_next_var(var);
                cur_factor->set_next_factor(hyp_factor);

                hyp_factor->set_prev_var(var);
                hyp_factor->set_prev_factor(cur_factor);

                cur_factor = hyp_factor;

                length++;

                break;
              }
            }

          }
          if (extension_found)
            break;
        }
      }
#endif

      nChains++;

      if (length >= 5)
        nAtLeast5++;
      if (length >= 10)
        nAtLeast10++;
      if (length >= 25)
        nAtLeast25++;


      // if (length > 15)
      //   std::cerr << "chain length " << length << std::endl;
    }
  }

  // std::cerr << nAtLeast5 << " chains have length at least 5." << std::endl;
  // std::cerr << nAtLeast10 << " chains have length at least 10." << std::endl;
  // std::cerr << nAtLeast25 << " chains have length at least 25." << std::endl;
  // std::cerr << nChains << " chains in total, " << nUsedFactors_ << " factors" << std::endl;
}

void FactorChainDualDecomposition::set_up_singleton_chains()
{
  for (uint f=0; f < nUsedFactors_; f++) {

    ChainDDFactor* cur_factor = factor_[f];

    cur_factor->set_next_var(0);
    cur_factor->set_next_factor(0);

    cur_factor->set_prev_var(0);
    cur_factor->set_prev_factor(0);
  }
}

void FactorChainDualDecomposition::extract_chains(std::vector<std::vector<ChainDDFactor*> >& chain,
    std::vector<std::vector<ChainDDVar*> >& out_var, std::vector<ChainDDVar*>& in_var)
{
  chain.clear();
  out_var.clear();
  in_var.clear();

  for (uint f=0; f < nUsedFactors_; f++) {

    //std::cerr << "f: " << f << " prev factor: " << factor_[f]->prev_factor() << std::endl;

    if (factor_[f]->prev_factor() == 0) {

      //std::cerr << "new chain" << std::endl;

      chain.push_back(std::vector<ChainDDFactor*>());
      out_var.push_back(std::vector<ChainDDVar*>());

      std::set<ChainDDVar*> cur_vars;

      ChainDDFactor* cur_factor = factor_[f];

      ChainDDVar* cur_in_var = 0;
      for (uint k=0; k < cur_factor->involved_vars().size(); k++) {
        if (cur_factor->involved_vars()[k] != cur_factor->next_var()) {
          cur_in_var = cur_factor->involved_vars()[k];
        }
      }
      in_var.push_back(cur_in_var);

      while (cur_factor != 0) {

        for (uint k=0; k < cur_factor->involved_vars().size(); k++) {
          cur_vars.insert(cur_factor->involved_vars()[k]);
        }

        chain.back().push_back(cur_factor);
        assert(cur_factor->next_var() != in_var.back());
        //assert(std::find(out_var.begin(),out_var.end(),cur_factor->next_var()) == out_var.end());
        out_var.back().push_back(cur_factor->next_var());
        cur_factor = cur_factor->next_factor();
      }

      //find chain end
      for (uint k=0; k < chain.back().back()->involved_vars().size(); k++) {

        if (chain.back().back()->involved_vars()[k] != chain.back().back()->prev_var()
            && chain.back().back()->involved_vars()[k] != in_var.back()) { //can happen for chains of length 1
          out_var.back().back() = chain.back().back()->involved_vars()[k];
          break;
        }
      }
    }
  }
}

double FactorChainDualDecomposition::optimize(uint nIter, double start_step_size, bool quiet)
{
  std::cerr.precision(10);

  if (!quiet) {
    std::cerr << "subgradient optimization" << std::endl;
    std::cerr << nUsedFactors_ << " factors" << std::endl;
  }

  if (!optimize_called_) {
    set_up_chains();

    for (uint v=0; v < nUsedVars_; v++)
      var_[v]->set_up_chains();
  }

  optimize_called_ = true;

  bool projective = true;

  double best_dual = -1e300;
#ifdef PRIMAL_DUAL_STEPSIZE
  double best_primal = 1e300;
#endif

  Math1D::Vector<uint> var_label(nUsedVars_);

  labeling_.resize(nUsedVars_,0);

  Storage1D<Math1D::Vector<uint> > factor_label(nUsedFactors_);

  // double denom = 0.0;

  for (uint f=0; f < nUsedFactors_; f++) {
    factor_label[f].resize(factor_[f]->involved_vars().size());
    //   denom += factor_[f]->involved_vars().size();
  }

  std::map<const ChainDDVar*,uint> var_num;
  for (uint v=0; v < nUsedVars_; v++)
    var_num[var_[v]] = v;

  std::map<const ChainDDFactor*,uint> factor_num;
  for (uint f=0; f < nUsedFactors_; f++)
    factor_num[factor_[f]] = f;

  uint nIncreases = 1;

  double delta = start_step_size;

  size_t effort_per_iter = 0;

  for (uint f=0; f < nUsedFactors_; f++) {

    uint size = factor_[f]->involved_vars().size();

    effort_per_iter += size;
  }

  //DEBUG (for ICML)
  //std::ofstream of("sg.dat");
  //END_DEBUG

  //store chains for efficient access

  std::vector<std::vector<ChainDDFactor*> > chain;
  std::vector<std::vector<ChainDDVar*> > out_var;
  std::vector<ChainDDVar*> in_var;

  extract_chains(chain, out_var, in_var);

  for (uint iter=1; iter <= nIter; iter++) {

    if (!quiet)
      std::cerr << "iteration #" << iter << std::endl;

    uint nDisagreements = 0;

    //double step_size = start_step_size / iter;
    double step_size = start_step_size / nIncreases;

    double cur_bound = 0.0;

    if (!projective) {
      for (uint v=0; v < nUsedVars_; v++) {
        uint cur_label = 0;
        cur_bound += var_[v]->dual_value(cur_label);
        var_label[v] = cur_label;
      }

      if (!quiet)
        std::cerr << "A, intermediate bound: " << cur_bound << std::endl;
    }

    //uint nLongChainsProcessed = 0;

    for (uint k=0; k < chain.size(); k++) {

      if (true) {

        const std::vector<ChainDDFactor*>& cur_chain = chain[k];
        const std::vector<ChainDDVar*>& cur_out_var = out_var[k];
        const ChainDDVar* cur_in_var = in_var[k];

        uint chain_length = cur_chain.size();

        Math1D::NamedVector<double> forward1(MAKENAME(forward1));
        Math1D::NamedVector<double> forward2(cur_in_var->nLabels(),MAKENAME(forward2));
        for (uint l=0; l < forward2.size(); l++)
          forward2[l] = cur_in_var->cost()[l];

        NamedStorage1D<Math2D::Matrix<uint> > trace(chain_length,MAKENAME(trace));

        //std::cerr << "start forward" << std::endl;

        //std::cerr << "chain: " << chain << std::endl;
        //std::cerr << "chain length: " << chain_length << std::endl;

        //compute forward
        cur_bound += cur_chain[0]->compute_forward(cur_in_var,cur_out_var[0],forward2,forward1,trace[0]);

        for (uint k=1; k < chain_length; k++) {

          //std::cerr << "k: " << k << std::endl;

          Math1D::Vector<double>& last_forward = ((k % 2) == 1) ? forward1 : forward2;
          Math1D::Vector<double>& new_forward = ((k % 2) == 0) ? forward1 : forward2;

          cur_bound += cur_chain[k]->compute_forward(cur_out_var[k-1],cur_out_var[k],last_forward,new_forward,trace[k]);

          // if (nLongChainsProcessed == 1 && k <= 10) {
          //   std::cerr << "k: " << k << std::endl;
          //   std::cerr << "forward: " << new_forward << std::endl;
          //   std::cerr << "in var involved in " << out_var[k-1]->nChains() << " chains" << std::endl;
          //   exit(0);
          // }
        }

        //std::cerr << "start traceback" << std::endl;

        //traceback
        Math1D::Vector<double>& total_forward = ((chain_length-1) % 2 == 0) ? forward1 : forward2;

        assert(total_forward.size() == cur_out_var[chain_length-1]->nLabels());

        // if (chain.size() > 1 && nLongChainsProcessed <= 10) {

        //   std::cerr << "long chain #" << nLongChainsProcessed << ", length " << chain_length << std::endl;
        //   std::cerr << "total forward: " << total_forward << std::endl;
        // }


        double best = 1e300;
        uint arg_best = MAX_UINT;
        for (uint l=0; l < total_forward.size(); l++) {
          if (total_forward[l] < best) {

            best = total_forward[l];
            arg_best = l;
          }
        }

        assert(arg_best < MAX_UINT);

        //std::cerr << "adding " << best << std::endl;
        cur_bound += best;

        for (int k=chain_length-1; k >= 0; k--) {

          //std::cerr << "k: " << k << std::endl;

          Math1D::Vector<uint>& labeling = factor_label[factor_num[cur_chain[k]]];
#ifdef PRIMAL_DUAL_STEPSIZE
          const Storage1D<ChainDDVar*>& involved_vars = factor_[factor_num[cur_chain[k]]]->involved_vars();
#endif

          assert(labeling.size() == trace[k].xDim());

          for (uint v=0; v < trace[k].xDim(); v++) {
            labeling[v] = trace[k](v,arg_best);
#ifdef PRIMAL_DUAL_STEPSIZE
            labeling_[var_num[involved_vars[v]]] = labeling[v];
#endif
          }

          //update arg_best
          if (k > 0) {
            for (uint v=0; v < trace[k].xDim(); v++) {

              if (cur_chain[k]->involved_vars()[v] == cur_out_var[k-1]) {
                arg_best = labeling[v];
                break;
              }
            }
          }
        }
      }
    }

    if (cur_bound > best_dual) {
      best_dual = cur_bound;
      delta *= 1.5;
    }
    else {
      nIncreases++;
      delta *= 0.95;
    }

    if (!quiet)
      std::cerr << "cur bound: " << cur_bound << ", best ever: " << best_dual << std::endl;

    //TRIAL: primal-dual bound
#ifdef PRIMAL_DUAL_STEPSIZE
    double cur_primal = 0.0;
    for (uint v=0; v < nUsedVars_; v++)
      cur_primal += var_[v]->cost()[labeling_[v]] * var_[v]->nChains(); //the cost are scaled internally

    for (uint f=0; f < nUsedFactors_; f++) {

      const Storage1D<ChainDDVar*>& involved_vars = factor_[f]->involved_vars();

      Math1D::Vector<uint> labeling(involved_vars.size());
      for (uint v=0; v < involved_vars.size(); v++)
        labeling[v] = labeling_[var_num[involved_vars[v]]];

      cur_primal += factor_[f]->cost(labeling);
    }

    best_primal = std::min(best_primal,cur_primal);
    std::cerr << "cur primal: " << cur_primal << ", best primal: " << best_primal << std::endl;

    assert(best_primal >= best_dual - 1e-5);

    Storage1D<Math1D::Vector<double> > gradient(nUsedVars_);
    for (uint v=0; v < nUsedVars_; v++)
      gradient[v].resize(var_[v]->cost().size(),0.0);

    for (uint f=0; f < nUsedFactors_; f++) {

      for (uint k=0; k < factor_label[f].size(); k++) {

        const ChainDDVar* cur_var = factor_[f]->involved_vars()[k];

        uint cur_fac_label = factor_label[f][k];
        gradient[var_num[cur_var]][cur_fac_label] += 1.0;
      }
    }
    double grad_sqr_norm = 0.0;
    for (uint v=0; v < nUsedVars_; v++)
      grad_sqr_norm += gradient[v].sqr_norm();

    step_size = start_step_size * (best_primal - cur_bound) / std::max(1.0,grad_sqr_norm);
#endif
    //END_TRIAL


    //TRIAL
    //like in the Torressani et al. paper
    //step_size = (best_dual + delta - cur_bound) / denom;
    //step_size = std::min(step_size,start_step_size);
    //std::cerr << "step size: " << step_size << std::endl;
    //END_TRIAL

    if (!projective) {

      for (uint f=0; f < nUsedFactors_; f++) {

        for (uint k=0; k < factor_label[f].size(); k++) {

          const ChainDDVar* cur_var = factor_[f]->involved_vars()[k];

          uint cur_fac_label = factor_label[f][k];
          uint cur_var_label = var_label[var_num[cur_var]];

          if (cur_fac_label != cur_var_label) {

            nDisagreements++;

            factor_[f]->dual_vars(cur_var)[cur_var_label] += step_size;
            factor_[f]->dual_vars(cur_var)[cur_fac_label] -= step_size;
          }
        }
      }


      if (!quiet) {
        std::cerr << nDisagreements << " disagreements" << std::endl;
        //std::cerr << "total repar cost: " << total_repar << std::endl;
      }
    }
    else {

      for (uint f=0; f < nUsedFactors_; f++) {

        for (uint k=0; k < factor_label[f].size(); k++) {

          const ChainDDVar* cur_var = factor_[f]->involved_vars()[k];

          uint cur_fac_label = factor_label[f][k];
          factor_[f]->dual_vars(cur_var)[cur_fac_label] -= step_size;
          assert(!isinf(factor_[f]->dual_vars(cur_var)[cur_fac_label]));
        }
      }

      //re-project
      for (uint v=0; v < nUsedVars_; v++) {

        ChainDDVar* cur_var = var_[v];

        const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

        if (cur_factor_list.size() > 0) {

          Math1D::Vector<double> sum(cur_var->nLabels(),0.0);

          for (uint k=0; k < cur_factor_list.size(); k++) {
            sum += cur_factor_list[k]->dual_vars(var_[v]);
          }

          sum *= 1.0 / cur_factor_list.size();

          for (uint k=0; k < cur_factor_list.size(); k++) {

            cur_factor_list[k]->dual_vars(var_[v]) -= sum;
            for (uint l=0; l < cur_var->nLabels(); l++)
              assert(!isinf(cur_factor_list[k]->dual_vars(var_[v])[l]));
          }
        }
      }
    }

    //of << (effort_per_iter * iter) << " " << best_dual << std::endl;
  }

  if (projective) {

    labeling_.set_constant(MAX_UINT);

    for (uint f=0; f < nUsedFactors_; f++) {

      for (uint k=0; k < factor_label[f].size(); k++) {

        const ChainDDVar* cur_var = factor_[f]->involved_vars()[k];

        uint vnum = var_num[cur_var];

        if (labeling_[vnum] == MAX_UINT)
          labeling_[vnum] = factor_label[f][k];
      }
    }
  }
  else
    labeling_ = var_label;

  size_t message_effort = 0;

  for (uint f=0; f < nUsedFactors_; f++) {

    uint size = factor_[f]->involved_vars().size();

    message_effort += size;
  }

  message_effort *= nIter;
  if (!quiet)
    std::cerr << "message effort: " << message_effort << std::endl;

  return best_dual;
}

void FactorChainDualDecomposition::compute_chain_sizes(const std::vector<std::vector<ChainDDFactor*> >& chain,
    std::vector<long double>& chain_size,
    std::vector<double>& log_chain_size)
{
  chain_size.clear();
  log_chain_size.clear();

  for (uint c=0; c < chain.size(); c++) {

    const std::vector<ChainDDFactor*>& cur_chain = chain[c];

    std::set<ChainDDVar*> vars;

    for (uint f=0; f < cur_chain.size(); f++) {

      const ChainDDFactor* cur_fac = cur_chain[f];
      const Storage1D<ChainDDVar*>& involved_vars = cur_fac->involved_vars();
      for (uint v=0; v < involved_vars.size(); v++)
        vars.insert(involved_vars[v]);
    }

    long double cur_size = 1.0;
    double cur_log_size = 0.0;

    for (std::set<ChainDDVar*>::const_iterator it = vars.begin(); it != vars.end(); it++) {
      cur_size *= (*it)->nLabels();
      cur_log_size += (*it)->nLabels();
    }

    chain_size.push_back(cur_size);
    log_chain_size.push_back(cur_log_size);
  }

}


double FactorChainDualDecomposition::smooth_optimization(uint nIter, double epsilon)
{
  std::cerr << "smoothed optimization [Jojic, Gould, Koller '10]" << std::endl;

  std::cerr << "WARNING: this is numerically heavily unstable, use log-space routine instead!!!" << std::endl;

  if (!optimize_called_) {
    set_up_chains();

    for (uint v=0; v < nUsedVars_; v++)
      var_[v]->set_up_chains();
  }

  //DEBUG
  set_up_singleton_chains();
  //END_DEBUG

  optimize_called_ = true;

  std::map<const ChainDDFactor*,uint> factor_num;
  for (uint f=0; f < nUsedFactors_; f++)
    factor_num[factor_[f]] = f;


  /**** store chains for easy access ****/

  std::vector<std::vector<ChainDDFactor*> > chain;
  std::vector<std::vector<ChainDDVar*> > out_var;
  std::vector<ChainDDVar*> in_var;
  std::vector<long double> chain_size;
  std::vector<double> log_chain_size;

  extract_chains(chain, out_var, in_var);

  compute_chain_sizes(chain, chain_size, log_chain_size);

  double mu_denom = 0.0;
  for (uint c=0; c < chain_size.size(); c++)
    mu_denom += log_chain_size[c];

  double mu = epsilon / (2.0*mu_denom);

  //DEBUG
  //mu = 1.0;
  //END_DEBUG

  std::cerr << "mu: " << mu << std::endl;
  std::cerr << "margin: " << mu* mu_denom << std::endl;

  //std::cerr << "chain size: " << chain_size << std::endl;

  //double lipschitz_constant = 1.0 / mu;
  double inv_lipschitz_constant = mu;

  //EXPLANATION: to calculate the gradient (=marginal), we need the auxiliary variables, not the primary ones.
  // Hence, the dual variables of the factors will be the auxiliary variables, and the real dual vars are stored externally

  Storage1D<Storage1D<Math1D::Vector<double> > > real_dual_var(nUsedFactors_);

  for (uint f=0; f < nUsedFactors_; f++) {
    real_dual_var[f].resize(factor_[f]->involved_vars().size());

    for (uint v=0; v < factor_[f]->involved_vars().size(); v++)
      real_dual_var[f][v].resize(factor_[f]->involved_vars()[v]->nLabels(),0.0);
  }

  double prev_t = 1.0;

  double energy = 0.0;

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "******** smooth Nesterov-iteration #" << iter << std::endl;

    energy = 0.0;

    //DEBUG
    double bwd_energy = 0.0;
    //END_DEBUG

    //compute gradient on-the-fly and subtract from aux_dual_var

    for (uint c=0; c < chain.size(); c++) {

      //std::cerr << "c: " << c << std::endl;

      const std::vector<ChainDDFactor*>& cur_chain = chain[c];
      const std::vector<ChainDDVar*>& cur_out_var_list = out_var[c];

      //a) forward pass along the chain
      Storage1D<Math1D::Vector<double> > forward(cur_chain.size());

      long double fwd_fac = 1.0;

      //std::cerr << "forward" << std::endl;

      for (uint f=0; f < cur_chain.size(); f++) {

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_out_var = cur_out_var_list[f];

        double log_offs;

        if (f == 0) {

          Math1D::Vector<double> start_forward;

          log_offs = cur_factor->compute_sum_forward(0, cur_out_var, start_forward, forward[f],mu);
        }
        else {

          ChainDDVar* cur_in_var = cur_out_var_list[f-1];
          log_offs = cur_factor->compute_sum_forward(cur_in_var, cur_out_var, forward[f-1], forward[f],mu);
        }

        //keep the numbers inside double precision
        double fac = forward[f].max();
        assert(fac != 0.0);

        //std::cerr << "fac: " << fac << std::endl;

        forward[f] *= 1.0 / fac;

        fwd_fac *= fac;
        energy += std::log(fac) + log_offs;
      }

      //long double avg = fwd_fac * (forward[cur_chain.size()-1].sum() / chain_size[c]);
      //energy += std::log(avg);

      energy += std::log(forward[cur_chain.size()-1].sum()) - log_chain_size[c];

      assert(!isnan(energy));

      //std::cerr << "backward" << std::endl;

      //b) backward pass along the chain
      Storage1D<Math1D::Vector<double> > backward(cur_chain.size());

      //DEBUG
      long double bwd_fac = 1.0;
      //END_DEBUG

      for (int f=cur_chain.size()-1; f >= 0; f--) {

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_in_var = cur_out_var_list[f];

        ChainDDVar* cur_out_var = (f > 0) ? cur_out_var_list[f-1] : in_var[c];

        double log_offs;

        if (f == int(cur_chain.size()-1)) {

          Math1D::Vector<double> start_backward;
          log_offs = cur_factor->compute_sum_forward(0, cur_out_var, start_backward, backward[f], mu);
        }
        else {

          log_offs = cur_factor->compute_sum_forward(cur_in_var, cur_out_var, backward[f+1], backward[f], mu);
        }

        //keep the numbers inside double precision
        double fac = backward[f].max();
        assert(fac != 0.0);

        backward[f] *= 1.0 / fac;

        //DEBUG
        bwd_fac *= fac;
        bwd_energy += std::log(fac) + log_offs;
        //END_DEBUG
      }

      //DEBUG
      //long double bwd_avg = bwd_fac * (backward[0].sum() / chain_size[c]);
      //bwd_energy += std::log(bwd_avg);

      bwd_energy += std::log(backward[0].sum()) - log_chain_size[c];
      //END_DEBUG

      //std::cerr << "now add to gradient" << std::endl;

      //c) compute gradients (=marginals, see [Jojic et al. ICML '10])
      // and go in neg. gradient direction

      for (uint f=0; f < cur_chain.size(); f++) {

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_out_var = (f+1 == cur_chain.size()) ? 0 : cur_out_var_list[f];

        ChainDDVar* cur_in_var = (f == 0) ? 0 : cur_out_var_list[f-1];

        const Storage1D<ChainDDVar*>& vars = cur_factor->involved_vars();

        Storage1D<Math1D::Vector<double> > marginals(vars.size());

        //first compute all marginals, then go in that direction. Otherwise we would use the updated dual vars
        // to compute incorrect marginals

        for (uint v=0; v < vars.size(); v++) {

          //std::cerr << "v: " << v << std::endl;

          if (vars[v] != cur_out_var || f+1 == cur_chain.size()) {

            //std::cerr << "passed" << std::endl;

            Math1D::Vector<double> in_msg;
            if (f > 0)
              in_msg = forward[f-1];

            Math1D::Vector<double> out_msg;
            if (f+1 < cur_chain.size())
              out_msg = backward[f+1];

            //std::cerr << "calling comp. marg" << std::endl;

            cur_factor->compute_marginals(vars[v],cur_in_var,cur_out_var,in_msg,out_msg,mu,marginals[v]);

            assert(marginals[v].sum() >= 0.99 && marginals[v].sum() <= 1.01);

            //std::cerr << "back" << std::endl;
          }
        }

        for (uint v=0; v < vars.size(); v++) {

          //std::cerr << "v: " << v << std::endl;

          if (vars[v] != cur_out_var || f+1 == cur_chain.size()) {

            Math1D::Vector<double>& cur_aux_dual_var = cur_factor->dual_vars(v);

            for (uint l=0; l < marginals[v].size(); l++) {

              cur_aux_dual_var[l] -= inv_lipschitz_constant * marginals[v][l];
            }

            //std::cerr << "last line in the loop" << std::endl;
          }
        }
      }
    }

    double max_abs_dual = 0.0;

    //reproject aux_dual_var
    for (uint v=0; v < nUsedVars_; v++) {

      ChainDDVar* cur_var = var_[v];

      const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

      if (cur_factor_list.size() > 0) {

        Math1D::Vector<double> sum(cur_var->nLabels(),0.0);

        for (uint k=0; k < cur_factor_list.size(); k++) {
          ChainDDFactor* cur_factor = cur_factor_list[k];

          sum += cur_factor->dual_vars(cur_var);
        }

        sum *= 1.0 / cur_factor_list.size();

        for (uint k=0; k < cur_factor_list.size(); k++) {
          ChainDDFactor* cur_factor = cur_factor_list[k];

          cur_factor->dual_vars(cur_var) -= sum;

          max_abs_dual = std::max(max_abs_dual, cur_factor->dual_vars(cur_var).max_abs());
        }
      }
    }

    std::cerr << "max abs. dual: " << max_abs_dual << std::endl;

    energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    std::cerr << "energy: " << energy << std::endl;

    //DEBUG
    bwd_energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    std::cerr << "check: bwd energy: " << bwd_energy << std::endl;
    //END_DEBUG

    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t* prev_t));
    const double nesterov_fac = (prev_t - 1) / new_t;

    for (uint f=0; f < nUsedFactors_; f++) {

      for (uint v=0; v < factor_[f]->involved_vars().size(); v++) {

        Math1D::Vector<double>& duals = factor_[f]->dual_vars(v);

        for (uint l=0; l < duals.size(); l++) {
          //double old_aux = aux_dual_var[f][v][l];
          //aux_dual_var[f][v][l] = old_aux + nesterov_fac * (old_aux - duals[l]);
          //duals[l] = old_aux;

          double old_aux = duals[l];
          duals[l] = old_aux + nesterov_fac * (old_aux - real_dual_var[f][v][l]);
          real_dual_var[f][v][l] = old_aux;
        }
      }
    }

    prev_t = new_t;
  }

  return energy; //note: this value is lagging behind one iteration
}

double FactorChainDualDecomposition::smooth_optimization_logspace(uint nIter, double epsilon)
{
  std::cerr << "smoothed optimization log-space [Jojic, Gould, Koller '10]" << std::endl;

  if (!optimize_called_) {
    set_up_chains();

    for (uint v=0; v < nUsedVars_; v++)
      var_[v]->set_up_chains();
  }

  //DEBUG
  //set_up_singleton_chains();
  //END_DEBUG

  optimize_called_ = true;

  std::map<const ChainDDFactor*,uint> factor_num;
  for (uint f=0; f < nUsedFactors_; f++)
    factor_num[factor_[f]] = f;


  /**** store chains for easy access ****/

  std::vector<std::vector<ChainDDFactor*> > chain;
  std::vector<std::vector<ChainDDVar*> > out_var;
  std::vector<ChainDDVar*> in_var;
  std::vector<long double> chain_size;
  std::vector<double> log_chain_size;

  extract_chains(chain, out_var, in_var);

  compute_chain_sizes(chain, chain_size, log_chain_size);

  double mu_denom = 0.0;
  for (uint c=0; c < chain_size.size(); c++) {
    mu_denom += log_chain_size[c];
  }

  double mu = epsilon / (2.0*mu_denom);
  double margin = mu * mu_denom;

  const double energy_offs = -mu_denom;

  //DEBUG
  //mu = 1.0;
  //END_DEBUG

  std::cerr << "mu: " << mu << std::endl;
  std::cerr << "margin: " << mu* mu_denom << std::endl;

  //std::cerr << "chain size: " << chain_size << std::endl;

  //double lipschitz_constant = 1.0 / mu;
  double inv_lipschitz_constant = mu;

  //EXPLANATION: to calculate the gradient (=marginal), we need the auxiliary variables, not the primary ones.
  // Hence, the dual variables of the factors will be the auxiliary variables, and the real dual vars are stored externally

  Storage1D<Storage1D<Math1D::Vector<double> > > real_dual_var(nUsedFactors_);

  for (uint f=0; f < nUsedFactors_; f++) {
    real_dual_var[f].resize(factor_[f]->involved_vars().size());

    for (uint v=0; v < factor_[f]->involved_vars().size(); v++)
      real_dual_var[f][v].resize(factor_[f]->involved_vars()[v]->nLabels(),0.0);
  }

  double prev_t = 1.0;

  double last_energy = -1e300;
  double energy = 0.0;

  //theoretically we may use step sizes up to inv_lipschitz_constant, but in practice one should start with larger values
  double step_size = 5000.0 * inv_lipschitz_constant;

  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "******** smooth log-space iteration #" << iter << std::endl;

    energy = energy_offs;

    //DEBUG
    double bwd_energy = energy_offs;
    //END_DEBUG

    //compute gradient on-the-fly and subtract from aux_dual_var

    for (uint c=0; c < chain.size(); c++) {

      //std::cerr << "c: " << c << std::endl;

      const std::vector<ChainDDFactor*>& cur_chain = chain[c];
      const std::vector<ChainDDVar*>& cur_out_var_list = out_var[c];

      //a) forward pass along the chain
      Storage1D<Math1D::Vector<double> > log_forward(cur_chain.size());

      //std::cerr << "forward" << std::endl;

      for (uint f=0; f < cur_chain.size(); f++) { //the final forward is used only for the energy printout

        Math1D::Vector<double>& cur_log_forward = log_forward[f];

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        //std::cerr << "factor with " << cur_factor->involved_vars().size() << " variables" << std::endl;

        ChainDDVar* cur_out_var = cur_out_var_list[f];

        double log_offs;

        if (f == 0) {

          Math1D::Vector<double> start_log_forward;

          log_offs = cur_factor->compute_sum_forward_logspace(0, cur_out_var, start_log_forward, cur_log_forward,mu);
        }
        else {

          ChainDDVar* cur_in_var = cur_out_var_list[f-1];
          log_offs = cur_factor->compute_sum_forward_logspace(cur_in_var, cur_out_var, log_forward[f-1], cur_log_forward,mu);
        }

        //keep the numbers inside double precision
        double offs = cur_log_forward.max();
        for (uint k=0; k < cur_log_forward.size(); k++)
          cur_log_forward[k] -= offs;

        energy += offs + log_offs;
      }


      double sum = 0.0;
      for (uint k=0; k < log_forward[cur_chain.size()-1].size(); k++)
        sum += std::exp(log_forward[cur_chain.size()-1][k]);

      energy += std::log(sum);

      assert(!isnan(energy));

      //std::cerr << "backward" << std::endl;

      //b) backward pass along the chain
      Storage1D<Math1D::Vector<double> > log_backward(cur_chain.size());

      for (int f=cur_chain.size()-1; f >= 0; f--) { //the backward pass for f=0 is used only for the debug energy printout

        Math1D::Vector<double>& cur_log_backward = log_backward[f];

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_in_var = cur_out_var_list[f];

        ChainDDVar* cur_out_var = (f > 0) ? cur_out_var_list[f-1] : in_var[c];

        double log_offs;

        if (f == int(cur_chain.size()-1)) {

          Math1D::Vector<double> start_log_backward;
          log_offs = cur_factor->compute_sum_forward_logspace(0, cur_out_var, start_log_backward, cur_log_backward, mu);
        }
        else {

          log_offs = cur_factor->compute_sum_forward_logspace(cur_in_var, cur_out_var, log_backward[f+1], cur_log_backward, mu);
        }

        //keep the numbers inside double precision
        double offs = cur_log_backward.max();

        for (uint k=0; k < cur_log_backward.size(); k++)
          cur_log_backward[k] -= offs;


        //DEBUG
        bwd_energy += offs + log_offs;
        //END_DEBUG
      }


      //DEBUG
      double bwd_sum = 0.0;
      for (uint k=0; k < log_backward[0].size(); k++)
        bwd_sum += std::exp(log_backward[0][k]);

      bwd_energy += std::log(bwd_sum);

      double check_diff = energy - bwd_energy;
      //if (!(fabs(bwd_energy-energy) < 0.001)) {
      //std::cerr << "fwd: " << energy << ", bwd: " << bwd_energy << std::endl;
      //}
      assert(fabs(check_diff / energy) < 0.0001);
      //END_DEBUG


      //c) compute gradients (=marginals, see [Jojic et al. ICML '10])
      // and go in neg. gradient direction

      Math1D::Vector<double> stored_marginal;

      for (uint f=0; f < cur_chain.size(); f++) {

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_out_var = (f+1 == cur_chain.size()) ? 0 : cur_out_var_list[f];

        ChainDDVar* cur_in_var = (f == 0) ? 0 : cur_out_var_list[f-1];

        const Storage1D<ChainDDVar*>& vars = cur_factor->involved_vars();

        Storage1D<Math1D::Vector<double> > marginals(vars.size());

        //first compute all marginals, then go in that direction. Otherwise we would use the updated dual vars
        // to compute incorrect marginals

        Math1D::Vector<double> in_log_msg;
        if (f > 0)
          in_log_msg = log_forward[f-1];

        Math1D::Vector<double> out_log_msg;
        if (f+1 < cur_chain.size())
          out_log_msg = log_backward[f+1];

        bool is_bilp = (dynamic_cast<AllPosBILPChainDDFactor*>(cur_factor) != 0)
                       || (dynamic_cast<BILPChainDDFactor*>(cur_factor) != 0);

        if (is_bilp || f+1 == cur_chain.size()) {

          if (f+1 == cur_chain.size()) {
            assert(cur_out_var == 0);
          }

          cur_factor->compute_all_marginals_logspace(cur_in_var,cur_out_var,in_log_msg,out_log_msg,mu,marginals);
        }
        else {

          //for variables that for chain links we still need the marginals for both involved factors as we need the derivatives
          // for both involved sets of dual vars. The two marginals are the same, though.

          for (uint v=0; v < vars.size(); v++) {

            //std::cerr << "v: " << v << std::endl;

            if (f == 0 || vars[v] != cur_in_var) {
              //if (true) {

              cur_factor->compute_marginals_logspace(vars[v],cur_in_var,cur_out_var,in_log_msg,out_log_msg,mu,marginals[v]);
            }
            else {
              marginals[v] = stored_marginal;
            }
          }
        }

        for (uint v=0; v < vars.size(); v++) {

          //std::cerr << "v: " << v << std::endl;

          const Math1D::Vector<double>& cur_marginal = marginals[v];
          Math1D::Vector<double>& cur_aux_dual_var = cur_factor->dual_vars(v);

          for (uint l=0; l < cur_marginal.size(); l++) {

            cur_aux_dual_var[l] -= step_size * cur_marginal[l];
          }

          if (vars[v] == cur_out_var)
            stored_marginal = cur_marginal;
        }
      }
    }

    double max_abs_dual = 0.0;

    //reproject aux_dual_var
    for (uint v=0; v < nUsedVars_; v++) {

      ChainDDVar* cur_var = var_[v];

      const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

      if (cur_factor_list.size() > 0) {

        Math1D::Vector<double> sum(cur_var->nLabels(),0.0);

        for (uint k=0; k < cur_factor_list.size(); k++) {
          ChainDDFactor* cur_factor = cur_factor_list[k];

          sum += cur_factor->dual_vars(cur_var);
        }

        sum *= 1.0 / cur_factor_list.size();

        for (uint k=0; k < cur_factor_list.size(); k++) {
          ChainDDFactor* cur_factor = cur_factor_list[k];

          cur_factor->dual_vars(cur_var) -= sum;

          max_abs_dual = std::max(max_abs_dual, cur_factor->dual_vars(cur_var).max_abs());
        }
      }
    }

    std::cerr << "max abs. dual: " << max_abs_dual << std::endl;

    energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    std::cerr << "energy: " << energy << std::endl;

    //DEBUG
    bwd_energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    std::cerr << "check: bwd energy: " << bwd_energy << std::endl;
    //END_DEBUG

    if (energy < last_energy && step_size > inv_lipschitz_constant) {

      std::cerr << "RESTART" << std::endl;
      step_size = std::max(0.25*step_size,inv_lipschitz_constant);
      prev_t = 1.0;
    }

    last_energy = energy;


    const double new_t = 0.5 * (1 + sqrt(1+4*prev_t* prev_t));
    const double nesterov_fac = (prev_t - 1.0) / new_t;

    for (uint f=0; f < nUsedFactors_; f++) {

      for (uint v=0; v < factor_[f]->involved_vars().size(); v++) {

        Math1D::Vector<double>& duals = factor_[f]->dual_vars(v);
        Math1D::Vector<double>& real_duals = real_dual_var[f][v];

        for (uint l=0; l < duals.size(); l++) {
          //double old_aux = aux_dual_var[f][v][l];
          //aux_dual_var[f][v][l] = old_aux + nesterov_fac * (old_aux - duals[l]);
          //duals[l] = old_aux;

          double old_aux = duals[l];
          duals[l] = old_aux + nesterov_fac * (old_aux - real_duals[l]);
          real_duals[l] = old_aux;
        }
      }
    }

    prev_t = new_t;
  }

  std::cerr << "final energy: " << energy << ", with margin: " << (energy-margin) << std::endl;

  labeling_.resize(nUsedVars_,0);
  for (uint v=0; v < nUsedVars_; v++) {
    var_[v]->dual_value(labeling_[v]);
  }

  return energy;
}

void FactorChainDualDecomposition::multiply_with_lbfgs_matrix(const Storage1D< Storage1D<Storage1D<Math1D::Vector<double> > > >& step,
    const Storage1D< Storage1D<Storage1D<Math1D::Vector<double> > > >& grad_diff,
    const Math1D::Vector<double>& rho, double initial_scale,
    int iter, int last_restart, Storage1D<Storage1D<Math1D::Vector<double> > >& vec) const
{
  const int L = step.size();

  Math1D::Vector<double> alpha(L);

  /** 1 forward multiplications **/
  for (int i=iter-1; i >= std::max<int>(last_restart,int(iter)-L); i--) {

    const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_step = step[i % L];

    double cur_alpha = 0.0;
    for (uint f=0; f < cur_step.size(); f++) {

      const Storage1D<Math1D::Vector<double> >& fac_cur_step = cur_step[f];
      const Storage1D<Math1D::Vector<double> >& fac_vec = vec[f];

      for (uint v=0; v < fac_cur_step.size(); v++)
        cur_alpha += fac_cur_step[v] % fac_vec[v];
    }

    cur_alpha *= rho[i % L];

    const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_grad_diff = grad_diff[i % L];

    for (uint f=0; f < vec.size(); f++) {

      Storage1D<Math1D::Vector<double> >& fac_vec = vec[f];
      const Storage1D<Math1D::Vector<double> >& fac_grad_diff = cur_grad_diff[f];

      for (uint v=0; v < fac_grad_diff.size(); v++) {

        fac_vec[v].add_vector_multiple(fac_grad_diff[v],-cur_alpha);
      }
    }

    alpha[i % L] = cur_alpha;
  }

  /** 2 apply initial matrix **/

  for (uint f=0; f < vec.size(); f++) {

    Storage1D<Math1D::Vector<double> >& fac_vec = vec[f];

    for (uint v=0; v < fac_vec.size(); v++)
      fac_vec[v] *= initial_scale;
  }

  /** 3 backward multiplications **/
  for (int i=std::max<int>(last_restart,int(iter)-L); i < iter; i++) {

    const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_grad_diff = grad_diff[i % L];

    double beta = 0.0;
    for (uint f=0; f < vec.size(); f++) {

      const Storage1D<Math1D::Vector<double> >& fac_vec = vec[f];
      const Storage1D<Math1D::Vector<double> >& fac_grad_diff = cur_grad_diff[f];

      for (uint v=0; v < fac_vec.size(); v++)
        beta += fac_vec[v] % fac_grad_diff[v];
    }
    beta *= rho[i % L];

    double gamma = alpha[i % L] - beta;

    const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_step = step[i % L];

    for (uint f=0; f < vec.size(); f++) {

      Storage1D<Math1D::Vector<double> >& fac_vec = vec[f];
      const Storage1D<Math1D::Vector<double> >& fac_step = cur_step[f];

      for (uint v=0; v < fac_vec.size(); v++) {

        fac_vec[v].add_vector_multiple(fac_step[v],gamma);
      }
    }
  }
}


double FactorChainDualDecomposition::smooth_optimization_LBFGS(uint nIter, double epsilon, uint L, SDD_LBFGS_MODE mode)
{
  std::cerr << "smoothed LBFGS-optimization log-space [Jojic, Gould, Koller '10]" << std::endl;
  std::cerr << "mode: " << mode << std::endl;

  //NOTE: Applying L-BFGS is not straightforward: It's a CONSTRAINED problem, though only with (linear) EQUALITY constraints
  //In implicit mode the objective is altered by just subtracting the mean of the dual variables. It seems that this does not preserve
  // convexity. Note that after a restart the method is equivalent to a projected gradient iteration
  //   (this follows from the calculation of the gradient)

  if (!optimize_called_) {
    set_up_chains();

    for (uint v=0; v < nUsedVars_; v++)
      var_[v]->set_up_chains();
  }

  //DEBUG
  //set_up_singleton_chains();
  //END_DEBUG

  optimize_called_ = true;

  std::map<const ChainDDVar*,uint> var_num;
  if (mode == SDD_LBFGS_KKT_SCHUR) {
    for (uint v=0; v < nUsedVars_; v++)
      var_num[var_[v]] = v;
  }

  std::map<const ChainDDFactor*,uint> factor_num;
  for (uint f=0; f < nUsedFactors_; f++)
    factor_num[factor_[f]] = f;


  /**** store chains for easy access ****/

  std::vector<std::vector<ChainDDFactor*> > chain;
  std::vector<std::vector<ChainDDVar*> > out_var;
  std::vector<ChainDDVar*> in_var;
  std::vector<long double> chain_size;
  std::vector<double> log_chain_size;

  extract_chains(chain, out_var, in_var);

  compute_chain_sizes(chain, chain_size, log_chain_size);

  double mu_denom = 0.0;
  for (uint c=0; c < chain_size.size(); c++)
    mu_denom += log_chain_size[c];


  double mu = epsilon / (2.0*mu_denom);
  double margin = mu * mu_denom;

  const double energy_offset = mu_denom;

  //DEBUG
  //mu = 1.0;
  //END_DEBUG

  std::cerr << "mu: " << mu << std::endl;
  std::cerr << "margin: " << mu* mu_denom << std::endl;

  //std::cerr << "chain size: " << chain_size << std::endl;

  //double lipschitz_constant = 1.0 / mu;
  double inv_lipschitz_constant = mu; //NOTE: for the modes implicit and elimination I'm not even sure if the function is convex


  /*** L-BFGS related variables ***/

  //these are the real variables, i.e. in implicit mode they do not satisfy the constraints.
  // in implicit and elimination mode the factors will only know the transformed variables
  // However, I think that for this specific function we could work without real_dual_var
  Storage1D<Storage1D<Math1D::Vector<double> > > real_dual_var(nUsedFactors_);

  for (uint f=0; f < nUsedFactors_; f++) {
    real_dual_var[f].resize(factor_[f]->involved_vars().size());

    for (uint v=0; v < factor_[f]->involved_vars().size(); v++)
      real_dual_var[f][v].resize(factor_[f]->involved_vars()[v]->nLabels(),0.0);
  }

  if (mode == SDD_LBFGS_ELIMINATION) {

    for (uint v=0; v < nUsedVars_; v++) {

      ChainDDVar* cur_var = var_[v];

      const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

      const uint cur_nFactors = cur_factor_list.size();
      if (cur_nFactors > 1) {

        ChainDDFactor* last_factor = cur_factor_list[cur_nFactors-1];
        uint last_fac_num = factor_num[last_factor];
        uint last_var_num = last_factor->var_idx(cur_var);


        //std::cerr << "resize(0)" << std::endl;
        real_dual_var[last_fac_num][last_var_num].resize(0);
      }
    }
  }

  //TODO: make sure that real_dual_var satisfies the constraints initially

  Storage1D<Storage1D<Math1D::Vector<double> > > neg_search_direction = real_dual_var;
  Storage1D<Storage1D<Math1D::Vector<double> > > gradient = real_dual_var;

  Storage1D< Storage1D<Storage1D<Math1D::Vector<double> > > > step(L);
  Storage1D< Storage1D<Storage1D<Math1D::Vector<double> > > > grad_diff(L);

  for (uint l=0; l < L; l++) {

    step[l] = real_dual_var;
    grad_diff[l] = real_dual_var;
  }

  Math1D::Vector<double> rho(L);

  Storage1D<Math1D::Vector<double> > lagrange_multiplier;
  if (mode == SDD_LBFGS_KKT_SCHUR) {
    lagrange_multiplier.resize(nUsedVars_);
    for (uint v=0; v < nUsedVars_; v++)
      lagrange_multiplier[v].resize(var_[v]->nLabels());
  }

  double energy = 0.0;

  //the function should be convex, but not strictly convex as we code the constraints implicitly => restarts may be necessary
  //CAUTION: I'm not sure if subtracting the mean preserves convexity. TODO: think this through
  uint last_restart = 1;
  for (uint iter = 1; iter <= nIter; iter++) {

    std::cerr << "******** smooth L-BFGS-iteration #" << iter << std::endl;

    energy = energy_offset;

    //DEBUG
    double bwd_energy = energy_offset;
    //END_DEBUG

    /**** 1. compute current energy and the gradient ****/

    /** 1a. process all chains **/
    for (uint c=0; c < chain.size(); c++) {

      //std::cerr << "c: " << c << std::endl;

      const std::vector<ChainDDFactor*>& cur_chain = chain[c];
      const std::vector<ChainDDVar*>& cur_out_var_list = out_var[c];

      //a) forward pass along the chain
      Storage1D<Math1D::Vector<double> > log_forward(cur_chain.size());

      //std::cerr << "forward" << std::endl;

      for (uint f=0; f < cur_chain.size(); f++) { //the final forward is used only for the energy printout

        Math1D::Vector<double>& cur_log_forward = log_forward[f];

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_out_var = cur_out_var_list[f];

        double log_offs;

        if (f == 0) {

          Math1D::Vector<double> start_log_forward;

          log_offs = cur_factor->compute_sum_forward_logspace(0, cur_out_var, start_log_forward, cur_log_forward,mu);
        }
        else {

          ChainDDVar* cur_in_var = cur_out_var_list[f-1];
          log_offs = cur_factor->compute_sum_forward_logspace(cur_in_var, cur_out_var, log_forward[f-1], cur_log_forward,mu);
        }

        //keep the numbers inside double precision
        double offs = cur_log_forward.max();
        for (uint k=0; k < cur_log_forward.size(); k++)
          cur_log_forward[k] -= offs;

        energy += offs + log_offs;
      }


      double sum = 0.0;
      for (uint k=0; k < log_forward[cur_chain.size()-1].size(); k++)
        sum += std::exp(log_forward[cur_chain.size()-1][k]);

      energy += std::log(sum);

      assert(!isnan(energy));

      //std::cerr << "backward" << std::endl;

      //b) backward pass along the chain
      Storage1D<Math1D::Vector<double> > log_backward(cur_chain.size());

      for (int f=cur_chain.size()-1; f >= 0; f--) { //the backward pass for f=0 is used only for the debug energy printout

        Math1D::Vector<double>& cur_log_backward = log_backward[f];

        ChainDDFactor* cur_factor = cur_chain[f];

        ChainDDVar* cur_in_var = cur_out_var_list[f];

        ChainDDVar* cur_out_var = (f > 0) ? cur_out_var_list[f-1] : in_var[c];

        double log_offs;

        if (f == int(cur_chain.size()-1)) {

          Math1D::Vector<double> start_backward;
          log_offs = cur_factor->compute_sum_forward_logspace(0, cur_out_var, start_backward, cur_log_backward, mu);
        }
        else {

          log_offs = cur_factor->compute_sum_forward_logspace(cur_in_var, cur_out_var, log_backward[f+1], cur_log_backward, mu);
        }

        //keep the numbers inside double precision
        double offs = cur_log_backward.max();

        for (uint k=0; k < cur_log_backward.size(); k++)
          cur_log_backward[k] -= offs;


        //DEBUG
        bwd_energy += offs + log_offs;
        //END_DEBUG
      }


      //DEBUG
      double bwd_sum = 0.0;
      for (uint k=0; k < log_backward[0].size(); k++)
        bwd_sum += std::exp(log_backward[0][k]);

      bwd_energy += std::log(bwd_sum);
      //END_DEBUG

      //c) compute part I of the gradients , i.e. the marginals (see [Jojic et al. ICML '10])

      for (uint f=0; f < cur_chain.size(); f++) {

        //std::cerr << "f: " << f << "/" << cur_chain.size() << std::endl;

        ChainDDFactor* cur_factor = cur_chain[f];

        const uint fac_num = factor_num[cur_factor];

        Storage1D<Math1D::Vector<double> >& cur_gradient = gradient[fac_num];

        ChainDDVar* cur_out_var = (f+1 == cur_chain.size()) ? 0 : cur_out_var_list[f];

        ChainDDVar* cur_in_var = (f == 0) ? 0 : cur_out_var_list[f-1];

        const Storage1D<ChainDDVar*>& vars = cur_factor->involved_vars();

        //first compute all marginals, then go in that direction. Otherwise we would use the updated dual vars
        // to compute incorrect marginals

        Math1D::Vector<double> in_log_msg;
        if (f > 0)
          in_log_msg = log_forward[f-1];

        Math1D::Vector<double> out_log_msg;
        if (f+1 < cur_chain.size())
          out_log_msg = log_backward[f+1];

        bool is_bilp = (dynamic_cast<AllPosBILPChainDDFactor*>(cur_factor) != 0)
                       || (dynamic_cast<BILPChainDDFactor*>(cur_factor) != 0);

        if (is_bilp || f+1 == cur_chain.size()) {

          if (f+1 == cur_chain.size()) {
            assert(cur_out_var == 0);
          }

          cur_factor->compute_all_marginals_logspace(cur_in_var,cur_out_var,in_log_msg,out_log_msg,mu,cur_gradient);
        }
        else {

          //for variables that for chain links we still need the marginals for both involved factors as we need the derivatives
          // for both involved sets of dual vars. The two marginals are the same, though.

          for (uint v=0; v < vars.size(); v++) {

            //std::cerr << "v: " << v << std::endl;

            if (f == 0 || vars[v] != cur_in_var) {

              cur_factor->compute_marginals_logspace(vars[v],cur_in_var,cur_out_var,in_log_msg,out_log_msg,mu,cur_gradient[v]);
            }
            else {
              const uint prev_fac_num = factor_num[cur_chain[f-1]];
              cur_gradient[v] = gradient[prev_fac_num][cur_chain[f-1]->var_idx(cur_in_var)];
            }
          }
        }
      }
    }

    energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    //DEBUG
    bwd_energy *= -mu; //minus since we have internally converted our maxi-min problem to a mini-max one
    std::cerr << "energy: " << energy << ", bwd: " << bwd_energy << std::endl;
    //END_DEBUG

    /** 1b finish the computation of the gradients **/
    if (mode == SDD_LBFGS_IMPLICIT_CONSTRAINTS) {
      for (uint v=0; v < nUsedVars_; v++) {

        ChainDDVar* cur_var = var_[v];

        const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

        const uint cur_nFactors = cur_factor_list.size();
        if (cur_nFactors > 0) {

          Math1D::Vector<double> grad_avg(cur_var->nLabels(),0.0);

          for (uint k=0; k < cur_nFactors; k++) {
            ChainDDFactor* cur_factor = cur_factor_list[k];
            uint fac_num = factor_num[cur_factor];
            uint var_num = cur_factor->var_idx(cur_var);

            grad_avg += gradient[fac_num][var_num];
          }

          grad_avg *= 1.0 / cur_nFactors;

          for (uint k=0; k < cur_nFactors; k++) {
            ChainDDFactor* cur_factor = cur_factor_list[k];
            uint fac_num = factor_num[cur_factor];
            uint var_num = cur_factor->var_idx(cur_var);

            gradient[fac_num][var_num] -= grad_avg;
          }
        }
      }
    }
    else if (mode == SDD_LBFGS_ELIMINATION) {

      for (uint v=0; v < nUsedVars_; v++) {

        ChainDDVar* cur_var = var_[v];

        const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

        const uint cur_nFactors = cur_factor_list.size();
        if (cur_nFactors > 1) {

          ChainDDFactor* last_factor = cur_factor_list[cur_nFactors-1];
          uint last_fac_num = factor_num[last_factor];
          uint last_var_num = last_factor->var_idx(cur_var);

          const Math1D::Vector<double>& val = gradient[last_fac_num][last_var_num];

          // std::cerr << "last factor has number " << last_fac_num
          //           << " and " << last_factor->involved_vars().size() << " variables" << std::endl;
          // std::cerr << "last var num: " << last_var_num << std::endl;
          // std::cerr << "val: " << val << std::endl;

          for (uint k=0; k < cur_nFactors-1; k++) {
            ChainDDFactor* cur_factor = cur_factor_list[k];
            uint fac_num = factor_num[cur_factor];
            uint var_num = cur_factor->var_idx(cur_var);

            gradient[fac_num][var_num] -= val;
          }

          //std::cerr << "resize(0)" << std::endl;
          gradient[last_fac_num][last_var_num].resize(0);
        }
      }
    }

    /** 1c compute squared gradient norm **/

    double sqr_grad_norm = 0.0;
    for (uint f=0; f < gradient.size(); f++) {
      const Storage1D<Math1D::Vector<double> >& fac_gradient = gradient[f];

      for (uint v=0; v < fac_gradient.size(); v++)
        sqr_grad_norm += fac_gradient[v].sqr_norm();
    }

    std::cerr << "squared gradient norm: " << sqr_grad_norm << std::endl;

    /**** 2. compute search direction ****/

    /** 2a finish setting the latest grad difference and compute the corresponding rho **/
    double latest_inv_rho = 0.0;

    if (iter > last_restart) {
      Storage1D<Storage1D<Math1D::Vector<double> > >& cur_grad_diff = grad_diff[(iter-1) % L];
      const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_step = step[(iter-1) % L];

      for (uint f=0; f < cur_grad_diff.size(); f++) {

        Storage1D<Math1D::Vector<double> >& cur_fac_grad_diff = cur_grad_diff[f];
        const Storage1D<Math1D::Vector<double> >& cur_fac_step = cur_step[f];
        const Storage1D<Math1D::Vector<double> >& fac_gradient = gradient[f];

        for (uint v=0; v < cur_fac_grad_diff.size(); v++) {

          Math1D::Vector<double>& cur_var_grad_diff = cur_fac_grad_diff[v];

          cur_var_grad_diff *= -1.0; //was set to the previous gradient in the last loop iteration
          cur_var_grad_diff += fac_gradient[v];

          latest_inv_rho += cur_var_grad_diff % cur_fac_step[v];
        }
      }

      //std::cerr << "new inv rho: " <<  latest_inv_rho << std::endl;

      rho[(iter-1) % L] = 1.0 / latest_inv_rho;
      if (latest_inv_rho < 1e-305) {

        std::cerr << "RESTART, inv_rho = " << latest_inv_rho << std::endl;
        last_restart = iter;
      }
    }


    //if we fix the step size anyway, setting scale = 1.0 should suffice
    double scale;

    if (iter > last_restart) {

      const Storage1D<Storage1D<Math1D::Vector<double> > >& cur_grad_diff = grad_diff[(iter-1) % L];
      double sqr_norm = 0.0;

      for (uint f=0; f < cur_grad_diff.size(); f++) {

        const Storage1D<Math1D::Vector<double> >& cur_fac_grad_diff = cur_grad_diff[f];

        for (uint v=0; v < cur_fac_grad_diff.size(); v++) {
          sqr_norm += cur_fac_grad_diff[v].sqr_norm();
        }
      }

      scale = latest_inv_rho / sqr_norm;

      //TRIAL
      scale *= 2.0; //should not be much higher than 2.0 (experimental result, 2.5 performed poorly)
      //END_TRIAL
    }
    else
      scale = 5000.0 * inv_lipschitz_constant;

    //std::cerr << "scale: " << scale << std::endl;

    /** 2b now apply the L-BFGS formula ***/

    if (mode != SDD_LBFGS_KKT_SCHUR) {


      neg_search_direction = gradient;
      multiply_with_lbfgs_matrix(step, grad_diff, rho, scale, iter, last_restart, neg_search_direction);
      //std::cerr << "multiplication done" << std::endl;
    }
    else {

      //KKT-SCHUR mode
      // for an explanation of how this works: this is based on the book  [Nocedal&Wright, 2nd edition]
      //  the model of each iteration is as in (18.7) and (16.3), its solution is as in (16.13) and (16.14)
      //   thanks to L-BFGS we have direct access to the inverse matrix.
      // Note: this approach is possible because we have only equality constraints and there are no redundant constraints

      /**** 1. solve for the optimal multipliers (after applying the Schur complement). This is a PD system, we use CG ****/

      Storage1D<Storage1D<Math1D::Vector<double> > > aux; //use neg_search_direction for this??

      //for now, we only consider starting with all multipliers 0
      for (uint v=0; v < nUsedVars_; v++)
        lagrange_multiplier[v].set_constant(0.0);

      //the starting point ensures that this is initialized with 0
      Storage1D<Math1D::Vector<double> > cg_residual = lagrange_multiplier;

      //compute first residual (with this starting point: simply the negative right hand side)
      aux = gradient;
      multiply_with_lbfgs_matrix(step, grad_diff, rho, scale, iter, last_restart, aux);
      for (uint v=0; v < nUsedVars_; v++) {

        Math1D::Vector<double>& cur_residual = cg_residual[v];

        ChainDDVar* cur_var = var_[v];
        const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

        const uint cur_nFactors = cur_factor_list.size();
        for (uint f=0; f < cur_nFactors; f++) {

          const ChainDDFactor* cur_factor = cur_factor_list[f];
          const uint fac_num = factor_num[cur_factor];
          const uint var_num = cur_factor->var_idx(cur_var);

          cur_residual -= cur_factor->dual_vars(var_num);
          cur_residual += aux[fac_num][var_num];
        }
      }

      //init direction
      Storage1D<Math1D::Vector<double> > cg_direction = cg_residual;
      Storage1D<Math1D::Vector<double> > cg_aux_direction = cg_residual; //used to store A*B⁻1*A^T *cg_direction
      for (uint v=0; v < nUsedVars_; v++)
        negate(cg_direction[v]);

      double sqr_res_norm = 0.0;
      for (uint v=0; v < nUsedVars_; v++)
        sqr_res_norm += cg_residual[v].sqr_norm();


      for (uint cg_iter=1; cg_iter <= 1000; cg_iter++) {

        std::cerr << "##cg-iter #" << cg_iter << ", sqr res norm: " << sqr_res_norm << std::endl;

        if (sqr_res_norm < 0.01)
          break;

        /** set cg_aux_direction to A*B⁻1*A^T *cg_direction **/

        //a) aux = A^T*cg_direction
        for (uint v=0; v < nUsedVars_; v++) {

          const Math1D::Vector<double>& cur_dir = cg_direction[v];

          const ChainDDVar* cur_var = var_[v];
          const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

          const uint cur_nFactors = cur_factor_list.size();
          for (uint f=0; f < cur_nFactors; f++) {

            const ChainDDFactor* cur_factor = cur_factor_list[f];
            const uint fac_num = factor_num[cur_factor];
            const uint var_num = cur_factor->var_idx(cur_var);

            aux[fac_num][var_num] = cur_dir;
          }
        }


        //b) aux = B⁻1*aux
        multiply_with_lbfgs_matrix(step, grad_diff, rho, scale, iter, last_restart, aux);

        //c) cg_aux_direction = A*aux
        for (uint v=0; v < nUsedVars_; v++)
          cg_aux_direction[v].set_constant(0.0);

        for (uint f=0; f < aux.size(); f++) {

          const Storage1D<Math1D::Vector<double> >& cur_aux = aux[f];

          const Storage1D<ChainDDVar*>& involved_vars = factor_[f]->involved_vars();
          for (uint k=0; k < involved_vars.size(); k++)
            cg_aux_direction[var_num[involved_vars[k]]] += cur_aux[k];
        }

        /** compute alpha **/
        double alpha_denom = 0.0;
        for (uint v=0; v < nUsedVars_; v++)
          alpha_denom += cg_direction[v] % cg_aux_direction[v];

        double alpha = sqr_res_norm / alpha_denom;

        /** update lagrange multipliers (= primary variables) **/
        for (uint v=0; v < nUsedVars_; v++)
          lagrange_multiplier[v].add_vector_multiple(cg_direction[v],alpha);

        /** update residual **/
        for (uint v=0; v < nUsedVars_; v++)
          cg_residual[v].add_vector_multiple(cg_aux_direction[v],alpha);

        /** compute new squared residual norm **/
        double new_sqr_res_norm = 0.0;
        for (uint v=0; v < nUsedVars_; v++)
          new_sqr_res_norm += cg_residual[v].sqr_norm();

        double beta = new_sqr_res_norm / sqr_res_norm;

        /** update search direction **/
        for (uint v=0; v < nUsedVars_; v++) {
          cg_direction[v] *= beta;
          cg_direction[v] -= cg_residual[v];
        }

        sqr_res_norm = new_sqr_res_norm;
      }


      /**** 2. now compute the negative search direction ****/
      neg_search_direction = gradient;
      //add A^T*lagrange_multipliers
      for (uint f=0; f < aux.size(); f++) {

        const Storage1D<ChainDDVar*>& involved_vars = factor_[f]->involved_vars();
        for (uint k=0; k < involved_vars.size(); k++)
          neg_search_direction[f][k] += lagrange_multiplier[var_num[involved_vars[k]]];
      }
      multiply_with_lbfgs_matrix(step, grad_diff, rho, scale, iter, last_restart, neg_search_direction);
    }

    /***** 3. update the point, the dual vars of the factors, record the latest step and prepare the latest grad diff. ****/
    double step_size = 1.0; //I am not aware of a guarantee that this step size cannot increase the energy - even in KKT mode.

    //in practice, calculating the optimal step size for the lipschitz-bound performs very poorly.
    // the resulting step sizes are often around 1e-5
    // double sp = 0.0;
    // double d_norm = 0.0;

    // for (uint f=0; f < gradient.size(); f++) {
    //   for (uint v=0; v < gradient[f].size(); v++) {
    //     sp += gradient[f][v] % neg_search_direction[f][v];
    //     d_norm += neg_search_direction[f][v].sqr_norm();
    //   }
    // }

    // step_size = inv_lipschitz_constant * sp / d_norm;

    // std::cerr << "step size: " << step_size << std::endl;

    Storage1D<Storage1D<Math1D::Vector<double> > >& new_step = step[iter % L];
    Storage1D<Storage1D<Math1D::Vector<double> > >& new_grad_diff = grad_diff[iter % L];

    for (uint f=0; f < real_dual_var.size(); f++) {

      Storage1D<Math1D::Vector<double> >& fac_new_step = new_step[f];
      Storage1D<Math1D::Vector<double> >& fac_dual_var = real_dual_var[f];
      Storage1D<Math1D::Vector<double> >& fac_new_grad_diff = new_grad_diff[f];
      const Storage1D<Math1D::Vector<double> >& fac_neg_search_dir = neg_search_direction[f];
      const Storage1D<Math1D::Vector<double> >& fac_gradient = gradient[f];

      for (uint v=0; v < fac_dual_var.size(); v++) {

        fac_new_step[v] = fac_neg_search_dir[v];
        fac_new_step[v] *= -step_size;
        fac_dual_var[v] += fac_new_step[v];
        fac_new_grad_diff[v] = fac_gradient[v]; //will be finished after point 1 in the next loop iter


        //for implicit constraints:
        //  since the gradients satisfy the constraints (and hence also the search direction as a linear transformation)
        //  there is no need to convert anything
        // for KKT_SCHUR the mulipliers ensure that we stay feasible
        // for elimination, will set the eliminated variables below
        if (fac_dual_var[v].size() > 0)
          factor_[f]->dual_vars(v) = fac_dual_var[v];
      }
    }

    if (mode == SDD_LBFGS_ELIMINATION) {

      //set eliminated dual variables in the respective factors

      for (uint v=0; v < nUsedVars_; v++) {

        ChainDDVar* cur_var = var_[v];

        const Storage1D<ChainDDFactor*>& cur_factor_list = cur_var->neighboring_factor();

        const uint cur_nFactors = cur_factor_list.size();
        if (cur_nFactors > 1) {

          Math1D::Vector<double> sum(cur_var->nLabels(),0.0);

          for (uint k=0; k < cur_nFactors-1; k++) {
            ChainDDFactor* cur_factor = cur_factor_list[k];
            uint fac_num = factor_num[cur_factor];
            uint var_num = cur_factor->var_idx(cur_var);

            sum += real_dual_var[fac_num][var_num];
          }

          negate(sum);

          ChainDDFactor* last_factor = cur_factor_list[cur_nFactors-1];
          last_factor->dual_vars(cur_var) = sum;
        }
      }
    }
  }

  std::cerr << "final energy: " << energy << ", with margin: " << (energy-margin) << std::endl;

  labeling_.resize(nUsedVars_,0);
  for (uint v=0; v < nUsedVars_; v++) {
    var_[v]->dual_value(labeling_[v]);
  }

  return energy;
}
