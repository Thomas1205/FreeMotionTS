/**** written by Thomas Schoenemann as an employee of Lund University, February 2010 ****/

#ifndef FLOW_EVAL_HH
#define FLOW_EVAL_HH

#include "tensor.hh"

double average_angular_error(const Math3D::Tensor<double>& flow, const Math3D::Tensor<double>& ground_truth_flow);

#endif
