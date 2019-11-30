/**** written by Thomas Schoenemann as an employee of Lund University, February 2010 ****/

#include "flow_eval.hh"

double average_angular_error(const Math3D::Tensor<double>& flow, const Math3D::Tensor<double>& ground_truth_flow) {

  uint xDim = flow.xDim();
  uint yDim = flow.yDim();

  assert(xDim == ground_truth_flow.xDim());
  assert(yDim == ground_truth_flow.yDim());

  double sum_aae = 0.0;

  uint nNonOccluded = 0;
  
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      double u = flow(x,y,0);
      double v = flow(x,y,1);

      double gu = ground_truth_flow(x,y,0);
      double gv = ground_truth_flow(x,y,1);

      if (fabs(gu) < 1e9 && fabs(gv) < 1e9) {

	double arg = (u*gu + v*gv + 1) / sqrt( (u*u + v*v + 1) * (gu*gu + gv*gv + 1) );
	
	double degree = 180.0 * acos(arg) / M_PI;
	
	sum_aae += degree;
	nNonOccluded++;
      }
    }
  }

  return (sum_aae / (xDim*yDim));
  return (sum_aae / nNonOccluded);
}
