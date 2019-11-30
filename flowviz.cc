#include "flowviz.hh"
#include "colorcode.h"

void visualize_flow(const Math3D::Tensor<double>& flow, double max_norm,
		    Math3D::ColorImage<float>& image, bool color_code_rim) {

  uint xDim = flow.xDim();
  uint yDim = flow.yDim();

  uchar rgb[3];

  if (flow.zDim() != 2) {

    INTERNAL_ERROR << "  attempt to interpret a tensor with " << flow.zDim() << " channels as a 2D-flow. Exiting."
		   << std::endl;
    exit(1);
  }

  uint h_addon = 0;
  uint v_addon = 0;

  if (color_code_rim) {

    //h_addon = (uint) (xDim*0.1);
    //v_addon = (uint) (yDim*0.1);
    
    h_addon = (uint) (std::min(xDim,yDim)*0.065);
    v_addon = h_addon;

    h_addon = std::max<uint>(3,h_addon);
    v_addon = std::max<uint>(3,v_addon);
  }

  uint viz_xDim = xDim + 2*h_addon;
  uint viz_yDim = yDim + 2*v_addon;

  image.resize(viz_xDim,viz_yDim,3);

  if (color_code_rim) {
    
    for (uint y=0; y < viz_yDim; y++) {
      for (uint x=0; x < viz_xDim; x++) {

	float fx = ((float) x) - 0.5 * ((float) viz_xDim);
	float fy = ((float) y) - 0.5 * ((float) viz_yDim);

	//std::cerr << "fx: " << fx << ", fy: " << fy << std::endl;

	float norm = sqrt(fx*fx+fy*fy) + 0.001;

	float scale = 1.0 / norm;

	computeColor(scale*fx,scale*fy,rgb);
	
	for (uint z=0; z < 3; z++)
	  image(x,y,z) = rgb[z];
      }
    }
  }

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      double fx = flow(x,y,0);
      double fy = flow(x,y,1);

      double scale = (max_norm > 0.0) ?  1.0 / max_norm : 0.0;

      computeColor(scale*fx,scale*fy,rgb);
      
      for (uint z=0; z < 3; z++)
	image(x+h_addon,y+v_addon,z) = rgb[z];
    }
  }
}
