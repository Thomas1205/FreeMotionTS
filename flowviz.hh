#ifndef FLOWVIZ_HH
#define FLOWVIZ_HH

#include "colorimage.hh"

void visualize_flow(const Math3D::Tensor<double>& flow, double max_norm,
		    Math3D::ColorImage<float>& image, bool color_code_rim = false);


#endif
