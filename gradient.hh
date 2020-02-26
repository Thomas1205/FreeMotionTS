#ifndef GRADIENT_HH
#define GRADIENT_HH

#include "storage1D.hh"
#include "colorimage.hh"


void compute_channel_gradients(const Math3D::ColorImage<float>& image, 
                               Storage1D<Math3D::NamedTensor<float> >& gradient);

void compute_symmetric_channel_gradients(const Math3D::ColorImage<float>& image, 
                                         Storage1D<Math3D::NamedTensor<float> >& gradient);

void compute_channel_mean_gradients(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second, 
                                    Storage1D<Math3D::NamedTensor<float> >& gradient);

void compute_symmetric_channel_mean_gradients(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second, 
                                              Storage1D<Math3D::NamedTensor<float> >& gradient);


#endif
