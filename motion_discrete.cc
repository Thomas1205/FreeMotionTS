/**** written by Thomas Schoenemann as an employee of Lund University, Sweden, August 2010 ****/

#include "motion_discrete.hh"
#include "motion_moves.hh"
#include "tensor_interpolation.hh"

typedef float Real;

double expmove_motion_estimation(const Math3D::Tensor<float>& first, const Math3D::Tensor<float>& second,
                                 int min_x_disp, int max_x_disp, int min_y_disp, int max_y_disp, uint spacing,
                                 uint neighborhood, double lambda, Math3D::Tensor<double>& velocity,
                                 Math2D::Matrix<uint>* given_labeling) {

  const uint xDim = first.xDim();
  const uint yDim = first.yDim();
  const uint nChannels = first.zDim();

  Real inv_spacing = 1.0 / spacing;

  const uint nHorLabels = (max_x_disp - min_x_disp + 1) * spacing - (spacing - 1);
  const uint nVertLabels = (max_y_disp - min_y_disp +1) * spacing - (spacing - 1);
  const uint nLabels = nHorLabels * nVertLabels;

  const uint zero_label = nHorLabels* (  (-min_y_disp + 1)*spacing - (spacing-1)    )
    + (-min_x_disp + 1)*spacing - (spacing-1);

  Math3D::NamedTensor<Real> label_cost(xDim,yDim,nLabels,MAKENAME(label_cost));
  
  Math2D::Matrix<uint> new_labeling;
  if (given_labeling == 0)
    new_labeling.resize(xDim,yDim,zero_label);
  else
    given_labeling->set_constant(zero_label);
  Math2D::Matrix<uint>& labeling = (given_labeling != 0) ? *given_labeling : new_labeling;

  //DO NOT MULTIPLY: this is handled in the expansion moves alraedy (by considering the corresponding velocities)!!
  //lambda *= inv_spacing;

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      for (uint lx = 0; lx < nHorLabels; lx++) {
        for (uint ly = 0; ly < nVertLabels; ly++) {

          float u = ((int) lx) * inv_spacing + min_x_disp;
          float v = ((int) ly) * inv_spacing + min_y_disp;
	
          float tx = ((int) x) + u;
          float ty = ((int) y) + v;

          if (tx < 0)
            tx = 0;
          if (tx >= (int) xDim)
            tx = xDim-1;

          if (ty < 0)
            ty = 0;
          if (ty >= (int) yDim)
            ty = yDim-1;
	    
          Real disp_cost = 0.0;

          for (uint z=0; z < nChannels; z++) {
            //Real diff = first(x,y,z) - second(tx,ty,z);
            Real diff = first(x,y,z) - bilinear_interpolation(second, tx, ty, z);

            disp_cost += fabs(diff);
            //disp_cost += diff*diff;
          }

          label_cost(x,y,ly*nHorLabels+lx) = disp_cost;
        }
      }
    }
  }

  discrete_motion_opt(label_cost, nHorLabels, spacing, lambda, neighborhood, labeling);

  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {


      uint label = labeling(x,y);

      uint lx = label % nHorLabels;
      uint ly = label / nHorLabels;

      velocity(x,y,0) = ((int) lx) * inv_spacing + min_x_disp;
      velocity(x,y,1) = ((int) ly) * inv_spacing + min_y_disp;
    }
  }

  return motion_energy(label_cost, nHorLabels, spacing, lambda, 
                       neighborhood, labeling);
}
