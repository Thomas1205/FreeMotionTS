#include "gradient.hh"

void compute_channel_gradients(const Math3D::ColorImage<float>& image, 
                               Storage1D<Math3D::NamedTensor<float> >& gradient) {


  uint xDim = uint( image.xDim() );
  uint yDim = uint( image.yDim() );
  uint zDim = uint( image.zDim() );

  assert(gradient.size() == zDim);
  for (uint z=0; z < zDim; z++) {
    assert(gradient[z].xDim() == xDim);
    assert(gradient[z].yDim() == yDim);
    assert(gradient[z].zDim() == 2);
  }

  for (uint z=0; z < zDim; z++) {
    for (uint y = 0; y < yDim; y++) {
      for (uint x = 0; x < xDim; x++) {

        float cur = image(x,y,z);
      
        gradient[z](x,y,0) = float( (x > 0) ? (cur - image(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) = float( (y > 0) ? (cur - image(x,y-1,z)) : 0.0 );
      }
    }
  }
}

void compute_symmetric_channel_gradients(const Math3D::ColorImage<float>& image, 
                                         Storage1D<Math3D::NamedTensor<float> >& gradient) {


  uint xDim = uint( image.xDim() );
  uint yDim = uint( image.yDim() );
  uint zDim = uint( image.zDim() );

  assert(gradient.size() == zDim);
  for (uint z=0; z < zDim; z++) {
    assert(gradient[z].xDim() == xDim);
    assert(gradient[z].yDim() == yDim);
    assert(gradient[z].zDim() == 2);
  }

  for (uint z=0; z < zDim; z++) {
    for (uint y = 0; y < yDim; y++) {
      for (uint x = 0; x < xDim; x++) {

        float cur_x = (x+1 < xDim) ? image(x+1,y,z) : image(x,y,z);
        float cur_y = (y+1 < yDim) ? image(x,y+1,z) : image(x,y,z);
      
        gradient[z](x,y,0) = 0.5 * float( (x > 0) ? (cur_x - image(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) = 0.5 * float( (y > 0) ? (cur_y - image(x,y-1,z)) : 0.0 );
      }
    }
  }
}


void compute_channel_mean_gradients(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second, 
                                    Storage1D<Math3D::NamedTensor<float> >& gradient) {

  uint xDim = uint( first.xDim() );
  uint yDim = uint( first.yDim() );
  uint zDim = uint( first.zDim() );

  assert(second.xDim() == xDim);
  assert(second.yDim() == yDim);
  assert(second.zDim() == zDim);

  assert(gradient.size() == zDim);
  for (uint z=0; z < zDim; z++) {
    assert(gradient[z].xDim() == xDim);
    assert(gradient[z].yDim() == yDim);
    assert(gradient[z].zDim() == 2);
  }

  for (uint z=0; z < zDim; z++) {
    for (uint y = 0; y < yDim; y++) {
      for (uint x = 0; x < xDim; x++) {

        float cur = first(x,y,z);
      
        gradient[z](x,y,0) = float( (x > 0) ? (cur - first(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) = float( (y > 0) ? (cur - first(x,y-1,z)) : 0.0 );

        cur = second(x,y,z);
        gradient[z](x,y,0) += float( (x > 0) ? (cur - second(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) += float( (y > 0) ? (cur - second(x,y-1,z)) : 0.0 );	
      }
    }
  
    gradient[z] *= 0.5f;
  }
}


void compute_symmetric_channel_mean_gradients(const Math3D::ColorImage<float>& first, const Math3D::ColorImage<float>& second, 
                                              Storage1D<Math3D::NamedTensor<float> >& gradient) {

  uint xDim = uint( first.xDim() );
  uint yDim = uint( first.yDim() );
  uint zDim = uint( first.zDim() );

  assert(second.xDim() == xDim);
  assert(second.yDim() == yDim);
  assert(second.zDim() == zDim);

  assert(gradient.size() == zDim);
  for (uint z=0; z < zDim; z++) {
    assert(gradient[z].xDim() == xDim);
    assert(gradient[z].yDim() == yDim);
    assert(gradient[z].zDim() == 2);
  }

  for (uint z=0; z < zDim; z++) {
    for (uint y = 0; y < yDim; y++) {
      for (uint x = 0; x < xDim; x++) {

        float cur_x = (x+1 < xDim) ? first(x+1,y,z) : first(x,y,z);
        float cur_y = (y+1 < yDim) ? first(x,y+1,z) : first(x,y,z);

        gradient[z](x,y,0) = 0.5*float( (x > 0) ? (cur_x - first(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) = 0.5*float( (y > 0) ? (cur_y - first(x,y-1,z)) : 0.0 );

        cur_x = (x+1 < xDim) ? second(x+1,y,z) : second(x,y,z);
        cur_y = (y+1 < yDim) ? second(x,y+1,z) : second(x,y,z);

        gradient[z](x,y,0) += 0.5*float( (x > 0) ? (cur_x - second(x-1,y,z)) : 0.0 );
        gradient[z](x,y,1) += 0.5*float( (y > 0) ? (cur_y - second(x,y-1,z)) : 0.0 );	
      }
    }
  
    gradient[z] *= 0.5f;
  }
}
