/*** first version written by Thomas Schoenemann as a private person without employment, October 2009 ***/
/*** continued by Thomas Schoenemann as an employee of Lund University, Sweden, February 2010 ***/


#include "application.hh"
#include "colorimage.hh"
#include "motion_estimator.hh"
#include "flowviz.hh"
#include "flow_eval.hh"
#include "smoothing.hh"
#include "timing.hh"
#include "sampling.hh"
#include "motion_lp.hh"
#include "motion_convexprog.hh"
#include <cstdio>
#include "motion_discrete.hh"
#include "motion_trws.hh"
#include "spline_interpolation.hh"

int main(int argc, char** argv) {


  if ( argc == 1 || (argc == 2 && strings_equal(argv[1],"-h"))) {

    std::cerr << "USAGE: " << argv[0] << std::endl
              << "  -first <ppm> : first image to be registered" << std::endl
              << "  -second <ppm> : second image to be registered" << std::endl
              << "  -alpha <double> : smoothness weight" << std::endl
              << " -o (filename, ppm) : file where the flow visualization is written" << std::endl
              << " [-txt (filename)] : write flows in txt-format" << std::endl
	      << " [-ground-truth <file>] " << std::endl
              << " [-method (vari | lp | generic-lp | convrelax | goldluecke | move | trws | mplp | msd | sg)] : choose optimization strategy" << std::endl
	      << " [-cuda]" << std::endl
	      << " [-smooth-iter <uint>]: number of smoothing iterations for preprocessing" << std::endl
	      << " ######## parameters for discretization methods ##########" << std::endl 
              << " [-n (4|8)] : neighborhood used for the discrete approaches" << std::endl
              << " [-spacing <uint>] : for discrete labels: number of labels that are placed in a pixel in each dimension" << std::endl
              << " [-epsilon <double>]: smoothing parameter used for the relaxation" << std::endl
	      << " ######## parameters for variational methods ##########" << std::endl 
	      << " [-spline-mode]: for variational methods, compute a spline representation of the images" << std::endl
	      << " [-data-norm (L1 | L2)]: kind of data term: absolutes or squares" << std::endl
	      << " [-reg-norm (L1 | L2 | L0.5)]: kind of regularity term: absolutes or squares" << std::endl
	      << " [-warping] : use warping scheme" << std::endl
	      << " [-warping-iter <uint>]: number of iterations if warping is selected" << std::endl
	      << " [-mscale] : use multi scale scheme" << std::endl
	      << std::endl;
    exit(1);
  }

  const int nParams = 20;
  ParamDescr  params[nParams] = {{"-first",mandInFilename,0,""},{"-second",mandInFilename,0,""},
                                 {"-alpha",optWithValue,1,"1.0"},{"-o",mandOutFilename,0,""},
                                 {"-smooth-iter",optWithValue,1,"3"},{"-data-norm",optWithValue,1,"l2"},
                                 {"-reg-norm",optWithValue,1,"l2"},{"-method",optWithValue,1,"vari"},
                                 {"-warping",flag,0,""},{"-n",optWithValue,1,"4"},{"-cuda",flag,0,""},
                                 {"-txt",optOutFilename,0,""},{"-ground-truth",optInFilename,0,""},
                                 {"-spacing",optWithValue,1,"1"},{"-range",optWithValue,1,"5"},
                                 {"-epsilon",optWithValue,1,"0.01"}, {"-standard-relax",flag,0,""},
                                 {"-spline-mode",flag,0,""},{"-warping-iter",optWithValue,1,"25"},
                                 {"-mscale",flag,0,""}};

  Application app(argc,argv,params,nParams);

  Math3D::NamedColorImage<float>  first(app.getParam("-first"),MAKENAME(first));
  Math3D::NamedColorImage<float>  second(app.getParam("-second"),MAKENAME(second));

  Math3D::NamedColorImage<float> org_second(MAKENAME(org_second));
  org_second = second;
  
  uint xDim = first.xDim();
  uint yDim = first.yDim();
  uint zDim = first.zDim();
  
  for (uint i = 0; i < 50; i++) {
    
    if ((i%2) == 0) 
      std::cerr << "DID YOU REMEMBER TO CURSE THE POLITICIANS????" << std::endl;
    else
      std::cerr << "YOU ARE REQUIRED TO CURSE THE POLITICIANS!!!!" << std::endl;
  } 

  uint nSmoothIter = convert<uint>(app.getParam("-smooth-iter"));
  
  std::cerr << "applying " << nSmoothIter << " iterations of smoothing " << std::endl;
  for (uint i=0; i < nSmoothIter; i++) {
    //     smooth_binomial(first);
    //     smooth_binomial(second);
    smooth_isotropic_gauss(first,1.0);
    smooth_isotropic_gauss(second,1.0);
  }
 
  first.savePPM("temp.ppm"); 

  double alpha = convert<double>(app.getParam("-alpha"));
  Math3D::NamedColorImage<float> out(first.xDim(),first.yDim(),3,MAKENAME(out));

  Math3D::NamedTensor<double> flow(xDim,yDim,2,0.0,MAKENAME(flow));

  uint spacing = convert<uint>(app.getParam("-spacing"));
  int range = convert<uint>(app.getParam("-range"));

  std::string method = app.getParam("-method");

  uint neighborhood = convert<uint>(app.getParam("-n"));

  Math2D::Matrix<uint> labeling(xDim,yDim,0);

  std::clock_t tStartMethod = std::clock();

  if (method == "lp") {

    if (neighborhood == 8)
      alpha *= 4.0 / (4.0 + 4.0*sqrt(0.5));

    double energy;

    std::clock_t tStart, tEnd;
    tStart = std::clock();

    if (app.is_set("-standard-relax")) {
      TODO("implement combination of augmented lagrangians and standard relaxation") ;
    }
    else {
      energy = implicit_conv_lp_motion_estimation(first, second, -range, range, -range, range, neighborhood, spacing, alpha, 
                                                  flow, /*&labeling*/ 0);

      //     energy = motion_estimation_smoothabs_nesterov(first, second, -range, range, -range, range, spacing, 
      // 						  neighborhood, alpha, flow, 0.05, app.is_set("-cuda"));
    }

    tEnd = std::clock();
    std::cerr << "method needed " << diff_seconds(tEnd,tStart) << " seconds." << std::endl;
    
    std::cerr << "energy " << energy << std::endl;
  }
  else if (method == "generic-lp") {
    
    double energy;

    if (app.is_set("-standard-relax"))
      energy = lp_motion_estimation_standard_relax(first, second, -range, range, -range, range, neighborhood, alpha, flow);
    else
      energy = lp_motion_estimation(first, second, -range, range, -range, range, neighborhood, alpha, flow);
  }
  else if (method == "convrelax") {

    if (neighborhood == 8)
      alpha *= 4.0 / (4.0 + 4.0*sqrt(0.5));

    double energy = 0.0;

    timeval tStart,tEnd;
    gettimeofday(&tStart,0);
    
    //energy = motion_estimation_quadprog(first, second, -5, 5, -5, 5, 4, alpha, flow);
    //energy = motion_estimation_quadprog(first, second, -1, 1, -1, 1, 4, alpha, flow);
    //energy = motion_estimation_quadprog_bcd(first, second, -5, 5, -5, 5, 4, alpha, flow);
    //energy = motion_estimation_quadprog_bcd(first, second, -1, 1, -1, 1, 4, alpha, flow);
    
    if (app.is_set("-standard-relax"))
      energy = motion_estimation_convprog_standardrelax_nesterov_smoothapprox(first, second, -range, range, -range, range, spacing, 
                                                                              neighborhood, alpha, flow, 0.001/*, app.is_set("-cuda")*/);
    else {
      //energy = motion_estimation_convprog_nesterov(first, second, -range, range, -range, range, spacing, 
      //					   neighborhood, alpha, flow, exponent, app.is_set("-cuda"));

      energy = motion_estimation_convprog_nesterov_smoothapprox(first, second, -range, range, -range, range, spacing, 
                                                                neighborhood, alpha, flow, convert<double>(app.getParam("-epsilon")), 
                                                                app.is_set("-cuda"));
    }

    std::cerr << "energy " << energy << std::endl;

    gettimeofday(&tEnd,0);
    std::cerr << "qp-flow computation took " << diff_seconds(tEnd,tStart) << " seconds." << std::endl;
  }
  else if (method == "vari") {

    std::string reg_string = app.getParam("-reg-norm");

    NormType reg_norm = L2;

    if (reg_string == "l1" || reg_string == "L1")
      reg_norm = L1;
    else if (reg_string == "l0.5" || reg_string == "L0.5")
      reg_norm = L0_5;
    else if (reg_string != "l2" && reg_string != "L2") {
      USER_ERROR << " unknown regularity term \"" << reg_string << "\". Exiting." << std::endl; 
      exit(1);
    }

    std::string data_string = app.getParam("-data-norm");

    DifferenceType data_norm = SquaredDiffs;

    if (data_string == "l1" || data_string == "L1")
      data_norm = AbsDiffs;
    else if (data_string != "l2" && data_string != "L2") {
      USER_ERROR << " unknown data norm \"" << data_string << "\". Exiting." << std::endl; 
      exit(1);
    }

    MotionEstimator motion_estimator(first,second,data_norm,reg_norm,alpha,!app.is_set("-warping"),
                                     app.is_set("-spline-mode"), convert<double>(app.getParam("-epsilon")));
    
    timeval tStart,tEnd;
    gettimeofday(&tStart,0);

    uint nWarpingIter = convert<uint>(app.getParam("-warping-iter"));
    if (app.is_set("-mscale"))
      motion_estimator.compute_flow_multi_scale(nWarpingIter);
    else
      motion_estimator.compute_flow(nWarpingIter);
    
    std::cerr << "energy: " << motion_estimator.energy() << std::endl;
    
    gettimeofday(&tEnd,0);
    std::cerr << "flow computation took " << diff_seconds(tEnd,tStart) << " seconds." << std::endl;
   
    //std::cerr << "A" << std::endl;
 
    flow = motion_estimator.flow();
  }
  else if (method == "goldluecke") {
    
    double energy;
    
    energy = motion_estimation_goldluecke_cremers(first, second, -range, range, -range, range, spacing, 
                                                  neighborhood, alpha, flow);
  }
  else if (method == "move") {

    if (neighborhood == 8)
      alpha *= 4.0 / (4.0 + 4.0*sqrt(0.5));

    expmove_motion_estimation(first, second,-range, range, -range, range, spacing, 
                              neighborhood, alpha, flow);
  }
  else if (method == "trws") {

    if (neighborhood == 8)
      alpha *= 4.0 / (4.0 + 4.0*sqrt(0.5));

    // trws_motion_estimation(first, second,-range, range, -range, range, spacing,
    // 			   neighborhood, alpha, flow);

    message_passing_motion_estimation(first, second,-range, range, -range, range, spacing,
                                      neighborhood, alpha, "trws", flow);

  }
  else if (method == "mplp" || method == "sg" || method == "msd") {

    if (neighborhood == 8)
      alpha *= 4.0 / (4.0 + 4.0*sqrt(0.5));

    message_passing_motion_estimation(first, second,-range, range, -range, range, spacing,
                                      neighborhood, alpha, method, flow);
  }
  else {
    USER_ERROR << "unknown method." << std::endl;
  }

  //std::cerr << "B" << std::endl;

  std::cerr << "maximal flow norm: " << flow.max_vector_norm() << std::endl;

  visualize_flow(flow, flow.max_vector_norm()+0.01,out,true);
  out.savePPM(app.getParam("-o"));

  if (app.is_set("-ground-truth")) {

    Math3D::NamedTensor<double> gt_flow(xDim,yDim,2,0.0,MAKENAME(gt_flow));

    std::string gt_name = app.getParam("-ground-truth").c_str();

    uint cxDim = 0;
    uint cyDim = 0;

    if (string_ends_with(gt_name,".flo")) {

      //std::ifstream gt(gt_name.c_str(),std::ios::binary);
      FILE* gt = fopen(gt_name.c_str(),"rb");
      
      std::cerr << "ground truth seems to be in Middlebury format" << std::endl;

      float temp;

      //gt >> temp;
      fread(&temp, 4, 1, gt);

      if (temp == 202021.25) {

        //gt >> cxDim;
        //gt >> cyDim;

        fread(&cxDim, 4, 1, gt);
        fread(&cyDim, 4, 1, gt);

        if (cxDim != xDim || cyDim != cyDim) 
          std::cerr << "ERROR: dimensions of image and ground truth mismatch" << std::endl;
        else {
	
          for (uint y=0; y < yDim; y++) {
            for (uint x=0; x < xDim; x++) {
              //gt >> temp;
              fread(&temp, 4, 1, gt);
              gt_flow(x,y,0) = temp;
              //gt >> temp;
              fread(&temp, 4, 1, gt);	    
              gt_flow(x,y,1) = temp;
            }
          }
        }
      }
      else 
        std::cerr << "ERROR: failed to read ground truth." << std::endl;
    }
    else {

      std::ifstream gt(gt_name.c_str());

      gt >> cxDim >> cyDim;
      if (cxDim != xDim || cyDim != yDim) { 
	
        USER_ERROR << ": the dimensions of the specified ground truth do not match the image" << std::endl;
      }
      else {
	
        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
            gt >> gt_flow(x,y,0) ;
          }
        }

        for (uint y=0; y < yDim; y++) {
          for (uint x=0; x < xDim; x++) {
            gt >> gt_flow(x,y,1) ;
          }
        }
      }
    }

    Math3D::ColorImage<float> gt_viz;
    double max_norm = 0.0;
    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {

        double norm = gt_flow.norm(x,y);

        if (norm < 1e8)
          max_norm = std::max(norm,max_norm);
      }
    }
    visualize_flow(gt_flow, max_norm+0.01,gt_viz,true);
    gt_viz.savePPM("gt.ppm");
    
    double aae = average_angular_error(flow,gt_flow);
    std::cerr << "AAE: " << aae << std::endl;
  }

  std::clock_t tEndMethod = std::clock();

  std::cerr << "computation needed " << diff_seconds(tEndMethod, tStartMethod) << " seconds." << std::endl;

  //std::cerr << "C" << std::endl;

  Math3D::NamedTensor<float> warped_im(xDim,yDim,zDim,MAKENAME(warped_im));
  for (uint y=0; y < yDim; y++) {
    for (uint x=0; x < xDim; x++) {

      float tx = x + flow(x,y,0);
      float ty = y + flow(x,y,1);

      for (uint z=0; z < zDim; z++) {

        warped_im(x,y,z) = bilinear_interpolation(org_second, tx,ty,z);
      }
    }
  }

  //std::cerr << "D" << std::endl;

  warped_im.savePPM("warped.ppm",org_second.max_intensity());

  if (app.is_set("-txt")) {

    std::ofstream txt(app.getParam("-txt").c_str());
    txt << xDim << std::endl
        << yDim << std::endl;

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
        txt << flow(x,y,0) << std::endl;
      }
    }

    for (uint y=0; y < yDim; y++) {
      for (uint x=0; x < xDim; x++) {
        txt << flow(x,y,1) << std::endl;
      }
    }
  }

}
