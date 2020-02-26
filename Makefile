include common/Makefile.common
INCLUDE += -I common/ -I optimization/ -I thirdparty/

all : $(DEBUGDIR) $(OPTDIR) .subdirs freemotionts.opt.L64

freemotionts.opt.L64: $(OPTDIR)/conv_lp_solving.o $(OPTDIR)/flow_eval.o $(OPTDIR)/flowviz.o $(OPTDIR)/motion_discrete.o $(OPTDIR)/motion_estimator.o $(OPTDIR)/motion_lp.o $(OPTDIR)/mrf_energy.o $(OPTDIR)/spline_interpolation.o
	$(LINKER) $(OPTFLAGS) $(INCLUDE) optic_flow.cc $(OPTDIR)/conv_lp_solving.o $(OPTDIR)/flow_eval.o $(OPTDIR)/flowviz.o $(OPTDIR)/motion_discrete.o $(OPTDIR)/motion_estimator.o $(OPTDIR)/motion_lp.o $(OPTDIR)/mrf_energy.o $(OPTDIR)/spline_interpolation.o

.subdirs :
	cd common; make; cd -; cd optimization; make; cd -
  
  
clean:
	cd common; make clean; cd -
	cd optimization; make clean; cd -
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o 

include common/Makefile.finish
  