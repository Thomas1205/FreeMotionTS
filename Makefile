include common/Makefile.common
INCLUDE += -I common/ -I optimization/ -I thirdparty/

all : $(DEBUGDIR) $(OPTDIR) .subdirs freemotionts.opt.L64

freemotionts.opt.L64: $(OPTDIR)/conv_lp_solving.o $(OPTDIR)/flow_eval.o $(OPTDIR)/flowviz.o $(OPTDIR)/gradient.o $(OPTDIR)/motion_discrete.o $(OPTDIR)/motion_trws.o $(OPTDIR)/motion_estimator.o $(OPTDIR)/motion_lp.o $(OPTDIR)/motion_convexprog.o $(OPTDIR)/submodular_energy_minimization.o $(OPTDIR)/mrf_energy.o $(OPTDIR)/spline_interpolation.o optimization/$(OPTDIR)/factorDualOpt.o optimization/$(OPTDIR)/factorChainDualDecomp.o optimization/$(OPTDIR)/factorMPBP.o optimization/$(OPTDIR)/trws.o optimization/$(OPTDIR)/factorTRWS.o common/$(OPTDIR)/tensor.o common/$(OPTDIR)/matrix.o common/$(OPTDIR)/vector.o common/$(OPTDIR)/stringprocessing.o common/$(OPTDIR)/application.o common/$(OPTDIR)/fileio.o common/$(OPTDIR)/makros.o common/$(OPTDIR)/timing.o
	$(LINKER) $(OPTFLAGS) $(INCLUDE) optic_flow.cc $(OPTDIR)/conv_lp_solving.o $(OPTDIR)/flow_eval.o $(OPTDIR)/flowviz.o $(OPTDIR)/gradient.o $(OPTDIR)/motion_discrete.o $(OPTDIR)/motion_trws.o $(OPTDIR)/motion_estimator.o $(OPTDIR)/motion_lp.o $(OPTDIR)/motion_convexprog.o $(OPTDIR)/submodular_energy_minimization.o $(OPTDIR)/mrf_energy.o $(OPTDIR)/spline_interpolation.o optimization/$(OPTDIR)/factorDualOpt.o optimization/$(OPTDIR)/factorChainDualDecomp.o optimization/$(OPTDIR)/factorMPBP.o optimization/$(OPTDIR)/trws.o optimization/$(OPTDIR)/factorTRWS.o common/$(OPTDIR)/tensor.o common/$(OPTDIR)/matrix.o common/$(OPTDIR)/vector.o common/$(OPTDIR)/stringprocessing.o common/$(OPTDIR)/application.o common/$(OPTDIR)/fileio.o common/$(OPTDIR)/makros.o common/$(OPTDIR)/timing.o

.subdirs :
	cd common; make; cd -; cd optimization; make; cd -

include common/Makefile.finish
    
clean:
	cd common; make clean; cd -
	cd optimization; make clean; cd -
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o 

  