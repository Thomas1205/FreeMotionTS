include ../common/Makefile.common

INCLUDE += -I ../common

all: $(LIB) $(DEBUGDIR) $(OPTDIR) $(LIB)/optlib.debug $(LIB)/optlib.opt 

$(LIB)/optlib.debug: $(DEBUGDIR)/factorDualOpt.o $(DEBUGDIR)/factorMPBP.o 
	ar rs $@ $(DEBUGDIR)/factorDualOpt.o $(DEBUGDIR)/factorMPBP.o 
	
$(LIB)/optlib.opt: $(OPTDIR)/factorDualOpt.o $(OPTDIR)/factorMPBP.o 
	ar rs $@ $(OPTDIR)/factorDualOpt.o $(OPTDIR)/factorMPBP.o 

clean:
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o
	rm -f lib/* 

include ../common/Makefile.finish

