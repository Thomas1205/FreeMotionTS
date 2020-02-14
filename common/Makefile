include Makefile.common

all: $(LIB) $(DEBUGDIR) $(OPTDIR) $(LIB)/commonlib.debug $(LIB)/commonlib.opt $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o

$(LIB)/commonlib.debug: $(DEBUGDIR)/application.o $(DEBUGDIR)/timing.o $(OPTDIR)/vector.o $(DEBUGDIR)/matrix.o $(DEBUGDIR)/tensor.o $(DEBUGDIR)/makros.o $(DEBUGDIR)/fileio.o $(DEBUGDIR)/stringprocessing.o
	ar rs $@ $(DEBUGDIR)/application.o $(DEBUGDIR)/timing.o $(OPTDIR)/vector.o $(DEBUGDIR)/matrix.o $(DEBUGDIR)/tensor.o  $(DEBUGDIR)/makros.o $(DEBUGDIR)/fileio.o $(DEBUGDIR)/stringprocessing.o

$(LIB)/commonlib.opt: $(OPTDIR)/application.o $(OPTDIR)/timing.o $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o  $(OPTDIR)/makros.o $(OPTDIR)/fileio.o $(OPTDIR)/stringprocessing.o
	ar rs $@ $(OPTDIR)/application.o $(OPTDIR)/timing.o $(OPTDIR)/vector.o $(OPTDIR)/matrix.o $(OPTDIR)/tensor.o  $(OPTDIR)/makros.o $(OPTDIR)/fileio.o $(OPTDIR)/stringprocessing.o

clean:
	rm -f $(DEBUGDIR)/*.o 
	rm -f $(OPTDIR)/*.o
	rm -f lib/* 

include Makefile.finish

