/*** written by Thomas Schoenemann as a private person without employment, October 2009 ***/

#ifndef SUBMODULAR_ENERGY_MINIMIZATION_HH
#define SUBMODULAR_ENERGY_MINIMIZATION_HH

#include "graph.h"
#include "matrix.hh"

#include <vector>
#include <map>

void add_truncated_linear(Graph<double,double,double>& graph, const std::vector<uint>& participating_nodes,
                          uint cutoff_point, double cutoff_value, bool is_source_term=true);


//the function is specified in terms of its breakpoints
void add_concave_increasing_function(Graph<double,double,double>& graph,
                                     const std::vector<uint>& participating_nodes,
                                     const std::map<uint,double>& function, bool is_source_term = true);

//returns an energy offset
double add_term2(Graph<double,double,double>& graph, uint node1, uint node2,
                 double e00, double e01, double e10, double e11);

double add_term3(Graph<double,double,double>& graph, uint node1, uint node2, uint node3,
                 double e000, double e001, double e010, double e011,
                 double e100, double e101, double e110, double e111);


//returns an energy offset
long double add_term2(Graph<long double,long double,long double>& graph, uint node1, uint node2,
                      long double e00, long double e01, long double e10, long double e11);

long double add_term3(Graph<long double,long double,long double>& graph, uint node1, uint node2, uint node3,
                      long double e000, long double e001, long double e010, long double e011,
                      long double e100, long double e101, long double e110, long double e111);



#endif
