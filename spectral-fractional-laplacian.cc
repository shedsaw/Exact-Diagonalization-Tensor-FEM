 /* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Toby D. Young, Polish Academy of Sciences,
 *          Wolfgang Bangerth, Texas A&M University
 *
 * Modified by: Shane Sawyer, Univeristy of Tennessee, Knoxville,
 *              Abner Salgado, University of Tennessee, Knoxville
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_series.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <complex>
#include <fstream>
#include <iostream>
#include <math.h>
#include <iterator>
#include <algorithm>
#include <limits>

#include<boost/math/special_functions/bessel.hpp>
#include "gsl/gsl_sf_bessel.h"

using namespace dealii;

namespace SpectralFractionalLaplacian
{
  using namespace dealii;

  class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);
 
  private:
    void              declare_parameters();
    ParameterHandler &prm;
  };
 
  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}
 
 
  void ParameterReader::declare_parameters()
  {
    prm.enter_subsection("Mesh & geometry parameters");
    {
      prm.declare_entry("Geometry",
                        "SQUARE",
			Patterns::Selection("SQUARE|LSHAPED|CIRCULAR"),
                        "Select the geometry: SQUARE, LSHAPED, CIRCULAR");
 
      prm.declare_entry("Starting refinement",
                        "4",
                        Patterns::Integer(0,20),
                        "Select the starting level of refinement");

      prm.declare_entry("Ending refinement",
                        "5",
                        Patterns::Integer(0,20),
                        "Select the ending level of refinement");
    }
    prm.leave_subsection();
 
    prm.enter_subsection("Source term");
    {
      prm.declare_entry("Source term", "SM", Patterns::Selection("SM|MM|BE"),
			"Select the right hand side: single sinusoidal mode (SM), "
			"three sinusoidal modes (MM), Bessel function of the first kind (BE)");
 
      prm.declare_entry("omega", "5.0e7", Patterns::Double(0), "Frequency");
    }
    prm.leave_subsection();
 
 
    prm.enter_subsection("Problem configuration");
    {
      prm.declare_entry("Height",
                        "10.0",
                        Patterns::Double(0.0),
                        "Set the height where the extended dimension is truncated");

      prm.declare_entry("Fractional power",
                        "0.5",
                        Patterns::Double(0.0,1.0),
                        "Set the fractional power of the Laplacian operator");

      prm.declare_entry("Number of eigenpairs",
                        "1000",
                        Patterns::Integer(1),
                        "Set the number of eigenvalue/eigenfunction pairs to compute and apply");
    }
    prm.leave_subsection();
  }
 
  void ParameterReader::read_parameters(const std::string &parameter_file)
  {
    declare_parameters();
 
    prm.parse_input(parameter_file);
  }
  
  // Single mode
  template <int dim> class SolutionSM : public Function<dim> {
  public:
    virtual double value(const Point<dim> & p,
			 const unsigned int = 0) const override {
      return (std::sin( numbers::PI * p(0)) * std::sin( numbers::PI * p(1)));
    };
    
    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
	     const unsigned int = 0) const override {
      Tensor<1, dim> return_value;
      return_value[0] = numbers::PI * std::cos( numbers::PI * p(0)) * std::sin( numbers::PI * p(1));
      return_value[1] = numbers::PI * std::sin( numbers::PI * p(0)) * std::cos( numbers::PI * p(1));
      return return_value;
    };
  };
  
  // Class for the right hand side function f.
  template <int dim> class RightHandSideSM : public Function<dim> {
  public:
    RightHandSideSM( const double ss )
      : s(ss)
      , multiplier( std::pow( 2.0*numbers::PI*numbers::PI , s ) )
    {}
    virtual double value(const Point<dim> & p, const unsigned int = 0) const override {
      return (multiplier*std::sin( numbers::PI*p(0) )*std::sin( numbers::PI*p(1) ) );
    }
  protected:
    double s, multiplier;
  };

  // Multiple Modes
  template <int dim> class SolutionMM : public Function<dim> {
  public:
    virtual double value(const Point<dim> & p,
			 const unsigned int = 0) const override {
      return (std::sin( numbers::PI * p(0)) * std::sin( numbers::PI * p(1)) +
              std::sin( 3.0*numbers::PI * p(0)) * std::sin( 2.0*numbers::PI * p(1)) +
              std::sin( 5.0*numbers::PI * p(0)) * std::sin( 4.0*numbers::PI * p(1)) );
    };
    
    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
	     const unsigned int = 0) const override {
      Tensor<1, dim> return_value;
      return_value[0] = numbers::PI * std::cos(    numbers::PI * p(0)) * std::sin(    numbers::PI * p(1)) +
                    3.0*numbers::PI * std::cos( 3.*numbers::PI * p(0)) * std::sin( 2.*numbers::PI * p(1)) +
                    5.0*numbers::PI * std::cos( 5.*numbers::PI * p(0)) * std::sin( 4.*numbers::PI * p(1));
      return_value[1] = numbers::PI * std::sin(    numbers::PI * p(0)) * std::cos(    numbers::PI * p(1)) +
                    2.0*numbers::PI * std::sin( 3.*numbers::PI * p(0)) * std::cos( 2.*numbers::PI * p(1)) +
                    4.0*numbers::PI * std::sin( 5.*numbers::PI * p(0)) * std::cos( 4.*numbers::PI * p(1));
      
      return return_value;
    };
  };
  
  // Class for the right hand side function f.
  template <int dim> class RightHandSideMM : public Function<dim> {
  public:
    RightHandSideMM( const double ss )
      : s(ss)
      , multiplier( std::pow( 2.0*numbers::PI*numbers::PI , s ) )
    {}
    virtual double value(const Point<dim> & p, const unsigned int = 0) const override {
      double temp = std::pow( 2.0*numbers::PI*numbers::PI , s )       * std::sin(     numbers::PI * p(0)) * std::sin(     numbers::PI * p(1));
      temp +=       std::pow( numbers::PI*numbers::PI*( 3*3 + 2*2), s)* std::sin( 3.0*numbers::PI * p(0)) * std::sin( 2.0*numbers::PI * p(1));
      temp +=       std::pow( numbers::PI*numbers::PI*( 5*5 + 4*4), s)* std::sin( 5.0*numbers::PI * p(0)) * std::sin( 4.0*numbers::PI * p(1));
      return temp;
    }
  protected:
    double s, multiplier;
  };
  
  
  // Circular Domain Function
  template <int dim> class SolutionBE : public Function<dim> {
  public:
    virtual double value(const Point<dim> & p,
			 const unsigned int = 0) const override {
      double root = 3.831705970207512;
      double A = 0.977410601343874;
      double r = std::sqrt( p(0)*p(0) + p(1)*p(1) );

      return ( A * ( (p(0)+p(1))/r ) * boost::math::cyl_bessel_j( 1.0, root*r ) );
    };
    
    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
	     const unsigned int = 0) const override {
      Tensor<1, dim> return_value;
      double root = 3.831705970207512;
      double A = 0.977410601343874;
      double r = std::sqrt( p(0)*p(0) + p(1)*p(1) );

      return_value[0] =  ( -1.0 * A * p(0) * (p(0)+p(1)) * boost::math::cyl_bessel_j( 1.0, root*r ) )/( std::pow( r, 3 ) );
      return_value[0] += ( A * boost::math::cyl_bessel_j( 1.0, root*r ) ) / r;
      return_value[0] += ( A * root * p(0) * ( p(0) + p(1) ) * boost::math::cyl_bessel_j( 0.0, root*r ) - boost::math::cyl_bessel_j( 2.0, root*r ) ) / (2.0 * r * r);

      return_value[0] =  ( -1.0 * A * p(1) * (p(0)+p(1)) * boost::math::cyl_bessel_j( 1.0, root*r ) )/( std::pow( r, 3 ) );
      return_value[0] += ( A * boost::math::cyl_bessel_j( 1.0, root*r ) ) / r;
      return_value[0] += ( A * root * p(1) * ( p(0) + p(1) ) * boost::math::cyl_bessel_j( 0.0, root*r ) - boost::math::cyl_bessel_j( 2.0, root*r ) ) / (2.0 * r * r);
      
      return return_value;
    };
  };
  
  // Class for the right hand side function f.
  template <int dim> class RightHandSideBE : public Function<dim> {
  public:
    RightHandSideBE( const double ss )
      : s(ss)
    {}
    virtual double value(const Point<dim> & p, const unsigned int = 0) const override {
      double root = 3.831705970207512;
      double A = 0.977410601343874;
      double r = std::sqrt( p(0)*p(0) + p(1)*p(1) );
      
      double temp = std::pow( root, 2.0*s) * ( A * ( (p(0)+p(1))/r ) * boost::math::cyl_bessel_j( 1.0, root*r ) );
      return temp;
    }
  protected:
    double s;
  };
  
 
  template <int dim> class SpectralFractionalLaplace
  {
  public:
    SpectralFractionalLaplace(ParameterHandler &);
    
    void run();
    
  private:
    void make_grid(unsigned int n_refine);
    void setup_system();
    void assemble_system();
    void solve();
    void output_results(int n) const;

    ParameterHandler &prm;
    
    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    PETScWrappers::SparseMatrix system_matrix;
    PETScWrappers::SparseMatrix mass_matrix;
    PETScWrappers::SparseMatrix stiffness_matrix;

    AffineConstraints<double> constraints;

    PETScWrappers::MPI::Vector p_solution;
    PETScWrappers::MPI::Vector p_system_rhs;
    PETScWrappers::MPI::Vector p_system_rhs_base;

    Vector<double> solution;
    Vector<double> system_rhs;
    Vector<double> system_rhs_base;
    
    Vector<double> local_solution;   // Local contribution to global solution.
    Vector<double> global_solution;  // Full global solution.
    
    std::vector<double>      evs;
    std::vector<double>      efs;

    std::string  geometry;
    std::string  source;
    unsigned int starting_refinements;
    unsigned int ending_refinements;
    unsigned int num_eigen_pairs;
    ConvergenceTable convergence_table;
    double s, Y;

    MPI_Comm mpi_communicator;
    int my_color;
    MPI_Comm local_comm;
    MPI_Group group_world;
    ConditionalOStream pcout;
  };

  template <int dim>
  SpectralFractionalLaplace<dim>::SpectralFractionalLaplace(ParameterHandler &param)
    : prm(param)
    , fe(1)
    , dof_handler(triangulation)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {
    prm.enter_subsection("Mesh & geometry parameters");
    geometry = prm.get("Geometry");
    starting_refinements  = prm.get_integer("Starting refinement");
    ending_refinements    = prm.get_integer("Ending refinement");
    prm.leave_subsection();

    prm.enter_subsection("Source term");
    source = prm.get("Source term");
    prm.leave_subsection();

    prm.enter_subsection("Problem configuration");
    Y = prm.get_double("Height");
    s = prm.get_double("Fractional power");
    num_eigen_pairs = prm.get_integer("Number of eigenpairs");
    prm.leave_subsection();

    int ierr=-1;
    int my_rank=-1;
    
    my_color = Utilities::MPI::this_mpi_process( mpi_communicator );
    
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    ierr = MPI_Comm_split(MPI_COMM_WORLD, my_color, my_rank, &local_comm);  
  }

  template <int dim>
  void SpectralFractionalLaplace<dim>::make_grid(unsigned int n_refine) {

    const std::string square = "SQUARE";
    const std::string lshaped = "LSHAPED";
    const std::string circular = "CIRCULAR";
    
    if ( geometry == square ) {
      GridGenerator::hyper_cube(triangulation, 0, 1);
      triangulation.refine_global(n_refine);
    }
    if ( geometry == lshaped ) {
      GridGenerator::hyper_L(triangulation, -1.0, 1.0);
      triangulation.refine_global(n_refine);
      GridTools::rotate(3.0*numbers::PI/2.0, triangulation);
    }
    if ( geometry == circular ) {
      GridGenerator::hyper_ball(triangulation);
      triangulation.refine_global(n_refine);
    }
  }

  template <int dim>
  void SpectralFractionalLaplace<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
	  << std::endl;

    pcout << "Number of locally owned degrees of freedom: " << dof_handler.n_locally_owned_dofs()
	  << std::endl;
 
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    
    stiffness_matrix.reinit(dof_handler.n_dofs(),
                            dof_handler.n_dofs(),
                            dof_handler.max_couplings_between_dofs());
    mass_matrix.reinit(dof_handler.n_dofs(),
		       dof_handler.n_dofs(),
                       dof_handler.max_couplings_between_dofs());
    
    system_matrix.reinit(dof_handler.n_dofs(),
                         dof_handler.n_dofs(),
                         dof_handler.max_couplings_between_dofs());
    
    local_solution.reinit( dof_handler.n_dofs() );
    global_solution.reinit( dof_handler.n_dofs() );

    p_solution.clear();
    p_solution.reinit( local_comm, dof_handler.n_locally_owned_dofs(), dof_handler.n_locally_owned_dofs() );

    p_system_rhs.clear();
    p_system_rhs.reinit( local_comm, dof_handler.n_locally_owned_dofs(), dof_handler.n_locally_owned_dofs() );

    p_system_rhs_base.clear();
    p_system_rhs_base.reinit( local_comm, dof_handler.n_locally_owned_dofs(), dof_handler.n_locally_owned_dofs() );
  }

  template <int dim>
  void SpectralFractionalLaplace<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fe.degree + 1);

    const std::string single_mode    = "SM";
    const std::string multiple_modes = "MM";
    const std::string bessel         = "BE";

    RightHandSideSM<dim> right_hand_sideSM(s);
    RightHandSideMM<dim> right_hand_sideMM(s);
    RightHandSideBE<dim> right_hand_sideBE(s);

    if ( source == single_mode ) {
      QGauss<dim> quadrature_formula(fe.degree + 1);
      
      FEValues<dim> fe_values(fe,
			      quadrature_formula,
			      update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    
      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      
      for (const auto &cell : dof_handler.active_cell_iterators()) {
	fe_values.reinit(cell);

	cell_mass_matrix      = 0;
	cell_stiffness_matrix = 0;
	cell_rhs              = 0;

	for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
	  for (const unsigned int i : fe_values.dof_indices())
	    for (const unsigned int j : fe_values.dof_indices()) {
	      cell_mass_matrix(i,j) += ( fe_values.shape_value(i, q_index)  *   // phi_i(x_q)
					 fe_values.shape_value(j, q_index)) *   // phi_j(x_q)
		                         fe_values.JxW(q_index);                  // dx

	      cell_stiffness_matrix(i,j) += ( fe_values.shape_grad(i, q_index)  *   // grad phi_i(x_q)
					      fe_values.shape_grad(j, q_index)) *   // grad phi_j(x_q)
	                                      fe_values.JxW(q_index);                 // dx
	    
	    }
	    
	
	  const auto &x_q = fe_values.quadrature_point(q_index);
	  for (const unsigned int i : fe_values.dof_indices())
	    cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
			    right_hand_sideSM.value(x_q) *        // f(x_q)
			    fe_values.JxW(q_index));            // dx
	}

	cell->get_dof_indices(local_dof_indices);

	constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
	constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               system_matrix);
	
	for (const unsigned int i : fe_values.dof_indices())
	  p_system_rhs_base(local_dof_indices[i]) += cell_rhs(i);
      }
      stiffness_matrix.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);
      system_matrix.compress(VectorOperation::add);
      p_system_rhs_base.compress(VectorOperation::add);
    }
    else if ( source == multiple_modes ) {
      QGauss<dim> quadrature_formula(fe.degree + 1);
      
      FEValues<dim> fe_values(fe,
			      quadrature_formula,
			      update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    
      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      
      for (const auto &cell : dof_handler.active_cell_iterators()) {
	fe_values.reinit(cell);

	cell_mass_matrix      = 0;
	cell_stiffness_matrix = 0;
	cell_rhs              = 0;

	for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
	  for (const unsigned int i : fe_values.dof_indices())
	    for (const unsigned int j : fe_values.dof_indices()) {
	      cell_mass_matrix(i,j) += ( fe_values.shape_value(i, q_index)  *   // phi_i(x_q)
					 fe_values.shape_value(j, q_index)) *   // phi_j(x_q)
		                         fe_values.JxW(q_index);                  // dx

	      cell_stiffness_matrix(i,j) += ( fe_values.shape_grad(i, q_index)  *   // grad phi_i(x_q)
					      fe_values.shape_grad(j, q_index)) *   // grad phi_j(x_q)
	                                      fe_values.JxW(q_index);                 // dx
	    
	    }
	    
	
	  const auto &x_q = fe_values.quadrature_point(q_index);
	  for (const unsigned int i : fe_values.dof_indices())
	    cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
			    right_hand_sideMM.value(x_q) *        // f(x_q)
			    fe_values.JxW(q_index));            // dx
	}

	cell->get_dof_indices(local_dof_indices);

	constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
	constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               system_matrix);

	for (const unsigned int i : fe_values.dof_indices())
	  p_system_rhs_base(local_dof_indices[i]) += cell_rhs(i);
      }

      stiffness_matrix.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);
      system_matrix.compress(VectorOperation::add);
      p_system_rhs_base.compress(VectorOperation::add);
    }
    else if ( source == bessel ) {
      QGauss<dim> quadrature_formula(fe.degree + 1);
      
      FEValues<dim> fe_values(fe,
			      quadrature_formula,
			      update_values | update_gradients |
			      update_quadrature_points | update_JxW_values);
    
      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      
      for (const auto &cell : dof_handler.active_cell_iterators()) {
	fe_values.reinit(cell);

	cell_mass_matrix      = 0;
	cell_stiffness_matrix = 0;
	cell_rhs              = 0;

	for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
	  for (const unsigned int i : fe_values.dof_indices())
	    for (const unsigned int j : fe_values.dof_indices()) {
	      cell_mass_matrix(i,j) += ( fe_values.shape_value(i, q_index)  *   // phi_i(x_q)
					 fe_values.shape_value(j, q_index)) *   // phi_j(x_q)
		                         fe_values.JxW(q_index);                  // dx

	      cell_stiffness_matrix(i,j) += ( fe_values.shape_grad(i, q_index)  *   // grad phi_i(x_q)
					      fe_values.shape_grad(j, q_index)) *   // grad phi_j(x_q)
	                                      fe_values.JxW(q_index);                 // dx
	    
	    }
	    
	
	  const auto &x_q = fe_values.quadrature_point(q_index);
	  for (const unsigned int i : fe_values.dof_indices())
	    cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
			    right_hand_sideBE.value(x_q) *        // f(x_q)
			    fe_values.JxW(q_index));            // dx
	}

	cell->get_dof_indices(local_dof_indices);

	constraints.distribute_local_to_global(cell_stiffness_matrix,
                                               local_dof_indices,
                                               stiffness_matrix);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
	constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               system_matrix);

	for (const unsigned int i : fe_values.dof_indices())
	  p_system_rhs_base(local_dof_indices[i]) += cell_rhs(i);
      }

      stiffness_matrix.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);
      system_matrix.compress(VectorOperation::add);
      p_system_rhs_base.compress(VectorOperation::add);
    }
    else {
      pcout<<"  Something went terribly wrong in assemble_system()!"<<std::endl;
      MPI_Abort( MPI_COMM_WORLD, 0 );
    }
  }

  template <int dim>
  void SpectralFractionalLaplace<dim>::solve() {
    
    SolverControl solver_control(1000, 1e-12);
    PETScWrappers::SolverCG cg(solver_control);
    PETScWrappers::PreconditionSOR preconditioner(system_matrix);

    cg.solve(system_matrix, p_solution, p_system_rhs, preconditioner);

    Vector<double> localized_solution(p_solution);
    constraints.distribute(localized_solution);
    solution = localized_solution;

    return;
  }
  
  template <int dim>
  void SpectralFractionalLaplace<dim>::output_results(int n) const {
    DataOut<dim> data_out;
    
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(global_solution, "Solution" );
    
    data_out.build_patches();
    
    std::string solution_filename = "solution-"+ std::to_string(n) + ".vtk";
    std::ofstream output(solution_filename);
    data_out.write_vtk(output);
  }

  template <int dim>
  void SpectralFractionalLaplace<dim>::run() {
    double starttime, endtime;
    unsigned int nprocs = Utilities::MPI::n_mpi_processes( mpi_communicator );
    unsigned int my_id  = Utilities::MPI::this_mpi_process( mpi_communicator );
    MPI_Barrier(mpi_communicator);
    starttime = MPI_Wtime();
    
    pcout<<" From spectral-fractional-laplacian.prm:"<<std::endl;
    pcout<<"   Geometry            - "<<geometry<<std::endl;
    pcout<<"   Starting refinement - "<<starting_refinements<<std::endl;
    pcout<<"   Ending refinement   - "<<ending_refinements<<std::endl;
    pcout<<"   Source tern         - "<<source<<std::endl;
    pcout<<"   Y                   - "<<Y<<std::endl;
    pcout<<"   s                   - "<<s<<std::endl;
    pcout<<"   Number of eigenpairs- "<<num_eigen_pairs<<std::endl;

    const bool isSqrt = fabs( s - 0.5 )<1e-12;
    double etak=0.0, muk=0.0, vk=0.0, Ak=1.0;

    // Assemble the base mass and stiffness matrices. along with the base right hand side.
    double ds = std::pow(2, 1.0-2.0*s)*std::tgamma(1.0-s)/std::tgamma(s);

    std::string local_filename = "rank-"+std::to_string(my_id)+".txt";
    std::ofstream local_output{ local_filename,std::ofstream::out };
      
    double starttime_local, stoptime_local;
    double start_setup, end_setup;
    double start_eigenpairs;
    MPI_Barrier(mpi_communicator);
    starttime_local = MPI_Wtime();
    
    for ( unsigned int nrefine = starting_refinements; nrefine <= ending_refinements; nrefine++) {
      pcout<<"Starting refinement level "<<nrefine<<std::endl;

      // Define string from split colour and use to set comm name
      std::string name = "Comm-";
      name += std::to_string(my_color);
      const char * commname = name.c_str();
      MPI_Comm_set_name(local_comm, commname);
      
      int split_rank;
      MPI_Group split_group;
      MPI_Comm_group(local_comm, &split_group);
      MPI_Comm_rank(local_comm, &split_rank);
      
      //Retrieve commname and print
      int rlen;
      char nameout[MPI_MAX_OBJECT_NAME];
      MPI_Comm_get_name(local_comm, nameout, &rlen);

      local_output << "rank: " << split_rank << " | comm: " << local_comm << " , comm_name: " << nameout << " , group: " << split_group << std::endl;
      
      local_output<<"On Refinement level "<<nrefine<<std::endl;
      
      double grid_size = 1.0 / ( std::pow( 2.0 , (double)nrefine ));
      Y = 2.0*s*std::fabs( std::log(grid_size) );
      num_eigen_pairs = Y / grid_size;

      pcout<<"    On refinement level "<<nrefine<<", h is "<<grid_size<<", Y is "<<Y<<", and N is "<<num_eigen_pairs<<std::endl;
      local_output<<"    On refinement level "<<nrefine<<", h is "<<grid_size<<", Y is "<<Y<<", and N is "<<num_eigen_pairs<<std::endl;
      
      // Compute the eigenpairs.
      for ( unsigned int j=0; j < num_eigen_pairs; j++ )  {
	if ( (j % nprocs) == my_id )  {
	  if( isSqrt ) {
	    etak = numbers::PI*( double(j) + 0.5 );
	    muk = (etak*etak)/(Y*Y);
	    vk = std::sqrt(2.0/Y);
	  }
	  else  {
	    etak = boost::math::cyl_bessel_j_zero( -s, j+1 );
	    muk = (etak*etak)/(Y*Y);
	    try  {
	      Ak = (std::sqrt(2.0)*std::cos( s * numbers::PI ))/ (std::pow( muk , 0.5*s) * Y * boost::math::cyl_bessel_j( 1.0 - s, etak));
	    }  catch( ... )  {
	      std::cerr<<"  Something bad happened in the call to boost::math::cyl_bessel_j_zero()."<<std::endl;
	    }
	    vk = ( std::pow(2.0,s) * Ak )/( std::cos( numbers::PI*s ) * std::tgamma( 1.0-s) );
	  }
	  evs.push_back(muk);
	  efs.push_back(vk);
	}
      }

      unsigned int local_num_eigen_pairs = evs.size();
      
      local_output << "Number of eigenpairs is "<<local_num_eigen_pairs<<std::endl;
      MPI_Barrier(mpi_communicator);
      start_setup = MPI_Wtime();

      // Now solve the associated Poisson problems.
      // After the first go through of the loop, we need to clean out the previous grid data.
      if (nrefine > starting_refinements) {
	triangulation.clear();
	dof_handler.clear();
	constraints.clear();
	system_matrix.clear();
	mass_matrix.clear();
	stiffness_matrix.clear();
      }
      
      make_grid(nrefine);
      setup_system();
      assemble_system();
      
      local_solution  = 0.0;
      global_solution = 0.0;

      end_setup = MPI_Wtime();
      local_output<<"Triangulation and grid setup time = "<<std::setprecision(8)<<(end_setup-start_setup)<<std::endl;
      
      unsigned int time_to_print = local_num_eigen_pairs/4;
      unsigned int counter = 0;
	
      
      // Now loop over all the eigenpairs this process is responsible for.
      double batch_start, batch_end;

      // Make sure we've all finished.
      MPI_Barrier(mpi_communicator);

      start_eigenpairs = MPI_Wtime();
      batch_start = MPI_Wtime();
	
      counter = 0;
      
      for (unsigned int nep = 0; nep < local_num_eigen_pairs; nep++) {

	if ( counter == time_to_print ) {
	  pcout<<" Working on eigen pair "<<nep<<" out of "<<local_num_eigen_pairs<<std::endl;
	  counter = 1;
	}
	else {
	  counter++;
	}

	system_matrix = 0.0;
	system_matrix.add(1.0 , stiffness_matrix);
	system_matrix.add( evs[nep] , mass_matrix);

	p_system_rhs = p_system_rhs_base;
	p_system_rhs *= ( ds * efs[nep] );

	p_system_rhs.compress(VectorOperation::add);
	p_solution.compress(VectorOperation::add);

	system_matrix.compress(VectorOperation::add);
	
	solve();

	local_solution.add(efs[nep] , solution);
      } // end loop over eigenpairs
      
      batch_end = MPI_Wtime();
      local_output<<std::setprecision(8)<<(batch_end-batch_start)<<std::endl;
      local_output<<"Loop over eigenpairs = "<<std::setprecision(8)<<(batch_end-start_eigenpairs)<<std::endl;
      
      // Make sure we've all finished.
      MPI_Barrier(mpi_communicator);
      
      stoptime_local = MPI_Wtime();
      local_output<<"Local runtime is "<<std::setprecision(8)<<(stoptime_local-starttime_local)<<" seconds, "<<std::endl;
      
      // Collect all local results and combine them to the global solution.
      Utilities::MPI::sum (local_solution, mpi_communicator, global_solution );

      // Make sure we've all finished.
      MPI_Barrier(mpi_communicator);

      // Let the root process output the solution.
      if ( Utilities::MPI::this_mpi_process(mpi_communicator)==0 ) {
	output_results(dof_handler.n_dofs());
      }

      Vector<double> difference_per_cell(triangulation.n_active_cells());
      double grid_spacing = 1.0 / ( std::pow( 2.0 , (double)nrefine ));
      
      const std::string single_mode    = "SM";
      const std::string multiple_modes = "MM";
      const std::string bessel         = "BE";

      RightHandSideSM<dim> right_hand_sideSM(s);
      RightHandSideMM<dim> right_hand_sideMM(s);
      RightHandSideBE<dim> right_hand_sideBE(s);

      pcout<<"  Integrating the differemce for L2 norm ..."<<std::endl;

      if ( source == single_mode )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionSM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::L2_norm);
      if ( source == multiple_modes )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionMM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::L2_norm);
      if ( source == bessel )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionBE<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::L2_norm);
	
      double l2_error =
	VectorTools::compute_global_error(triangulation,
					  difference_per_cell,
					  VectorTools::L2_norm);

      pcout<<"  Integrating the differemce for H1 norm ..."<<std::endl;

      if ( source == single_mode )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionSM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::H1_seminorm);
      if ( source == multiple_modes )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionMM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::H1_seminorm);
      if ( source == bessel )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionBE<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::H1_seminorm);

      double h1_error =
	VectorTools::compute_global_error(triangulation,
					  difference_per_cell,
					  VectorTools::H1_seminorm);
      
      pcout<<"  Integrating the differemce for |H_s norm ..."<<std::endl;

      if ( source == single_mode )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionSM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::mean,&right_hand_sideSM,1.0);
      if ( source == multiple_modes )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionMM<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::mean,&right_hand_sideMM,1.0);
      if ( source == bessel )
	VectorTools::integrate_difference(dof_handler,global_solution,SolutionBE<dim>(),difference_per_cell,QGauss<dim>(2),VectorTools::mean,&right_hand_sideBE,1.0);
      
      double hs_error =
	VectorTools::compute_global_error(triangulation,
					  difference_per_cell,
					  VectorTools::mean);
      
      hs_error = std::sqrt( std::fabs(hs_error) );
	
      const unsigned int n_active_cells = triangulation.n_active_cells();
      const unsigned int n_dofs         = dof_handler.n_dofs();
    
      convergence_table.add_value("h", grid_spacing);
      convergence_table.add_value("cells", n_active_cells);
      convergence_table.add_value("dofs", n_dofs);
      convergence_table.add_value("L2", l2_error);
      convergence_table.add_value("H1", h1_error);
      convergence_table.add_value("|H_s", hs_error);
      
      evs.clear();
      efs.clear();

      local_output<<std::endl; 
      
    } // End loop over refinement levels.

    local_output.close();
    
    convergence_table.set_precision("L2", 5);
    convergence_table.set_precision("H1", 5);
    convergence_table.set_precision("|H_s", 5);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("|H_s", true);

    convergence_table.set_tex_caption("h","\\# grid spacing");
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
    convergence_table.set_tex_caption("|H_s", "@f$mathbb{H}_s@f$-error");
    
    convergence_table.set_tex_format("h", "r");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
    
    if ( Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ) {

      pcout << std::endl;
      convergence_table.write_text(std::cout);
    
      std::string error_filename = "error.tex";
      std::ofstream error_table_file(error_filename);
      
      convergence_table.write_tex(error_table_file);
	
      std::string error_filename2 = "error.text";
      std::ofstream error_table_file2(error_filename2);
      const TableHandler::TextOutputFormat format = TableHandler::simple_table_with_separate_column_description;
      convergence_table.write_text(error_table_file2, format);
      
      convergence_table.add_column_to_supercolumn("cells", "n cells");

      std::vector<std::string> new_order;
      new_order.emplace_back("n cells");
      new_order.emplace_back("L2");
      new_order.emplace_back("H1");
      new_order.emplace_back("|H_s");
      convergence_table.set_column_order(new_order);
      
      convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates("|H_s", ConvergenceTable::reduction_rate);
      convergence_table.evaluate_convergence_rates("|H_s", ConvergenceTable::reduction_rate_log2);
      
      pcout << std::endl;
      convergence_table.write_text(std::cout);
      
      std::string conv_filename = "convergence.tex";
      std::ofstream table_file(conv_filename);
      convergence_table.write_tex(table_file);
	
      std::string   conv_filename2 = "convergence.text";
      std::ofstream table_file2(conv_filename2);
      convergence_table.write_text(table_file2, format);
    }
    endtime = MPI_Wtime();
    pcout<<std::endl;
    pcout<<"Runtime is "<<std::setprecision(8)<<(endtime-starttime)<<" seconds, "<<std::setprecision(8)<<((endtime-starttime)/60.0)<<" minutes, or ";
    pcout<<std::setprecision(8)<<((endtime-starttime)/3600.0)<<" hours."<<std::endl;
  }
} // namespace SpectralFractionalLaplacian

int main(int argc, char **argv)
{
  using namespace dealii;
  using namespace SpectralFractionalLaplacian;

  ParameterHandler prm;
  ParameterReader  param(prm);
  param.read_parameters("spectral-fractional-laplacian.prm");

  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    SpectralFractionalLaplace<2> problem(prm);
    problem.run();
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  std::cout << std::endl << "   Job done." << std::endl;
  return 0;
}
