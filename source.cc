#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm> // std::min_element
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/base/convergence_table.h>
#include <chrono>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <random>


using namespace dealii;
const double pi = numbers::PI;

namespace PrescribedSolution
{
    const int test_case = 8;
    /*
    test_case == 6:   For Linear Case
    test_case == 7:   Burger Equation_Convergence_Test
    test_case == 8:   KPP_Rotation_Problem
    test_case == 9:   KPP_Problem_Convergence_Test
    test_case == 10:  Buckley
    */
    namespace Transp_EG
    {
        template <int dim>
        class Exact_transp : public Function<dim>
        {
        public:
            Exact_transp ()
            :
            Function<dim>(2/*2 for EG*/)
            {
            }
            virtual double value(const Point<dim> & p,
                                const unsigned int component = 0) const override;
            virtual void vector_value(const Point<dim> &p,
                                        Vector<double> &values) const;
            virtual Tensor<1, dim> gradient(const Point<dim> & p,
                                            const unsigned int component = 0) const override;
        };
        template<int dim>
        void Exact_transp<dim>::vector_value(const Point<dim> &p,
                                            Vector<double> &values) const
        {
            for(unsigned int comp = 0; comp < this->n_components; ++comp)
                values(comp) = Exact_transp<dim>::value(p,comp);
        }
        template <int dim>
        double Exact_transp<dim>::value(const Point<dim> &p,
                                        const unsigned int /*component*/) const
        {
            const double t = this->get_time();
            double x = p(0);
            double y = p(1);
            double return_value = 0;
            if (test_case == 6)
            {
                return_value = cos(pi * (x-t)); 
            }else if(test_case == 7)
            {
                double x_0 = x;
                double iteration_value = 0;
                if(abs(x) < 1e-10 || abs(1. - x) < 1e-10) return_value = 0;
                else{
                    if(t < 1e-10)
                    {
                        return_value = sin(2*pi*x_0);
                    }else{
                        int max_iteration = 300;
                        int i = 0;
                        while( i < max_iteration)
                        {
                            iteration_value = x -  sin(2*pi*x_0 ) * t ;
                            x_0 = iteration_value;
                            i = i+1;
                        }
                        return_value = sin(2*pi*x_0);
                    }
                }
            }else if(test_case == 8)
            {
                if(pow(x,2.) + pow(y,2.) <=  1.)
                {
                    return_value = 7. * pi / 2.;
                }else{
                    return_value = pi / 4.;
                }
            }
            else if(test_case == 9)
            {
                {
                    if(pow(x,2.) + pow(y,2.) <= 1.)
                    {
                        return_value = pi/4. * (1. + 1./20. * (1 + cos(pi*sqrt(x*x+y*y))) );
                    }else{
                        return_value = pi/4.;
                    }
                }
            }else if(test_case == 10)
            {
                if(pow(x,2.) + pow(y,2.) < 0.5 )
                    return_value = 1;
                else
                    return_value = 0;
            }
            return return_value;

        }
        template <int dim>
        Tensor<1, dim> Exact_transp<dim>::gradient(const Point<dim> &p,
                                            const unsigned int) const
        {
            Tensor<1, dim> return_value;
            const double t = this->get_time();
            double x = p(0);
            if (test_case == 6)
            {
                return_value[0] = -sin(pi*(x-t))  * pi ;
                return_value[1] = 0;
            }
            return return_value;
        }




        
    };

};


template<int dim>
class ScalarHyperbolic
{
public:
    ScalarHyperbolic(double time_step, double tmax, int RefineTime,
                     bool bOutput, 
                     int task_number,
                     std::string task_name);
    double       time = 0;
    double       time_step;
    unsigned int timestep_number;
    double       tmax;
    int          RefineTime;
    bool         bOutput = false;
    bool         bDebug = true;
    double      tol = 1e-15;
    Triangulation<dim> triangulation;

    DoFHandler<dim> dof_handler_transp;
    AffineConstraints<double> constraints_transp;
    FESystem<dim>             fe_transp;
    //////////////////////////////////////////
    unsigned int  n_cg;
    unsigned int  n_dg;
    BlockSparsityPattern      sparsity_pattern_transp;

    Tensor<1,dim> Hyperbolic_flux(double u);
    Tensor<1,dim> Hyperbolic_derivative(double u);
    double Hyperbolic_max_wave_speed(double u_L, double u_R, Tensor<1,dim> n);
    double Hyperbolic_LLF(double u_L, double u_R, Tensor<1,dim> n);

    double entropy_square(double u);
    double entropy_square_variable(double u);
    Tensor<1,dim> entropy_q(double u);

    void run_KPP();

    void make_grid_KPP();
    void setup_system();
    void Compute_EG_Solution();
        BlockVector<double>       EG_solution;
    void Error();
        double   L1_error_transp = 0;
        double   L2_error_transp = 0.;
        double   sup_L1_error_transp = 0;
        double   sup_L2_error_transp = 0;
    void Output();

    BlockVector<double> EulerUpdate(BlockVector<double> &solution_input,int stage);
        BlockVector<double>       solution_transp_EG;
        BlockVector<double>       old_solution_transp_EG;
        BlockVector<double>       old_old_solution_transp_EG;
        BlockVector<double>       loworder_solution_transp_EG;

    void Assemble_Mass_Matrix();
        BlockSparseMatrix<double>  mass_matrix_transp;

    void Low_High_Approx_u(BlockVector<double> &solution_input,int stage);
        BlockSparseMatrix<double> low_high_matrix_cg;
    void Low_High_Approx_U(BlockVector<double> &solution_input,int stage);
        FullMatrix<double> low_matrix_average;
        FullMatrix<double> high_matrix_average;
    

    void Limiting_Condition_Check();

    void MCL_BP_Limiting_u(BlockVector<double> &solution_input);
    void FCT_BP_Limiting_u(BlockVector<double> &solution_input);
    
        BlockSparseMatrix<double> bp_max_min_flux_matrix_cg;
        BlockSparseMatrix<double> bp_flux_limiter_matrix_cg;
    void MCL_BP_Limiting_U(BlockVector<double> &solution_input);
    void FCT_BP_Limiting_U(BlockVector<double> &solution_input);
        FullMatrix<double> bp_max_matrix_average;
        FullMatrix<double> bp_min_matrix_average;
        FullMatrix<double> bp_flux_limiter_matrix_average;

    void ES_Limiting_u(BlockVector<double> &solution_input);
        BlockSparseMatrix<double> es_P_Q_flux_matrix_cg;
        BlockSparseMatrix<double> es_flux_limiter_matrix_cg;
        Vector<double> entropy_production_u;
        
    void ES_Limiting_U(BlockVector<double> &solution_input);
        FullMatrix<double> es_P_matrix_average;
        FullMatrix<double> es_Q_matrix_average;
        FullMatrix<double> es_flux_limiter_matrix_average;
        Vector<double> entropy_production_U;

    void Clip_and_Scale_u_and_U();
        BlockSparseMatrix<double> clip_and_scale_flux_limiter_matrix_cg;
        FullMatrix<double> clip_and_scale_flux_limiter_matrix_average;
    void Extra_ES_Limiting_cg(BlockVector<double> &solution_input);
        Vector<double> extra_es_limiting_P_cg;
        Vector<double> extra_es_limiting_Q_cg;
        Vector<double> extra_es_limiter_cg;

        

    void Construct_LowOrder_Solution(BlockVector<double> &solution_input);
    BlockVector<double> Construct_Solution(BlockVector<double> &solution_input);

    void FluxLimiter_Output();

    
    void FCR_Process();
    BlockVector<double>       ustar_solution;

    // BlockVector<double>       solution_transp_EG_low_order;
    // BlockVector<double>       old_solution_transp_EG_low_order;
    // BlockVector<double>       old_old_solution_transp_EG_low_order;

    BlockVector<double>       CG_Average;

    int task_number;
    std::string task_name;


    /*DataOutput dof_handler*/
    DoFHandler<dim>   dof_handler_DGQ0;
    const FE_DGQ<dim> fe_DGQ0;

    DoFHandler<dim>   dof_handler_DGQ1;
    const FE_DGQ<dim> fe_DGQ1;


    double   max_cell_dia = 0;
};

template<int dim>
ScalarHyperbolic<dim>::ScalarHyperbolic(double time_step, double tmax, int RefineTime,
                                        bool bOutput, 
                                        int task_number,
                                        std::string task_name)
                    : time_step(time_step), tmax(tmax), RefineTime(RefineTime)
                    , bOutput(bOutput)
                    , dof_handler_transp(triangulation)
                    , fe_transp(FESystem<dim>(FE_Q<dim>(1),1, FE_DGQ<dim>(0), 1 ))
                    , task_number(task_number)
                    , task_name(task_name)
                    /*The following is for DataOutput*/
                    , dof_handler_DGQ0(triangulation)
                    , fe_DGQ0(0)
                    , dof_handler_DGQ1(triangulation)
                    , fe_DGQ1(1)
                    {

                    }

template<int dim>
void ScalarHyperbolic<dim>::make_grid_KPP()
{
    if(PrescribedSolution::test_case == 6 || PrescribedSolution::test_case == 7)
    {
        Point<dim> p0(0.,0);
        Point<dim> p1(1, 1);
        GridGenerator::hyper_rectangle(triangulation, p0, p1);
    }
    else if(PrescribedSolution::test_case == 8 || PrescribedSolution::test_case == 9)
    {
        Point<dim> p0(-2.,-2.5);
        Point<dim> p1(2., 1.5);
        GridGenerator::hyper_rectangle(triangulation, p0, p1);
    }else if(PrescribedSolution::test_case == 10)
    {
        Point<dim> p0(-1.5,-1.5);
        Point<dim> p1( 1.5, 1.5);
        GridGenerator::hyper_rectangle(triangulation, p0, p1);
    }

    triangulation.refine_global(RefineTime);
    std::cout << "\tNumber of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "\tTotal number of cells: " << triangulation.n_cells()
              << std::endl;
    if(bOutput)
    {
        std::ofstream mesh_file("mesh.gnuplot");
        GridOut().write_gnuplot(triangulation, mesh_file);
    }
    max_cell_dia   = GridTools::maximal_cell_diameter(triangulation);
    std::cout << "____________make_grid_KPP()___________" << std::endl;
}


template<int dim>
void ScalarHyperbolic<dim>::setup_system()
{
    dof_handler_transp.distribute_dofs(fe_transp);  
    DoFRenumbering::component_wise(dof_handler_transp);
    const std::vector<types::global_dof_index> 
        dofs_per_component = DoFTools::count_dofs_per_fe_component(dof_handler_transp);

    n_cg = dofs_per_component[0];
    n_dg = dofs_per_component[1];
    std::cout << "\tn_cg = " << n_cg << ",\t n_dg = " << n_dg << std::endl;
    constraints_transp.clear();
    PrescribedSolution::Transp_EG::Exact_transp<dim> exact_transp;
        exact_transp.set_time(time);
    DoFTools::make_hanging_node_constraints(dof_handler_transp, constraints_transp);
    VectorTools::interpolate_boundary_values(dof_handler_transp,
                                                 0,
                                                 exact_transp,
                                                 constraints_transp);
    constraints_transp.close();
    
    BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit(n_cg, n_cg);
        dsp.block(1, 0).reinit(n_dg, n_cg);
        dsp.block(0, 1).reinit(n_cg, n_dg);
        dsp.block(1, 1).reinit(n_dg, n_dg);
        dsp.collect_sizes();
    DoFTools::make_sparsity_pattern(dof_handler_transp, dsp);
        sparsity_pattern_transp.copy_from(dsp);
    {
        std::map<types::global_dof_index, Point<dim>> dof_location_map;
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                                dof_handler_transp, 
                                                dof_location_map);
    
        std::ofstream out("sparsity_pattern_EG.svg");
        sparsity_pattern_transp.print_svg(out);

        // std::ofstream out_fct("sparsity-pattern-EG-fct.svg");
        // sparsity_pattern_transp_EG_fct.print_svg(out_fct);

        std::ofstream dof_location_file("dof_transp_EG.gnuplot");
        DoFTools::write_gnuplot_dof_support_point_info(dof_location_file, dof_location_map);
    }
    mass_matrix_transp.reinit(sparsity_pattern_transp);

    low_high_matrix_cg.reinit(sparsity_pattern_transp);
    bp_max_min_flux_matrix_cg.reinit(sparsity_pattern_transp);
    bp_flux_limiter_matrix_cg.reinit(sparsity_pattern_transp);
    es_P_Q_flux_matrix_cg.reinit(sparsity_pattern_transp);
    es_flux_limiter_matrix_cg.reinit(sparsity_pattern_transp);

    solution_transp_EG.reinit(2);
            solution_transp_EG.block(0).reinit(n_cg);
            solution_transp_EG.block(1).reinit(n_dg);
            solution_transp_EG.collect_sizes();

    old_solution_transp_EG.reinit(2);
        old_solution_transp_EG.block(0).reinit(n_cg);
        old_solution_transp_EG.block(1).reinit(n_dg);
        old_solution_transp_EG.collect_sizes();

    old_old_solution_transp_EG.reinit(2);
        old_old_solution_transp_EG.block(0).reinit(n_cg);
        old_old_solution_transp_EG.block(1).reinit(n_dg);
        old_old_solution_transp_EG.collect_sizes();
    EG_solution.reinit(2);
            EG_solution.block(0).reinit(n_cg);
            EG_solution.block(1).reinit(n_dg);
            EG_solution.collect_sizes();
    ustar_solution.reinit(2);
        ustar_solution.block(0).reinit(n_cg);
        ustar_solution.block(1).reinit(n_dg);
        ustar_solution.collect_sizes();
    // ustar_solution.reinit(n_cg);

    CG_Average.reinit(2);
            CG_Average.block(0).reinit(n_cg);
            CG_Average.block(1).reinit(n_dg);
            CG_Average.collect_sizes();  

    loworder_solution_transp_EG.reinit(2);
            loworder_solution_transp_EG.block(0).reinit(n_cg);
            loworder_solution_transp_EG.block(1).reinit(n_dg);
            loworder_solution_transp_EG.collect_sizes();  

    high_matrix_average = FullMatrix<double>(n_dg, 4);
    low_matrix_average  = FullMatrix<double>( n_dg, 4);
    bp_max_matrix_average = FullMatrix<double>(n_dg, 4);
    bp_min_matrix_average = FullMatrix<double>(n_dg, 4);

    bp_flux_limiter_matrix_average = FullMatrix<double>(n_dg, 4);
    bp_flux_limiter_matrix_average = 1;

    es_P_matrix_average = FullMatrix<double>(n_dg, 4);
    es_Q_matrix_average = FullMatrix<double>(n_dg, 4);
    es_flux_limiter_matrix_average = FullMatrix<double>(n_dg, 4);
    es_flux_limiter_matrix_average = 1;

    clip_and_scale_flux_limiter_matrix_cg.reinit(sparsity_pattern_transp);
    clip_and_scale_flux_limiter_matrix_average = FullMatrix<double>(n_dg, 4);
    clip_and_scale_flux_limiter_matrix_cg = 1;
    // std::cout << "____________setup_system()___________" << std::endl;
}

template<int dim>
void ScalarHyperbolic<dim>::Assemble_Mass_Matrix()
{
    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    const unsigned int 
        dofs_per_cell = fe_transp.n_dofs_per_cell();
    FullMatrix<double> 
        cell_matrix_mass     (dofs_per_cell, dofs_per_cell); /*m^e_{ij}*/
    Vector<double>     
        m_ei(dofs_per_cell);
    std::vector<types::global_dof_index> 
        local_dof_indices     (dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        m_ei = 0;
        cell_matrix_mass = 0;
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for(const unsigned int i : fe_values.dof_indices())
        {
            for(const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                /*m^e_i*/
                m_ei(i) += fe_values.shape_value(i,q_index)
                            * fe_values.JxW(q_index);
            }
            // add m_ei into m_i (the local contribution)
            mass_matrix_transp.add(local_dof_indices[i], local_dof_indices[i], m_ei(i));
            //// put m_e_i into mass_matrix_transp_EG;
            // when i = 4, m_e_i(4) = the area of cell
            // and we use .set(4,4) to store this value 
            mass_matrix_transp.set(local_dof_indices[i], local_dof_indices[4], m_ei(i));
        }/*End of i-loop*/
    }/*End of cell-loop*/

    // mass_matrix_transp.block(0,1).print_formatted(std::cout);
    // std::cout << std::endl;
    // mass_matrix_transp.block(1,1).print_formatted(std::cout);
    //  std::cout << "____________Assemble_Mass_Matrix()___________" << std::endl;
}

template<int dim> 
Tensor<1,dim> ScalarHyperbolic<dim>::Hyperbolic_flux(double u)
{
    Tensor<1,dim> return_value;
    if(PrescribedSolution::test_case == 6)
    {
        return_value[0] = u;
        return_value[1] = 0.;
    }else if(PrescribedSolution::test_case == 7)
    {
        return_value[0] = pow(u,2.)/2.;
        return_value[1] = 0.;
    }
    else if(PrescribedSolution::test_case == 8 || PrescribedSolution::test_case == 9)
    {
        return_value[0] = sin(u);
        return_value[1] = cos(u);
    }else if(PrescribedSolution::test_case == 10)
    {
        double coeff = pow(u,2.) / (pow(u,2.) + pow(1-u,2.));
        return_value[0] = 1. * coeff;
        return_value[1] = (1 - 5 * pow(1-u,2.)) * coeff ;
    }
    return return_value;
}

template<int dim>
Tensor<1,dim> ScalarHyperbolic<dim>::Hyperbolic_derivative(double u)
{
    Tensor<1,dim> return_value;
    if(PrescribedSolution::test_case == 6)
    {
        return_value[0] = 1.;
        return_value[1] = 0.;
    }else if(PrescribedSolution::test_case == 7)
    {
        return_value[0] = u;
        return_value[1] = 0.;
    }
    else if(PrescribedSolution::test_case == 8 || PrescribedSolution::test_case == 9)
    {
        return_value[0] = cos(u);
        return_value[1] = -sin(u);
    }else if(PrescribedSolution::test_case == 10)
    {
        double coeff1 = pow(u,2.) / (pow(u,2.) + pow(1-u,2.));
        double coeff2 = (-2. * pow(u,2.) + 2 * u) / pow(2*u*u-2*u + 1., 2.);
        return_value[0] = coeff2;
        return_value[1] = coeff2 * (1 - 5 * pow(1-u,2.)) + coeff1 * 10. * (1-u) ;
    }
    
    return return_value;
}

template<int dim>
double ScalarHyperbolic<dim>::Hyperbolic_max_wave_speed(double u_L, double u_R, Tensor<1,dim> n)
{
    double return_value = 1;
    if(PrescribedSolution::test_case == 6)
    {
        return_value = 1;
    }else if(PrescribedSolution::test_case == 7)
    {
        Tensor<1,dim> unit;
        unit[0] = 1;
        unit[1] = 0;
        return_value = std::max(abs(u_L), abs(u_R)) * std::abs(unit * n);
    }
    else if(PrescribedSolution::test_case == 8 || PrescribedSolution::test_case == 9)
    {
        return_value = 1;
    }else if(PrescribedSolution::test_case == 10 )
    {
        return_value = 3.4;
    }
    return return_value;
}

template<int dim>
double ScalarHyperbolic<dim>::Hyperbolic_LLF(double u_L, double u_R, Tensor<1,dim> n)
{
    double return_value = 0;
    Tensor<1,dim> F_L = Hyperbolic_flux(u_L);
    Tensor<1,dim> F_R = Hyperbolic_flux(u_R);
    double max_wave_speed = Hyperbolic_max_wave_speed(u_L, u_R, n);
    return_value = 0.5 *(F_L + F_R ) * n;
    return_value -= 0.5 * (u_R - u_L) * max_wave_speed  ;
    return return_value;
}

template<int dim>
double ScalarHyperbolic<dim>::entropy_square(double u)
{
    double return_value;
    return_value = pow(u,2.) / 2.;
    if(PrescribedSolution::test_case == 7)
    {
        return_value = pow(u, 4.)/ 4.;
    }
    return return_value;
}

template<int dim>
double ScalarHyperbolic<dim>::entropy_square_variable(double u)
{
    double return_value;
    return_value = u;
    if(PrescribedSolution::test_case == 7)
    {
        return_value = pow(u, 3.);
    }
    return return_value;
}

template<int dim>
Tensor<1,dim> ScalarHyperbolic<dim>::entropy_q(double u)
{
    Tensor<1,dim> return_value;
    if(PrescribedSolution::test_case == 6)
    {
        return_value[0] =  pow(u,2.) / 2.;
        return_value[1] = 0.;
    }else if(PrescribedSolution::test_case == 7)
    {
        return_value[0] = pow(u,5.)/5. ;
        return_value[1] = 0.;
    }
    else if(PrescribedSolution::test_case == 8 || PrescribedSolution::test_case == 9)
    {
        return_value[0] = u * sin(u) + cos(u);
        return_value[1] = u * cos(u) - sin(u);
    }
   
    return return_value;
}

template<int dim>
void ScalarHyperbolic<dim>::Low_High_Approx_u(BlockVector<double> &solution_input,int stage)
{

    PrescribedSolution::Transp_EG::Exact_transp<dim>  exact_transp;
    if(stage == 1)     exact_transp.set_time(time - time_step);
    else if(stage == 2)exact_transp.set_time(time );

    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    QGauss<dim-1>   face_quadrature_formula(fe_transp.degree + 5);
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe_transp,
                                        face_quadrature_formula, 
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor (fe_transp,
                                                face_quadrature_formula,
                                                update_values | update_gradients |
                                                update_quadrature_points  |
                                                update_normal_vectors |
                                                update_JxW_values);
    const FEValuesExtractors::Scalar cg(0);
    const FEValuesExtractors::Scalar dg(0);  
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    FullMatrix<double> cell_matrix_mass     (dofs_per_cell, dofs_per_cell); /*m^e_{ij}*/
    FullMatrix<double> cell_matrix_d_cg     (dofs_per_cell, dofs_per_cell); /*d^e_{ij}*/      

    FullMatrix<double> cell_matrix_lambda(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_adv_comp0      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/
    FullMatrix<double> cell_matrix_adv_comp1      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/

    Vector<double>     m_ei(dofs_per_cell);
    Vector<double>     g_L(dofs_per_cell);
    Vector<double>     g_L_B(dofs_per_cell);

    Vector<double>     g_H(dofs_per_cell);
    Vector<double>     g_H_B(dofs_per_cell);

    // Vector<double>     f_ei(dofs_per_cell);
    std::vector<double>  solution_values_transp_cg(n_q_points);
    std::vector<double>  solution_values_face_transp_cg(n_face_q_points);
    std::vector<double>  solution_values_face_neighbor_transp_cg(n_face_q_points);

    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_face(dofs_per_cell);

    low_high_matrix_cg = 0;

    std::map<types::global_dof_index, Point<dim> > support_points;
    DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
                                          dof_handler_transp,
                                          support_points);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    if(timestep_number == 1 && stage == 1)
    {
        CG_Average.block(1) = solution_input.block(1);
    }else{
        for(; cell!=endc; ++cell)
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values[cg].get_function_values(solution_input,
                                                solution_values_transp_cg);
            double cell_area = mass_matrix_transp(local_dof_indices[4],
                                                local_dof_indices[4]);
            double average_value = 0;
            for(const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                double u_cg_q = solution_values_transp_cg[q_index];
                average_value += u_cg_q * fe_values.JxW(q_index);
            }
            CG_Average(local_dof_indices[4]) = average_value / cell_area;
        }  /*End of cell-loop*/   
    }
    

    BlockVector<double> udot_solution;
    udot_solution.reinit(2);
    udot_solution.block(0).reinit(n_cg);
    udot_solution.block(1).reinit(n_dg);
    udot_solution.collect_sizes();
    std::vector<Vector<double>>  solution_values_transp(n_q_points, Vector<double>(2));
    FullMatrix<double> udot_face_contribution(n_cg+n_dg, 4);
    cell = dof_handler_transp.begin_active();
    for(; cell !=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; 
            ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                double cg_bar_cell = CG_Average(local_dof_indices[4]);
                double eg_bar_cell = solution_input(local_dof_indices[4]);

                fe_face_values.reinit(cell, face_number);
                fe_face_values[cg].get_function_values(solution_input,
                                                       solution_values_face_transp_cg);
                
                
                for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                {
                    const auto &x_q = fe_face_values.quadrature_point(q_index);
                    double u_cg_h_q = solution_values_face_transp_cg[q_index];
                    double normal_velocity = fe_face_values.normal_vector(q_index) *
                                            Hyperbolic_derivative(u_cg_h_q);
                    if(normal_velocity <= 0)
                    {
                        double in_flow_value = exact_transp.value(x_q);
                        // 
                        double dg_cell = eg_bar_cell - cg_bar_cell;
                        udot_face_contribution(local_dof_indices[4],face_number) += 
                                                                - Hyperbolic_LLF(u_cg_h_q+dg_cell, in_flow_value, fe_face_values.normal_vector(q_index))
                                                                * fe_face_values.JxW(q_index);
                    }else{
                        double dg_cell = eg_bar_cell - cg_bar_cell;
                        
                        udot_face_contribution(local_dof_indices[4],face_number) += 
                                                                // - (u_cg_h_q /*+ dg_cell*/)
                                                                // - (u_cg_h_q + dg_cell)
                                                                // -exact_transp.value(x_q) 
                                                                - Hyperbolic_LLF(u_cg_h_q+dg_cell, u_cg_h_q+dg_cell, fe_face_values.normal_vector(q_index))
                                                            // * normal_velocity
                                                            * fe_face_values.JxW(q_index);
                    }
                }
            }/*End of boundary-face*/
            else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                    cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4];
                unsigned int e_neighbor = local_dof_indices_face[4];
                // double eg_bar_cell     = solution_transp_EG_low_order(local_dof_indices[4]);
                // double cg_bar_cell     = CG_Average(local_dof_indices[4]);
                
                // double cg_bar_neighbor = CG_Average(local_dof_indices_face[4]);
                // double eg_bar_neighbor = solution_transp_EG_low_order(local_dof_indices_face[4]);
                double eg_bar_cell     = solution_input(local_dof_indices[4]);
                double cg_bar_cell     = CG_Average(local_dof_indices[4]);
                
                double eg_bar_neighbor = solution_input(local_dof_indices_face[4]);
                double cg_bar_neighbor = CG_Average(local_dof_indices_face[4]);
                double dg_cell     = eg_bar_cell - cg_bar_cell;
                double dg_neighbor = eg_bar_neighbor - cg_bar_neighbor;

                fe_face_values.reinit(cell, face_number);
                fe_face_values[cg].get_function_values(
                    solution_input,
                    solution_values_face_transp_cg);
                double normal_velocity = 0;
                if(cell->face(face_number)->user_flag_set() == false)
                {
                    cell->face(face_number)->set_user_flag();
                    for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                    {
                        // const auto &x_q = fe_face_values.quadrature_point(q_index);
                        double u_cg_h_q = solution_values_face_transp_cg[q_index];
                        normal_velocity = Hyperbolic_derivative(u_cg_h_q) * fe_face_values.normal_vector(q_index);
                        if(normal_velocity <= 0)
                        {
                            udot_face_contribution(e, face_number) += 
                                                        // - (u_cg_h_q )
                                                        // - (u_cg_h_q + dg_neighbor)
                                                        - Hyperbolic_LLF(u_cg_h_q+dg_cell, u_cg_h_q+dg_neighbor, fe_face_values.normal_vector(q_index))
                                                        // -exact_transp.value(x_q) 
                // - exact_value
                                                    // * normal_velocity
                                                    * fe_face_values.JxW(q_index);
                        }else{
                            udot_face_contribution(e, face_number) += 
                                                        // - (u_cg_h_q )
                                                        // - (u_cg_h_q + dg_cell)
                                                        // -exact_transp.value(x_q) 
                                                        - Hyperbolic_LLF(u_cg_h_q+dg_cell, u_cg_h_q+dg_neighbor, fe_face_values.normal_vector(q_index))
                // - exact_value
                                                    // * normal_velocity
                                                    * fe_face_values.JxW(q_index);
                        }
                    }
                }else{
                    udot_face_contribution(e, face_number) = -udot_face_contribution(e_neighbor, neighbor_face);
                                                            
                }
            }/*Inner face*/
        }/*End of face-loop*/
    }/*End of cell-loop*/
    triangulation.clear_user_flags();

    for(unsigned int i = 0; i < n_cg; ++i)
    {
        double m_i = mass_matrix_transp(i,i);

        IndexSet DOF(dof_handler_transp.n_dofs());
        BlockSparseMatrix<double>::const_iterator
                index = mass_matrix_transp.begin(i);
        BlockSparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.end(i);
        for(; index!=index_end; ++index)
        {
            unsigned int e = index->column();
            if(e >= n_cg) DOF.add_index(e);
        }

        for(auto e : DOF)
        {
            double cell_area = mass_matrix_transp(e,e);
            // double value = udot_solution(e);
            double value = 0;
            for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
                ++face_number)
            {
                value += udot_face_contribution(e, face_number);
            }
            if(sparsity_pattern_transp.exists(i,e))
            {
                double m_ei = mass_matrix_transp(i,e);
                udot_solution(i) += (1. / m_i) * (m_ei / cell_area) * value; 
            }
        }
    }
    // std::cout << "\t\tCompute: udot_solution()\n";

    cell = dof_handler_transp.begin_active();
    for(; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        
        cell->get_dof_indices(local_dof_indices);


        cell_matrix_adv_comp0 = 0;
        cell_matrix_adv_comp1 = 0;
        cell_matrix_lambda    = 0;

        cell_matrix_mass = 0;   // -> for m_e_ij
        cell_matrix_d_cg = 0;   // -> for d_e_ij
        g_L   = 0;
        g_L_B = 0;
        
        g_H   = 0;
        g_H_B = 0;

        for(unsigned int i  = 0; i < 4; ++i)
        {
            for(unsigned int j  = 0; j < 4; ++j)
            {
                for(const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    // m_e_ij
                    cell_matrix_mass(i,j) += fe_values.shape_value(j, q_index)
                                            * fe_values.shape_value(i, q_index)
                                            * fe_values.JxW(q_index);
                    
                    // Calculate c_e_ij, x-component
                    cell_matrix_adv_comp0(i,j) += fe_values.shape_grad(j, q_index)[0] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);
                    // Calculate c_e_ij, y-component
                    cell_matrix_adv_comp1(i,j) += fe_values.shape_grad(j, q_index)[1] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);     

                }/*End of quadrature*/

                // if(i < 4 && j < 4)
                {
                    Tensor<1,dim> n_ij;
                    n_ij[0] = cell_matrix_adv_comp0(i,j);
                    n_ij[1] = cell_matrix_adv_comp1(i,j);
                    n_ij = n_ij / n_ij.norm();
                    double u_i = solution_input(local_dof_indices[i]);
                    double u_j = solution_input(local_dof_indices[j]);
                    
                    cell_matrix_lambda(i,j) = Hyperbolic_max_wave_speed(u_i, u_j, n_ij);
                }

            }/*End of j-loop*/
        }/*End of i-loop*/
        for(unsigned int i  = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double lambda_e_ij = cell_matrix_lambda(i,j);
                    
                    Tensor<1,dim> c_e_ji;
                    c_e_ji[0] = cell_matrix_adv_comp0(j,i);
                    c_e_ji[1] = cell_matrix_adv_comp1(j,i);
                    double lambda_e_ji = cell_matrix_lambda(j,i);

                    cell_matrix_d_cg(i,j) = std::max(lambda_e_ij * c_e_ij.norm(),
                                                     lambda_e_ji * c_e_ji.norm());
                    cell_matrix_d_cg(i,i) += -cell_matrix_d_cg(i,j);
                }
            }
        }

        for(unsigned int i = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double d_e_ij = cell_matrix_d_cg(i,j);
                    double u_i = solution_input(local_dof_indices[i]);
                    double u_j = solution_input(local_dof_indices[j]);

                    Tensor<1,dim> F_j = Hyperbolic_flux(u_j);
                    Tensor<1,dim> F_i = Hyperbolic_flux(u_i);
                    
                    
                    g_L(i) += 
                        - d_e_ij * (u_i - u_j)
                        + (F_i - F_j) * c_e_ij ;

                    double m_e_ij = cell_matrix_mass(i,j);
                    double udot_i = udot_solution(local_dof_indices[i]);
                    double udot_j = udot_solution(local_dof_indices[j]);

                    g_H(i) +=  (F_i - F_j) * c_e_ij +
                            m_e_ij * (udot_i - udot_j);
                }
            }
        }

        unsigned int e = local_dof_indices[4];
        double eg_average_cell = solution_input(e);
        double cg_average_cell = CG_Average(e);
        fe_values[cg].get_function_values(solution_input,
                                            solution_values_transp_cg);
        for(unsigned int i = 0; i < 4; ++i)
        {
            for(const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                // const auto &x_q = fe_values.quadrature_point(q_index);
                g_H(i) += (eg_average_cell - cg_average_cell)
                          * fe_values.shape_grad(i, q_index)
                          * Hyperbolic_derivative(solution_values_transp_cg[q_index])
                          * fe_values.JxW(q_index);
                
            }
        }

        

        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                double normal_velocity = 0;

                fe_face_values.reinit(cell, face_number);
                fe_face_values[cg].get_function_values(solution_input, 
                                                       solution_values_face_transp_cg);
                
                unsigned int e = local_dof_indices[4];
                double eg_average_cell = solution_input(e);
                double cg_average_cell = CG_Average(e);
                for(unsigned int i = 0; i < 4; ++i)
                {
                
                    double b_i_face = 0;
                    double uhat_i = 0;
                    double u_i = solution_input[local_dof_indices[i]];
                    for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                    {
                        const auto &x_q = fe_face_values.quadrature_point(q_index);
                        uhat_i += fe_face_values[cg].value(i,q_index) *
                                    exact_transp.value(x_q) *
                                    fe_face_values.JxW(q_index);
                        b_i_face += fe_face_values[cg].value(i,q_index) *
                                    fe_face_values.JxW(q_index);
                    }
                    if( abs(b_i_face) > 1e-8 ) uhat_i = uhat_i / b_i_face;
                    Tensor<1,dim> F_i = Hyperbolic_flux(u_i);
                    normal_velocity = Hyperbolic_derivative(u_i) * fe_face_values.normal_vector(0);
                    if(normal_velocity < -1e-8)
                    {
                        // g_L(i) += b_i_face * 
                        //           0.5 *
                        //           (u_i - uhat_i) ;
                        // g_L(i) += b_i_face * 
                        //           (uhat_i - u_i); 
                        g_L_B(i) += b_i_face * F_i * fe_face_values.normal_vector(0);
                        // g_L(i) -= b_i_face * LLF_Linear(u_i, uhat_i, fe_face_values.normal_vector(0));
                        g_L_B(i) -= b_i_face * Hyperbolic_LLF(u_i, uhat_i, fe_face_values.normal_vector(0));
                    }
                    // if(b_i_face > 1e-8)
                    {

                        for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                        {
                            const auto &x_q = fe_face_values.quadrature_point(q_index);
                            double uh_q = solution_values_face_transp_cg[q_index];
                            // double U = solution_input[local_dof_indices[4]];
                            double uhat_q = 0;
                            normal_velocity = 
                                fe_face_values.normal_vector(q_index) * Hyperbolic_derivative(uh_q);
                            
                            if(normal_velocity > 1e-8)
                            {
                                g_H_B(i) += -(eg_average_cell - cg_average_cell) *
                                            fe_face_values[cg].value(i, q_index) * 
                                            Hyperbolic_derivative(uh_q) *
                                            fe_face_values.normal_vector(q_index) *
                                            fe_face_values.JxW(q_index);
                            }else if(normal_velocity < -1e-8){
                                uhat_q = exact_transp.value(x_q);
                                // uhat_q = 1.;
                                double lambda_q = 1;
                                // std::max(abs(uhat_q), abs(uh_q)) * 
                                        //    abs(velocity * fe_face_values.normal_vector(q_index));
                                for(unsigned int j = 0; j < 4 ; ++j)
                                {
                                    double u_j = solution_input(local_dof_indices[j]);
                                    Tensor<1,dim> F_j = Hyperbolic_flux(u_j);
                                    g_H_B(i) +=  0.5 * F_j * 
                                                fe_face_values[cg].value(i,q_index) *
                                                fe_face_values[cg].value(j, q_index) * 
                                                fe_face_values.normal_vector(q_index) * 
                                                fe_face_values.JxW(q_index);
                                }
                                g_H_B(i) += -fe_face_values[cg].value(i, q_index) *
                                            // 0.5 * pow(uhat_q, 2.) / 2. * velocity *
                                            0.5 * Hyperbolic_flux(uhat_q) *
                                            fe_face_values.normal_vector(q_index) *
                                            fe_face_values.JxW(q_index);
                                
                                g_H_B(i) +=  0.5 *lambda_q  * fe_face_values[cg].value(i, q_index) *
                                            (uhat_q - uh_q) 
                                            * fe_face_values.JxW(q_index);

                                g_H_B(i) += - (eg_average_cell - cg_average_cell) *
                                            fe_face_values[cg].value(i, q_index) * 
                                            // 0.5 * velocity * uhat_q * 
                                            0.5 * Hyperbolic_derivative(uhat_q) *
                                            fe_face_values.normal_vector(q_index) *
                                            fe_face_values.JxW(q_index);
                                g_H_B(i) +=  -(eg_average_cell - cg_average_cell) *
                                                0.5 *lambda_q * 
                                            fe_face_values[cg].value(i, q_index) *
                                            fe_face_values.JxW(q_index);
                            }

                        }/*End of quadratrure*/
                    }
                
                }/*End of i-loop*/
            }/*End of boundary face*/
        }/*End of faces-loop*/
        for(unsigned int i = 0; i < 4; ++i)
        {
            // f_ei(i) = g_H(i) - g_L(i) 
            // + g_H_B(i) 
            // - g_L_B(i);
            // the correction flux for cg component are stored into system matrix_EG
            low_high_matrix_cg.set(local_dof_indices[i], local_dof_indices[4], g_H(i) + g_H_B(i)); 
            low_high_matrix_cg.set(local_dof_indices[4], local_dof_indices[i], g_L(i) + g_L_B(i)); 
            
        }
    }/*End of cell-loop*/
    // std::cout << "\t\tCompute low and high order approximations\n";
    // std::cout << "____________Low_High_Approx_u(BlockVector<double> solution_input)_____\n";
}


template<int dim>
void ScalarHyperbolic<dim>::Low_High_Approx_U(BlockVector<double> &solution_input,int stage)
{
    PrescribedSolution::Transp_EG::Exact_transp<dim>  exact_transp;
    if(stage == 1)     exact_transp.set_time(time - time_step);
    else if(stage == 2)exact_transp.set_time(time );
       
    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    QGauss<dim-1>   face_quadrature_formula(fe_transp.degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe_transp,
                                        face_quadrature_formula, 
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor (fe_transp,
                                                face_quadrature_formula,
                                                update_values | update_gradients |
                                                update_quadrature_points  |
                                                update_normal_vectors |
                                                update_JxW_values);
    const FEValuesExtractors::Scalar cg(0);
    const FEValuesExtractors::Scalar dg(0);  
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();


    std::vector<double>  solution_values_face_transp_cg(n_face_q_points);

    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_face(dofs_per_cell);

    std::vector<double>  solution_values_transp_cg(n_q_points);

    low_matrix_average =  0;
    high_matrix_average = 0;
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();

    if(timestep_number == 1 && stage == 1)
    {
        CG_Average.block(1) = solution_input.block(1);
    }else{
        for(; cell!=endc; ++cell)
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            fe_values[cg].get_function_values(solution_input,
                                                solution_values_transp_cg);
            double cell_area = mass_matrix_transp(local_dof_indices[4],
                                                local_dof_indices[4]);
            double average_value = 0;
            for(const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                double u_cg_q = solution_values_transp_cg[q_index];
                average_value += u_cg_q * fe_values.JxW(q_index);
            }
            CG_Average(local_dof_indices[4]) = average_value / cell_area;
        }  /*End of cell-loop*/   
    }

    cell  = dof_handler_transp.begin_active();
    for(; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        unsigned int e = local_dof_indices[4];
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            double gbar_L = 0;
            double gbar_H = 0;
            if(cell->at_boundary(face_number))
            {
                fe_face_values.reinit(cell, face_number);
                fe_face_values[cg].get_function_values(solution_input, 
                                                       solution_values_face_transp_cg);
                double eg_bar_cell     = solution_input(local_dof_indices[4]);
                double cg_bar_cell     = CG_Average(local_dof_indices[4]);
                double dg_cell = eg_bar_cell - cg_bar_cell;
                for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                {
                    const auto &x_q = fe_face_values.quadrature_point(q_index);
                    double Uhat_e = 0;
                    double u_cg_h_q = solution_values_face_transp_cg[q_index];
                    double U_e = eg_bar_cell;
                    double normal_velocity = Hyperbolic_derivative(U_e) * fe_face_values.normal_vector(q_index);
                    if(normal_velocity < 0)
                    {
                        double in_flow_value = exact_transp.value(x_q);
                        Uhat_e = in_flow_value;
                        gbar_L += - Hyperbolic_LLF(U_e, Uhat_e, fe_face_values.normal_vector(q_index))
                                * fe_face_values.JxW(q_index);
                        gbar_H += - Hyperbolic_LLF(u_cg_h_q+dg_cell, in_flow_value, fe_face_values.normal_vector(q_index))
                                * fe_face_values.JxW(q_index);
                    }else{
                        // double in_flow_value = exact_transp.value(x_q);
                        Uhat_e = U_e;
                        gbar_L += - Hyperbolic_LLF(U_e, Uhat_e, fe_face_values.normal_vector(q_index))
                                * fe_face_values.JxW(q_index);
                        gbar_H += - Hyperbolic_LLF(u_cg_h_q+dg_cell, u_cg_h_q+dg_cell, fe_face_values.normal_vector(q_index))
                                * fe_face_values.JxW(q_index);
                    }
                    
                    
                }
            }/*End of boundary face*/
            else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                fe_face_values.reinit(cell, face_number);
                fe_face_values_neighbor.reinit(neighbor, neighbor_face);
                fe_face_values[cg].get_function_values(solution_input,  
                                                       solution_values_face_transp_cg);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4];
                unsigned int e_neighbor = local_dof_indices_face[4];

                double eg_bar_cell     = solution_input(e);
                double cg_bar_cell     = CG_Average(e);
                
                double eg_bar_neighbor = solution_input(e_neighbor);
                double cg_bar_neighbor = CG_Average(e_neighbor);
                if(cell->face(face_number)->user_flag_set() == false)
                {
                    cell->face(face_number)->set_user_flag();
                    for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
                    {
                        // const auto &x_q = fe_face_values.quadrature_point(q_index);
                        
                        double Uhat_e = eg_bar_neighbor;
                        double U_e = eg_bar_cell;

                        gbar_L += - Hyperbolic_LLF(U_e, Uhat_e, fe_face_values.normal_vector(q_index))
                                        * fe_face_values.JxW(q_index);
                        
                        double u_cg_h_q = solution_values_face_transp_cg[q_index];

                        
                        double dg_cell     = eg_bar_cell - cg_bar_cell;
                        double dg_neighbor = eg_bar_neighbor - cg_bar_neighbor;
                        gbar_H += - Hyperbolic_LLF(u_cg_h_q+dg_cell, u_cg_h_q+dg_neighbor, fe_face_values.normal_vector(q_index))
                                * fe_face_values.JxW(q_index);
                    }/*End of quadrature*/
                }
                else{
                    gbar_L = -low_matrix_average(e_neighbor - n_cg, neighbor_face);
                    gbar_H = -high_matrix_average(e_neighbor - n_cg, neighbor_face);
                }
            }/*End of inner face*/
            low_matrix_average(e - n_cg, face_number) =  gbar_L;
            high_matrix_average(e- n_cg, face_number) =  gbar_H;
        }/*End of face-loop*/
    }
    triangulation.clear_user_flags();
    // std::cout << "____________Low_High_Approx_U(BlockVector<double> solution_input)_____\n";
}


template<int dim> 
void ScalarHyperbolic<dim>::Limiting_Condition_Check()
{
    BlockSparseMatrix<double> raw_correction_flux;
    raw_correction_flux.reinit(sparsity_pattern_transp);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(0,1).end(i);
        for(; index!=index_end; ++index)
        {

            unsigned int e = index->column();
            DOF.add_index(e+n_cg);
        }
        for(auto e : DOF)
        {
            double high_approx = low_high_matrix_cg(i,e);
            double low_approx = low_high_matrix_cg(e,i);
            raw_correction_flux.set(i, e, high_approx - low_approx);
        }
    }/*End of i-loop: assemble raw_correction_flux*/
    
    dof_handler_DGQ0.distribute_dofs(fe_DGQ0);
    Vector<double> Zero_Sum_Check;
    Zero_Sum_Check.reinit(dof_handler_DGQ0.n_dofs());


    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    const unsigned int dofs_per_cell_DGQ0 = fe_DGQ0.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices_DGQ0(dofs_per_cell_DGQ0);


    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        cell0  = dof_handler_DGQ0.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell != endc; ++cell, ++cell0 )
    {
        cell->get_dof_indices(local_dof_indices);
        cell0->get_dof_indices(local_dof_indices_DGQ0);
        double temp = 0;
        unsigned int e = local_dof_indices[4];
        for(unsigned int i = 0; i < 4; ++i)
        {
            temp += raw_correction_flux(local_dof_indices[i],e);
        }
        Zero_Sum_Check(local_dof_indices_DGQ0[0]) = temp;
    }
    DataOut<dim> data_out_DGQ0;
    data_out_DGQ0.attach_dof_handler(dof_handler_DGQ0);
    std::vector<std::string> Zero_Sum_Name;
    Zero_Sum_Name.push_back("ZeroSum");
    data_out_DGQ0.add_data_vector(Zero_Sum_Check, Zero_Sum_Name);
    data_out_DGQ0.build_patches();
    data_out_DGQ0.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    const std::string filename_average =
            task_name + 
            Utilities::int_to_string(RefineTime,1)+
            "_Zero_Sum_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_average(filename_average);
        data_out_DGQ0.write_vtk(output_average);
}

template<int dim>
void ScalarHyperbolic<dim>::MCL_BP_Limiting_u(BlockVector<double> &solution_input)
{
    BlockSparseMatrix<double> raw_correction_flux;
    raw_correction_flux.reinit(sparsity_pattern_transp);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(0,1).end(i);
        for(; index!=index_end; ++index)
        {

            unsigned int e = index->column();
            DOF.add_index(e+n_cg);
        }
        for(auto e : DOF)
        {
            double high_approx = low_high_matrix_cg(i,e);
            double low_approx = low_high_matrix_cg(e,i);
            raw_correction_flux.set(i, e, high_approx - low_approx);
        }
    }/*End of i-loop: assemble raw_correction_flux*/
    
    Vector<double> MCL_max_u;
    Vector<double> MCL_min_u;
    MCL_max_u.reinit(n_cg);
    MCL_min_u.reinit(n_cg);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        BlockSparseMatrix<double>::const_iterator 
            index = mass_matrix_transp.begin(i);
        BlockSparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.end(i);
        std::vector<double> u_input_solution;
        for(; index!=index_end; ++index)
        {
            unsigned int j = index->column();
            u_input_solution.push_back(solution_input(j));
        }
        auto   result_input_max = std::max_element(std::begin(u_input_solution), 
                                                 std::end(u_input_solution));
        double input_max = *result_input_max;
        ////////////////////
        auto   result_input_min = std::min_element(std::begin(u_input_solution), 
                                                 std::end(u_input_solution));
        double input_min = *result_input_min;
        ////////////////////
        MCL_max_u(i) = input_max;
        MCL_min_u(i) = input_min;
    }/*End of i-loop: Find MCL_max and MCL_min*/
    

    bp_max_min_flux_matrix_cg = 0;
    bp_flux_limiter_matrix_cg = 1;
    /*
    Fillin 
        bp_max_min_flux_matrix_cg
        bp_flux_limiter_matrix_cg
    */
    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    FullMatrix<double> cell_matrix_d_cg     (dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_lambda(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_adv_comp0      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/
    FullMatrix<double> cell_matrix_adv_comp1      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/
    FullMatrix<double> cell_matrix_bar_state      (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        cell_matrix_adv_comp0 = 0; // -> c_e_ij, the first  component
        cell_matrix_adv_comp1 = 0; // -> c_e_ij, the second component
        cell_matrix_lambda    = 0;
        cell_matrix_bar_state = 0;
        for(unsigned int i = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                for(const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    // Calculate c_e_ij, x-component
                    cell_matrix_adv_comp0(i,j) += fe_values.shape_grad(j, q_index)[0] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);
                    // Calculate c_e_ij, y-component
                    cell_matrix_adv_comp1(i,j) += fe_values.shape_grad(j, q_index)[1] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);     
                }/*End of quadrature*/
                Tensor<1,dim> n_ij;
                        n_ij[0] = cell_matrix_adv_comp0(i,j);
                        n_ij[1] = cell_matrix_adv_comp1(i,j);
                        n_ij = n_ij / n_ij.norm();
                        double u_i = solution_input(local_dof_indices[i]);
                        double u_j = solution_input(local_dof_indices[j]);
                        
                        cell_matrix_lambda(i,j) = Hyperbolic_max_wave_speed(u_i, u_j,n_ij);
            }
        }/*End of i-loop*/
        for(unsigned int i  = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double lambda_e_ij = cell_matrix_lambda(i,j);
                    
                    Tensor<1,dim> c_e_ji;
                    c_e_ji[0] = cell_matrix_adv_comp0(j,i);
                    c_e_ji[1] = cell_matrix_adv_comp1(j,i);
                    double lambda_e_ji = cell_matrix_lambda(j,i);

                    cell_matrix_d_cg(i,j) = std::max(lambda_e_ij * c_e_ij.norm(),
                                                     lambda_e_ji * c_e_ji.norm());
                    cell_matrix_d_cg(i,i) += -cell_matrix_d_cg(i,j);
                }/*Only for CG-components*/
            }/*End of j-loop*/
        }/*End of i-loop*/
        for(unsigned int i = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                double d_e_ij = cell_matrix_d_cg(i,j);
                double u_i = solution_input(local_dof_indices[i]);
                double u_j = solution_input(local_dof_indices[j]);
                Tensor<1,dim> F_j = Hyperbolic_flux(u_j);
                Tensor<1,dim> F_i = Hyperbolic_flux(u_i);
                cell_matrix_bar_state (i,j) = 
                    0.5 * (u_j + u_i) - (F_j - F_i) * c_e_ij / (2. * d_e_ij);
            }
            
        }
        for(unsigned int i = 0; i < 4; ++i)
        {
            double d_bar_e_i = 0;
            double u_bar_state_e_i = 0;
            for(unsigned int j = 0; j < 4; ++j)
            {
                if(j!=i)
                {
                    double d_e_ij = cell_matrix_d_cg(i,j);
                    d_bar_e_i += d_e_ij;
                
                    double u_bar_state_e_ij = cell_matrix_bar_state(i,j);
                    u_bar_state_e_i += d_e_ij * u_bar_state_e_ij;

                }
            }
            u_bar_state_e_i = u_bar_state_e_i / d_bar_e_i;
            // CG_bar_state.set(local_dof_indices[4], local_dof_indices[i], d_bar_e_i);
            // CG_bar_state.set(local_dof_indices[i], local_dof_indices[4], u_bar_state_e_i);
            bp_max_min_flux_matrix_cg.set(local_dof_indices[i],
                                          local_dof_indices[4], 
                                          (MCL_max_u(local_dof_indices[i]) - u_bar_state_e_i) * 2 * d_bar_e_i );
            bp_max_min_flux_matrix_cg.set(local_dof_indices[4],
                                          local_dof_indices[i], 
                                          (MCL_min_u(local_dof_indices[i]) - u_bar_state_e_i) * 2 * d_bar_e_i );
            
        }
        for(unsigned int i = 0; i < 4; ++i)
        {
            double f_max_ei = bp_max_min_flux_matrix_cg(local_dof_indices[i],local_dof_indices[4]);
            double f_min_ei = bp_max_min_flux_matrix_cg(local_dof_indices[4],local_dof_indices[i]);
            double f_ei     = raw_correction_flux(local_dof_indices[i],local_dof_indices[4]);
            double beta_bp_ei = 1;

            if(f_ei >= tol)        beta_bp_ei = std::min(1., f_max_ei / (f_ei ) );
            else if (f_ei <= -tol) beta_bp_ei = std::min(1., f_min_ei / (f_ei ) );
            else if (f_ei < tol && f_ei > -tol) beta_bp_ei = 1.;

            bp_flux_limiter_matrix_cg.set(local_dof_indices[i],
                                          local_dof_indices[4],
                                          beta_bp_ei);
        }

        if(cell->at_boundary())
        {
            for(unsigned int i = 0; i < 4; ++i)
                bp_flux_limiter_matrix_cg.set(local_dof_indices[i],
                                          local_dof_indices[4],
                                          1);
        }
    }  /*End of cell-loop*/ 
    
    // std::cout << "____________MCL_BP_Limiting_u(solution_input)_____\n";
}


// template<int dim>
// void ScalarHyperbolic<dim>::FCT_BP_Limiting_u(BlockVector<double> &solution_input)
// {
//     loworder_solution_transp_EG = solution_input;
//     for(unsigned int i = 0; i < n_cg;++i)
//     {
//         SparseMatrix<double>::const_iterator 
//             index = mass_matrix_transp.block(0,1).begin(i);
//         SparseMatrix<double>::const_iterator 
//             index_end = mass_matrix_transp.block(0,1).end(i);
//         double m_i = mass_matrix_transp(i,i);
//         double CFL = time_step / m_i;
//         for(; index!=index_end; ++index)
//         {
//             unsigned int e_support = index->column() + n_cg;
//             loworder_solution_transp_EG(i) += CFL * low_high_matrix_cg(e_support, i);
//         }   
//     }
//     for(unsigned int e = 0; e < n_dg; ++e)
//     {
//         unsigned int e_dof = e + n_cg;
//         double m_e = mass_matrix_transp(e_dof,e_dof);
//         double CFL = time_step / m_e;
//         for(unsigned int face_number = 0; face_number < 4; ++face_number)
//             loworder_solution_transp_EG(e_dof) += CFL * low_matrix_average(e,face_number);
//     }
//     if(bOutput)
//     {
//       DataOut<dim> data_out;
//       std::vector<std::string> EG_names;
//       EG_names.push_back("u_c");
//       EG_names.push_back("ubar");
//       data_out.attach_dof_handler(dof_handler_transp);
//       data_out.add_data_vector(loworder_solution_transp_EG, EG_names);
//       data_out.build_patches();
//       data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
//       const std::string filename = 
//         // Utilities::int_to_string(task_number, 1)
//         task_name
//         +Utilities::int_to_string(RefineTime,1)
//         +"_LowOrder_TestCase_" + std::to_string(PrescribedSolution::test_case)
//         +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
//       std::ofstream output(filename);
//       data_out.write_vtk(output);
//     }

//     BlockSparseMatrix<double> raw_correction_flux;
//     raw_correction_flux.reinit(sparsity_pattern_transp);
//     for(unsigned int i = 0; i < n_cg; ++i)
//     {
//         IndexSet DOF(dof_handler_transp.n_dofs());
//         SparseMatrix<double>::const_iterator
//             index = mass_matrix_transp.block(0,1).begin(i);
//         SparseMatrix<double>::const_iterator
//             index_end = mass_matrix_transp.block(0,1).end(i);
//         for(; index!=index_end; ++index)
//         {

//             unsigned int e = index->column();
//             DOF.add_index(e+n_cg);
//         }
//         for(auto e : DOF)
//         {
//             double high_approx = low_high_matrix_cg(i,e);
//             double low_approx = low_high_matrix_cg(e,i);
//             raw_correction_flux.set(i, e, high_approx - low_approx);
//         }
//     }/*End of i-loop: assemble raw_correction_flux*/
    
//     Vector<double> FCT_max_u;
//     Vector<double> FCT_min_u;
//     FCT_max_u.reinit(n_cg);
//     FCT_min_u.reinit(n_cg);
//     for(unsigned int i = 0; i < n_cg; ++i)
//     {
//         BlockSparseMatrix<double>::const_iterator 
//             index = mass_matrix_transp.begin(i);
//         BlockSparseMatrix<double>::const_iterator
//             index_end = mass_matrix_transp.end(i);
//         std::vector<double> u_input_solution;
//         std::vector<double> u_c_low_solution;
//         IndexSet DOF(dof_handler_transp.n_dofs());

//         for(; index!=index_end; ++index)
//         {
//             unsigned int j = index->column();
//             DOF.add_index(j);
//         }
//         for(auto k : DOF)
//         {
//             u_input_solution.push_back(solution_input(k));
//             if(k < n_cg) {
//                 u_c_low_solution.push_back(loworder_solution_transp_EG(k));
//             }
//         }
//         auto   result_input_min = std::min_element(std::begin(u_input_solution), 
//                                                  std::end(u_input_solution));
//         double input_min = *result_input_min;
//         ////////////////////
//         auto   result_input_max = std::max_element(std::begin(u_input_solution), 
//                                                    std::end(u_input_solution));
//         double input_max = *result_input_max;
//         ////////////////////
//         auto   result_low_min = std::min_element(std::begin(u_c_low_solution), 
//                                                std::end(u_c_low_solution));
//         double low_min = *result_low_min;
//         // ////////////////////
//         auto   result_low_max = std::max_element(std::begin(u_c_low_solution), 
//                                                std::end(u_c_low_solution));
//         double low_max = *result_low_max;
//         ////////////////////
//         // std::cout << "input_max = " << input_max <<std::endl;
//         // std::cout << "low_max = " << low_max <<std::endl;
//         // std::cout << "input_max = " << input_min <<std::endl;
//         // std::cout << "low_max = " << low_min <<std::endl;
//         // std::cout << std::endl;
//         FCT_max_u(i) = std::max(input_max, low_max);
//         FCT_min_u(i) = std::min(input_min, low_min);
//         // FCT_max_u(i) = input_max;
//         // FCT_min_u(i) = input_min;
        
//     }/*End of i-loop: Find MCL_max and MCL_min*/
    

//     bp_max_min_flux_matrix_cg = 0;
//     bp_flux_limiter_matrix_cg = 1;
//     /*
//     Fillin 
//         bp_max_min_flux_matrix_cg
//         bp_flux_limiter_matrix_cg
//     */
//     QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
//     FEValues<dim>     fe_values(fe_transp,
//                                 quadrature_formula,
//                                 update_values | update_gradients |
//                                 update_quadrature_points |
//                                 update_JxW_values);
//     const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
//     std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
//     typename DoFHandler<dim>::active_cell_iterator
//         cell  = dof_handler_transp.begin_active(),
//         endc  = dof_handler_transp.end();
//     for(; cell!=endc; ++cell)
//     {
//         fe_values.reinit(cell);
//         cell->get_dof_indices(local_dof_indices);
        
//         for(unsigned int i = 0; i < 4; ++i)
//         {
//             double m_ei = mass_matrix_transp(local_dof_indices[i],local_dof_indices[4]);
//             double f_max_i = 
//                 m_ei / time_step * 
//                 (FCT_max_u(local_dof_indices[i]) - loworder_solution_transp_EG(local_dof_indices[i]));
//             // if(f_max_i < 0.) std::cout << "f_max_i < 0\n";
//             // std::cout << "lowordersolution = " << loworder_solution_transp_EG(local_dof_indices[i]) << std::endl;
//             f_max_i = std::max(f_max_i, 0.);
//             bp_max_min_flux_matrix_cg.set(local_dof_indices[i],
//                                           local_dof_indices[4], 
//                                           f_max_i);

//             double f_min_i = 
//                 m_ei / time_step * 
//                 (FCT_min_u(local_dof_indices[i]) - loworder_solution_transp_EG(local_dof_indices[i]));
//             bp_max_min_flux_matrix_cg.set(local_dof_indices[4],
//                                           local_dof_indices[i], 
//                                           f_min_i);
//             // if(f_min_i > 0.) std::cout << "f_max_i > 0\n";
//             f_min_i = std::min(f_min_i, 0.);
            
//         // }
//         // for(unsigned int i = 0; i < 4; ++i)
//         // {
//             // double f_max_ei = bp_max_min_flux_matrix_cg(local_dof_indices[i],local_dof_indices[4]);
//             // double f_min_ei = bp_max_min_flux_matrix_cg(local_dof_indices[4],local_dof_indices[i]);
//             double f_ei     = raw_correction_flux(local_dof_indices[i],local_dof_indices[4]);
//             double beta_bp_ei = 1;

//             if(f_ei >= tol)        beta_bp_ei = std::min(1., f_max_ei / (f_ei  ) );
//             else if (f_ei <= -tol) beta_bp_ei = std::min(1., f_min_ei / (f_ei ) );
//             else if (f_ei < tol && f_ei > -tol) beta_bp_ei = 1.;

//             bp_flux_limiter_matrix_cg.set(local_dof_indices[i],
//                                           local_dof_indices[4],
//                                           beta_bp_ei);
//         }

//         if(cell->at_boundary())
//         {
//             for(unsigned int i = 0; i < 4; ++i)
//                 bp_flux_limiter_matrix_cg.set(local_dof_indices[i],
//                                           local_dof_indices[4],
//                                           1);
//         }


//         for(unsigned int i = 0; i < 4; ++i)
//         {
//             double f_max_ei = bp_max_min_flux_matrix_cg(local_dof_indices[i],local_dof_indices[4]);
//             double f_min_ei = bp_max_min_flux_matrix_cg(local_dof_indices[4],local_dof_indices[i]);
//             double f_ei     = raw_correction_flux(local_dof_indices[i],local_dof_indices[4]);
//             double beta_bp_ei = bp_flux_limiter_matrix_cg(local_dof_indices[i],local_dof_indices[4]);

//             // if(beta_bp_ei * f_ei > f_max_ei || beta_bp_ei * f_ei < f_min_ei) 
//             // {
//             //     std::cout << "f_max_ei = " << f_max_ei << std::endl;
//             //     std::cout << "f_min_ei = " << f_min_ei << std::endl;
//             //     std::cout << "f_ei     = " << f_ei << std::endl;
//             //     std::cout << "beta_bp_ei =" << beta_bp_ei << std::endl;
//             //     std::cout <<std::endl; 
//             // }
//         }
//     }  /*End of cell-loop*/ 
    
//     // std::cout << "____________MCL_BP_Limiting_u(solution_input)_____\n";
// }




template<int dim>
void ScalarHyperbolic<dim>::MCL_BP_Limiting_U(BlockVector<double> &solution_input)
{
    bp_flux_limiter_matrix_average= 1;
    FullMatrix<double> raw_correction_flux;
    raw_correction_flux = FullMatrix<double>(n_dg, 4);
    for(unsigned int e = 0; e < n_dg; ++e)
    {
        for(unsigned int face_number = 0; face_number < 4; ++face_number)
        {
            raw_correction_flux(e, face_number) 
                    = high_matrix_average(e, face_number) 
                    - low_matrix_average(e, face_number);
        }
    }

    Vector<double> MCL_max_U;
    Vector<double> MCL_min_U;
    MCL_max_U.reinit(n_dg);
    MCL_min_U.reinit(n_dg);
    for(unsigned int e = 0; e < n_dg; ++e)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator 
            index = mass_matrix_transp.block(1,0).begin(e);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(1,0).end(e);
        for(; index!=index_end; ++index)
        {
            unsigned int j = index->column();
            DOF.add_index(j);
            SparseMatrix<double>::const_iterator 
                index_j = mass_matrix_transp.block(0,1).begin(j);
            SparseMatrix<double>::const_iterator
                index_j_end = mass_matrix_transp.block(0,1).end(j);
            for(; index_j!=index_j_end; ++index_j)
            {
                unsigned int j_support = index_j->column();
                DOF.add_index(j_support+n_cg);
            }
        }
        std::vector<double> u_input_solution;
        for(auto k : DOF)
        {
            u_input_solution.push_back(solution_input(k));
        }
        auto result_input_max = std::max_element(std::begin(u_input_solution), 
                                                 std::end(u_input_solution));
        double input_max = *result_input_max;
        ///////////
        auto   result_input_min = std::min_element(std::begin(u_input_solution), 
                                                   std::end(u_input_solution));
        double input_min = *result_input_min;
        MCL_max_U(e) = input_max;
        MCL_min_U(e) = input_min;
        
    }

    

    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    QGauss<dim-1>   face_quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe_transp,
                                        face_quadrature_formula, 
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor (fe_transp,
                                                face_quadrature_formula,
                                                update_values | update_gradients |
                                                update_quadrature_points  |
                                                update_normal_vectors |
                                                update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();                                 
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_face(dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(;cell!=endc; ++cell)
    {
        Vector<double> face_length(GeometryInfo<dim>::faces_per_cell);
        double legenth_of_boundary_of_cell = 0;
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            fe_face_values.reinit(cell, face_number);
            for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
            {
                face_length(face_number) += fe_face_values.JxW(q_index);
            }
            legenth_of_boundary_of_cell += face_length(face_number);
        }
        
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                fe_face_values.reinit(cell, face_number);
                cell->get_dof_indices(local_dof_indices);
                unsigned int e = local_dof_indices[4] - n_cg;
                // std::cout << "e = " << e << std::endl;
                double U_bar_state = solution_input(e + n_cg);
                double max_wave_speed = Hyperbolic_max_wave_speed(U_bar_state, U_bar_state, fe_face_values.normal_vector(0));
                bp_max_matrix_average(e,face_number) = max_wave_speed * face_length(face_number) * 
                                          std::max(0. , 
                                                   MCL_max_U(e) - U_bar_state
                                                   );
                bp_min_matrix_average(e,face_number) = max_wave_speed * face_length(face_number) * 
                                          std::min(0. , 
                                                   MCL_min_U(e) - U_bar_state
                                                   );
            }else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                // const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                fe_face_values.reinit(cell, face_number);
                unsigned int e = local_dof_indices[4]- n_cg;
                unsigned int e_neighbor = local_dof_indices_face[4] - n_cg;
                double U_bar = solution_input(e+n_cg);
                double U_bar_neighbor = solution_input(e_neighbor+n_cg);
                
                double max_wave_speed = Hyperbolic_max_wave_speed(U_bar, U_bar_neighbor, fe_face_values.normal_vector(0));
                double U_bar_state = 0.5 * (U_bar + U_bar_neighbor)
                                     - fe_face_values.normal_vector(0) *
                                      (Hyperbolic_flux(U_bar_neighbor) - Hyperbolic_flux(U_bar)) / (2. * max_wave_speed);
                bp_max_matrix_average(e,face_number) 
                        = max_wave_speed * face_length(face_number) * 
                                          std::max(0. , 
                                                   std::min(MCL_max_U(e) - U_bar_state, 
                                                            U_bar_state - MCL_min_U(e_neighbor)  )
                                                   );
                bp_min_matrix_average(e,face_number) 
                        = max_wave_speed * face_length(face_number) * 
                                          std::min(0. , 
                                                   std::max(MCL_min_U(e) - U_bar_state, 
                                                            U_bar_state - MCL_max_U(e_neighbor)  )
                                                   );

            }
        }
    }/*End of cell-loop*/


    cell             = dof_handler_transp.begin_active();
    for(; cell!=endc; ++cell)
    {

        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            cell->get_dof_indices(local_dof_indices);
            unsigned int e = local_dof_indices[4] - n_cg;
            double F_ee_neighbor = raw_correction_flux(e, face_number);
            bp_flux_limiter_matrix_average(e, face_number) = 1.;
            if(cell->at_boundary(face_number))
            {
                bp_flux_limiter_matrix_average(e, face_number )  = 1;
            }else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4]- n_cg;
                unsigned int e_neighbor = local_dof_indices_face[4] - n_cg;

                if(cell->face(face_number)->user_flag_set() == false)
                {
                    cell->face(face_number)->set_user_flag();
                    if(F_ee_neighbor > tol)
                    {
                        double F_ee_neighbor_max = bp_max_matrix_average(e,face_number); 
                        if(F_ee_neighbor_max > tol)
                            bp_flux_limiter_matrix_average(e, face_number ) 
                                = std::min(1.,(F_ee_neighbor_max ) / (F_ee_neighbor)); 
                        else bp_flux_limiter_matrix_average(e, face_number )  = 1;
                        
                    }else if(F_ee_neighbor < -tol)
                    {
                        double F_ee_neighbor_min = bp_min_matrix_average(e,face_number);
                        if(F_ee_neighbor_min < -tol)
                            bp_flux_limiter_matrix_average(e, face_number ) 
                                = std::min(1.,(F_ee_neighbor_min ) /(F_ee_neighbor ));  
                        else bp_flux_limiter_matrix_average(e, face_number )  = 1;
                    }
                    else{
                        bp_flux_limiter_matrix_average(e, face_number )  = 1;
                    }
                }else{
                    bp_flux_limiter_matrix_average(e, face_number) = bp_flux_limiter_matrix_average(e_neighbor, neighbor_face);
                }
            }
        }/*face-loop*/
    }/*End of cell-loop*/
    // bp_flux_limiter_matrix_average.print_formatted(std::cout);
    // bp_flux_limiter_matrix_average = 1;
    // std::cout << "____________MCL_BP_Limiting_U(solution_input)_____\n";
    triangulation.clear_user_flags();
    
}

template<int dim>
void ScalarHyperbolic<dim>::FCT_BP_Limiting_U(BlockVector<double> &solution_input)
{
    bp_flux_limiter_matrix_average= 1;
    FullMatrix<double> raw_correction_flux;
    raw_correction_flux = FullMatrix<double>(n_dg, 4);
    for(unsigned int e = 0; e < n_dg; ++e)
    {
        for(unsigned int face_number = 0; face_number < 4; ++face_number)
        {
            raw_correction_flux(e, face_number) 
                    = high_matrix_average(e, face_number) 
                    - low_matrix_average(e, face_number);
        }
    }

    BlockVector<double> FCT_max;
    BlockVector<double> FCT_min;
    FCT_max.reinit(2);
    FCT_max.block(0).reinit(n_cg);
    FCT_max.block(1).reinit(n_dg);
    FCT_max.collect_sizes();

    FCT_min.reinit(2);
    FCT_min.block(0).reinit(n_cg);
    FCT_min.block(1).reinit(n_dg);
    FCT_min.collect_sizes();
    for(unsigned int e = n_cg; e < n_cg + n_dg; e++)
    {
        BlockSparseMatrix<double>::const_iterator 
            index = mass_matrix_transp.begin(e);
        BlockSparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.end(e);

        IndexSet DOF(dof_handler_transp.n_dofs());
        
        for(; index!=index_end; ++index)
        {
            unsigned int j = index->column();
            if(j < n_cg)
            {
                DOF.add_index(j);
                BlockSparseMatrix<double>::const_iterator 
                    index_j = mass_matrix_transp.begin(j);
                BlockSparseMatrix<double>::const_iterator
                    index_j_end = mass_matrix_transp.end(j);
                for(; index_j!=index_j_end; ++index_j)
                {
                    unsigned int j2 = index_j->column();
                    if(j2 >= n_cg) DOF.add_index(j2);
                    
                }
            }
        }
        std::vector<double> u_input_solution;
        std::vector<double> ubar_low_solution;
        for(auto k : DOF)
        {
            u_input_solution.push_back(solution_input(k));
            if(k >= n_cg) ubar_low_solution.push_back(loworder_solution_transp_EG(k));
        }
        ////////////////
        auto   result_input_min = std::min_element(std::begin(u_input_solution), 
                                                   std::end(u_input_solution));
        double input_min = *result_input_min;
        ////////////////
        auto result_low_min = std::min_element(std::begin(ubar_low_solution), 
                                               std::end(ubar_low_solution));
        double low_min = *result_low_min;
        ////////////////
        auto result_input_max = std::max_element(std::begin(u_input_solution), 
                                                 std::end(u_input_solution));
        double input_max = *result_input_max;
        ////////////////
        auto result_low_max = std::max_element(std::begin(ubar_low_solution), 
                                               std::end(ubar_low_solution));
        double low_max = *result_low_max;
        ////////////////
        FCT_max(e) = std::max(input_max, low_max);
        FCT_min(e) = std::min(input_min, low_min);
        
    }

    

    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    QGauss<dim-1>   face_quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe_transp,
                                        face_quadrature_formula, 
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor (fe_transp,
                                                face_quadrature_formula,
                                                update_values | update_gradients |
                                                update_quadrature_points  |
                                                update_normal_vectors |
                                                update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();                                 
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_face(dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(;cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        unsigned int e = local_dof_indices[4];
        Vector<double> face_length(GeometryInfo<dim>::faces_per_cell);
        double legenth_of_boundary_of_cell = 0;
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            fe_face_values.reinit(cell, face_number);
            for(const unsigned int q_index : fe_face_values.quadrature_point_indices())
            {
                face_length(face_number) += fe_face_values.JxW(q_index);
            }
            legenth_of_boundary_of_cell += face_length(face_number);
        }
        double cell_area = mass_matrix_transp(local_dof_indices[4],
                                                        local_dof_indices[4]);

        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                bp_max_matrix_average(e-n_cg,face_number) = cell_area / time_step * face_length(face_number) / legenth_of_boundary_of_cell *
                    std::max(0., FCT_max(e)-loworder_solution_transp_EG(e) );
                bp_min_matrix_average(e-n_cg,face_number) = cell_area / time_step * face_length(face_number) / legenth_of_boundary_of_cell *
                    std::min(0., FCT_min(e)-loworder_solution_transp_EG(e) );
            }else{
                bp_max_matrix_average(e-n_cg,face_number) = cell_area / time_step * face_length(face_number) / legenth_of_boundary_of_cell *
                    std::max(0., 
                             std::min(FCT_max(e) - loworder_solution_transp_EG(e),
                                      loworder_solution_transp_EG(e) - FCT_min(e))
                    );
                bp_min_matrix_average(e-n_cg,face_number) = cell_area / time_step * face_length(face_number) / legenth_of_boundary_of_cell *
                    std::min(0., 
                             std::max(FCT_min(e) - loworder_solution_transp_EG(e),
                                      loworder_solution_transp_EG(e) - FCT_max(e))
                    );
            }
        }
    }/*End of cell-loop*/


    cell             = dof_handler_transp.begin_active();
    for(; cell!=endc; ++cell)
    {

        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            cell->get_dof_indices(local_dof_indices);
            unsigned int e = local_dof_indices[4] - n_cg;
            double F_ee_neighbor = raw_correction_flux(e, face_number);
            bp_flux_limiter_matrix_average(e, face_number) = 1.;
            if(cell->at_boundary(face_number))
            {
                bp_flux_limiter_matrix_average(e, face_number )  = 1;
                if(F_ee_neighbor > 0. )
                {
                    bp_flux_limiter_matrix_average(e, face_number) 
                        = std::min(1.,bp_max_matrix_average(e,face_number) /F_ee_neighbor);
                }else if(F_ee_neighbor < 0.)
                {
                    bp_flux_limiter_matrix_average(e, face_number) 
                        = std::min(1.,bp_min_matrix_average(e,face_number) /F_ee_neighbor);
                }
            }else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4]- n_cg;
                unsigned int e_neighbor = local_dof_indices_face[4] - n_cg;
                if(cell->face(face_number)->user_flag_set() == false)
                {
                    cell->face(face_number)->set_user_flag();
                    if(F_ee_neighbor > 0.)
                    {
                        bp_flux_limiter_matrix_average(e, face_number ) 
                                = std::min(1.,bp_max_matrix_average(e, face_number )  / (F_ee_neighbor)); 
                    }else if(F_ee_neighbor < 0.)
                    {
                        bp_flux_limiter_matrix_average(e, face_number ) 
                                = std::min(1.,bp_min_matrix_average(e, face_number )  / (F_ee_neighbor)); 
                    }

                }else{
                    bp_flux_limiter_matrix_average(e, face_number) = bp_flux_limiter_matrix_average(e_neighbor, neighbor_face);
                }
            }
        }/*face-loop*/
    }/*End of cell-loop*/
    // bp_flux_limiter_matrix_average.print_formatted(std::cout);
    // bp_flux_limiter_matrix_average = 1;
    // std::cout << "____________MCL_BP_Limiting_U(solution_input)_____\n";
    triangulation.clear_user_flags();
    
}


template<int dim>
void ScalarHyperbolic<dim>::ES_Limiting_u(BlockVector<double> &solution_input)
{

    BlockSparseMatrix<double> raw_correction_flux;
    raw_correction_flux.reinit(sparsity_pattern_transp);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(0,1).end(i);
        for(; index!=index_end; ++index)
        {

            unsigned int e = index->column();
            DOF.add_index(e+n_cg);
        }
        for(auto e : DOF)
        {
            double high_approx = low_high_matrix_cg(i,e);
            double low_approx = low_high_matrix_cg(e,i);
            raw_correction_flux.set(i, e, high_approx - low_approx);
        }
    }/*End of i-loop: assemble raw_correction_flux*/


    es_P_Q_flux_matrix_cg = 0;
    es_flux_limiter_matrix_cg = 1;

    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    // const unsigned int n_q_points      = quadrature_formula.size();
    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    const FEValuesExtractors::Scalar cg(0);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    FullMatrix<double> cell_matrix_d_cg     (dofs_per_cell, dofs_per_cell); /*d^e_{ij}*/      
    FullMatrix<double> cell_matrix_lambda(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_adv_comp0      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/
    FullMatrix<double> cell_matrix_adv_comp1      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/

    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        cell_matrix_adv_comp0 = 0; // -> c_e_ij, the first  component
        cell_matrix_adv_comp1 = 0; // -> c_e_ij, the second component
        cell_matrix_lambda    = 0;
        for(unsigned int i = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                for(const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    // Calculate c_e_ij, x-component
                    cell_matrix_adv_comp0(i,j) += fe_values.shape_grad(j, q_index)[0] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);
                    // Calculate c_e_ij, y-component
                    cell_matrix_adv_comp1(i,j) += fe_values.shape_grad(j, q_index)[1] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);     
                }/*End of quadrature*/
                Tensor<1,dim> n_ij;
                        n_ij[0] = cell_matrix_adv_comp0(i,j);
                        n_ij[1] = cell_matrix_adv_comp1(i,j);
                        n_ij = n_ij / n_ij.norm();
                        double u_i = solution_input(local_dof_indices[i]);
                        double u_j = solution_input(local_dof_indices[j]);
                        
                        cell_matrix_lambda(i,j) = Hyperbolic_max_wave_speed(u_i, u_j, n_ij);
            }
        }/*End of i-loop*/
        for(unsigned int i  = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double lambda_e_ij = cell_matrix_lambda(i,j);
                    
                    Tensor<1,dim> c_e_ji;
                    c_e_ji[0] = cell_matrix_adv_comp0(j,i);
                    c_e_ji[1] = cell_matrix_adv_comp1(j,i);
                    double lambda_e_ji = cell_matrix_lambda(j,i);

                    cell_matrix_d_cg(i,j) = std::max(lambda_e_ij * c_e_ij.norm(),
                                                     lambda_e_ji * c_e_ji.norm());
                    cell_matrix_d_cg(i,i) += -cell_matrix_d_cg(i,j);
                }/*Only for CG-components*/
            }/*End of j-loop*/
        }/*End of i-loop*/
        double v_e = 0;
        for(unsigned int i = 0; i < 4; ++i)
        {
            double u_i = solution_input(local_dof_indices[i]);
            double v_i = entropy_square_variable(u_i);
            v_e += v_i;
        }
        v_e = v_e * 0.25;
        for(unsigned int i = 0; i < 4; ++i)
        {
            double u_i = solution_input(local_dof_indices[i]);
            double v_i = entropy_square_variable(u_i);
            double f_ei = raw_correction_flux(local_dof_indices[i], local_dof_indices[4]);
            es_P_Q_flux_matrix_cg.set(local_dof_indices[4], local_dof_indices[i], (v_i - v_e) * f_ei);
        }
        for(unsigned int i = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double d_e_ij = cell_matrix_d_cg(i,j);
                    double u_i = solution_input(local_dof_indices[i]);
                    double u_j = solution_input(local_dof_indices[j]);

                    Tensor<1,dim> F_j = Hyperbolic_flux(u_j);
                    Tensor<1,dim> F_i = Hyperbolic_flux(u_i);
                    // Tensor<1,dim> F_central = Hyperbolic_flux(0.5 * (u_j+ u_i) );
                
                    double v_i = entropy_square_variable(u_i);
                    double v_j = entropy_square_variable(u_j);
                    
                    
                    Tensor<1,dim> q_i = entropy_q(u_i);
                    Tensor<1,dim> q_j = entropy_q(u_j);
                    Tensor<1,dim> psi_i = v_i * F_i - q_i;
                    Tensor<1,dim> psi_j = v_j * F_j - q_j;

                    // double Q_e_ij = (psi_j - psi_i) * c_e_ij 
                    //                  - 0.5 * (v_i - v_j) * (d_e_ij * (u_j - u_i) - 
                    //                                         (F_j + F_i) * c_e_ij);

                    double Q_e_ij = (psi_j - psi_i) * c_e_ij 
                                     - 0.5 * (v_j - v_i) *  (F_j + F_i) * c_e_ij;
                                                            
                    
                    // double Q_e_ij = (psi_j - psi_i) * c_e_ij 
                    //                  - (v_i - v_e) * (d_e_ij * (u_j - u_i) - 
                    //                                         (F_j + F_i) * c_e_ij);

                    // double temp = std::min(0., 0.5 * (v_i - v_j) * (F_j + F_i - 2 * F_central) * c_e_ij);

                    // double temp = std::min(0., (v_i - v_e) * (F_j + F_i - 2 * F_central) * c_e_ij);

                    // double Q_ED_ij = std::max(0., Q_e_ij + temp);
                    double Q_ED_ij = std::max(0., (v_j - v_i)/2. * d_e_ij*(u_j - u_i) + std::min(0., Q_e_ij)  
                    );

                    es_P_Q_flux_matrix_cg.add(local_dof_indices[i], local_dof_indices[4], Q_ED_ij);


                }
            }
        }

        for(unsigned int i = 0; i < 4; ++i)
        {
             double Q_ei = es_P_Q_flux_matrix_cg(local_dof_indices[i], local_dof_indices[4]);
             double P_ei = es_P_Q_flux_matrix_cg(local_dof_indices[4], local_dof_indices[i]);
            //  if(P_ei < 1e-12 || Q_ei < 1e-12) 
            //         es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
            //                                        local_dof_indices[4], 
            //                                        1);
            //  else if(Q_ei < 1e-12) 
            //         es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
            //                                        local_dof_indices[4], 
            //                                        1);
            //  else if(P_ei >= 1e-12 && P_ei > Q_ei ) 
            //         es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
            //                                        local_dof_indices[4], 
            //                                        std::min(1., Q_ei / P_ei));
            es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
                                                   local_dof_indices[4], 
                                                   1);
            if(Q_ei < 0) std::cout << "ERRROR: Q_ei < 0\n";

            if(P_ei > Q_ei)
            {
                es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
                                                   local_dof_indices[4], 
                                                   std::min(1., Q_ei / P_ei));
                if(P_ei < tol)
                    es_flux_limiter_matrix_cg.set (local_dof_indices[i], 
                                                   local_dof_indices[4], 
                                                   1.);
            }
                                    
        }
    }/*End of cell-loop*/
    if(bOutput)
    {
        dof_handler_DGQ1.distribute_dofs(fe_DGQ1);
        Vector<double> es_P_output_cg;
        Vector<double> es_Q_output_cg;
            es_P_output_cg.reinit(dof_handler_DGQ1.n_dofs());
            es_Q_output_cg.reinit(dof_handler_DGQ1.n_dofs());
        const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const unsigned int dofs_per_cell_DGQ1 = fe_DGQ1.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices_DGQ1(dofs_per_cell_DGQ1);
        typename DoFHandler<dim>::active_cell_iterator
            cell   = dof_handler_transp.begin_active(),
            endc   = dof_handler_transp.end(),
            cell1  = dof_handler_DGQ1.begin_active();
        for(; cell != endc; ++cell, ++cell1 )
        {
            cell->get_dof_indices(local_dof_indices);
            cell1->get_dof_indices(local_dof_indices_DGQ1);
            unsigned int e = local_dof_indices[4];
            for(unsigned int i = 0; i < 4; ++i)
            {
                double P_ei = es_P_Q_flux_matrix_cg(e, local_dof_indices[i]);
                double Q_ei = es_P_Q_flux_matrix_cg(local_dof_indices[i], e);
                    es_P_output_cg(local_dof_indices_DGQ1[i]) = P_ei;
                    es_Q_output_cg(local_dof_indices_DGQ1[i]) = Q_ei;
            }
        }
        DataOut<dim> data_out_DGQ1;
        data_out_DGQ1.attach_dof_handler(dof_handler_DGQ1);
        std::vector<std::string> es_P_names_cg;
            es_P_names_cg.push_back("P_ei");
            data_out_DGQ1.add_data_vector(es_P_output_cg,es_P_names_cg);
        std::vector<std::string> es_Q_names_cg;
            es_Q_names_cg.push_back("Q_ei_ED");
            data_out_DGQ1.add_data_vector(es_Q_output_cg, es_Q_names_cg);
        data_out_DGQ1.build_patches();
        data_out_DGQ1.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        const std::string filename_cg =
            task_name 
            +Utilities::int_to_string(RefineTime,1)
            + "_P_Q_CG_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_cg(filename_cg);
        data_out_DGQ1.write_vtk(output_cg);
    }
}

template<int dim>
void ScalarHyperbolic<dim>::ES_Limiting_U(BlockVector<double> &solution_input)
{
    es_P_matrix_average = 0;
    es_Q_matrix_average = 0;
    es_flux_limiter_matrix_average = 1;

    FullMatrix<double> raw_correction_flux;
    raw_correction_flux = FullMatrix<double>(n_dg, 4);
    for(unsigned int e = 0; e < n_dg; ++e)
    {
        for(unsigned int face_number = 0; face_number < 4; ++face_number)
        {
            raw_correction_flux(e, face_number) 
                    = high_matrix_average(e, face_number) 
                    - low_matrix_average(e, face_number);
        }
    }


    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    QGauss<dim-1>   face_quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe_transp,
                                        face_quadrature_formula, 
                                        update_values | update_gradients |
                                        update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
    FEFaceValues<dim> fe_face_values_neighbor (fe_transp,
                                                face_quadrature_formula,
                                                update_values | update_gradients |
                                                update_quadrature_points  |
                                                update_normal_vectors |
                                                update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();                                 
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_face(dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(;cell!=endc; ++cell)
    {
        cell->get_dof_indices(local_dof_indices);
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                cell->get_dof_indices(local_dof_indices);
                unsigned int e = local_dof_indices[4] - n_cg;

                es_P_matrix_average(e, face_number) = 1;
                es_Q_matrix_average(e, face_number) = 1;
            }
            else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                fe_face_values.reinit(cell, face_number);
                fe_face_values_neighbor.reinit(neighbor, neighbor_face);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4]- n_cg;
                unsigned int e_neighbor = local_dof_indices_face[4] - n_cg;

                double eg_bar_cell     = solution_input(e + n_cg);
                double eg_bar_neighbor = solution_input(e_neighbor + n_cg);

                double v_neighbor = entropy_square_variable(eg_bar_neighbor);
                double v_cell     = entropy_square_variable(eg_bar_cell);

                double P =  (v_neighbor - v_cell) * raw_correction_flux(e,face_number);
                         
                es_P_matrix_average(e, face_number) = P;

                Tensor<1,dim> F_neighbor = Hyperbolic_flux(eg_bar_neighbor);
                Tensor<1,dim> F_cell     = Hyperbolic_flux(eg_bar_cell);
                Tensor<1,dim> q_neighbor = entropy_q(eg_bar_neighbor);
                Tensor<1,dim> q_cell     = entropy_q(eg_bar_cell);
                Tensor<1,dim> psi_neighbor = v_neighbor * F_neighbor - q_neighbor;
                Tensor<1,dim> psi_cell     = v_cell * F_cell - q_cell;

                double F_cd = 0.5 * (F_cell + F_neighbor) * fe_face_values.normal_vector(0);
                double Q_cd = (psi_neighbor - psi_cell) * fe_face_values.normal_vector(0) - (v_neighbor - v_cell) * F_cd;
                
                    
                double max_wave_speed = Hyperbolic_max_wave_speed(eg_bar_cell, eg_bar_neighbor, fe_face_values.normal_vector(0));
                double 
                    Q_ES = std::max(0., 
                                 ((v_neighbor - v_cell) * max_wave_speed / 2. * (eg_bar_neighbor - eg_bar_cell) + 
                                 std::min(0., Q_cd) )
                                 );
                
                es_Q_matrix_average(e, face_number) = Q_ES;
            }
        }/*End of faces-loop*/
    }/*End of cell-loop*/

    cell  = dof_handler_transp.begin_active();
    for(;cell!=endc; ++cell)
    {
        for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
        {
            if(cell->at_boundary(face_number))
            {
                cell->get_dof_indices(local_dof_indices);
                unsigned int e = local_dof_indices[4] - n_cg;

                // double P = es_P_matrix_average(e, face_number);
                // double Q = es_Q_matrix_average(e, face_number);
                es_flux_limiter_matrix_average(e,face_number) = 1;
            }else{
                const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_number);
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_number);
                fe_face_values.reinit(cell, face_number);
                fe_face_values_neighbor.reinit(neighbor, neighbor_face);
                cell->get_dof_indices(local_dof_indices);
                neighbor->get_dof_indices(local_dof_indices_face);
                unsigned int e = local_dof_indices[4]- n_cg;
                unsigned int e_neighbor = local_dof_indices_face[4] - n_cg;

                if(cell->face(face_number)->user_flag_set() == false)
                {
                    cell->face(face_number)->set_user_flag();
                    double P = es_P_matrix_average(e, face_number);
                    double Q = es_Q_matrix_average(e, face_number);
                    if(P < tol) es_flux_limiter_matrix_average(e, face_number) = 1;
                    else if(P >=tol && P > Q) es_flux_limiter_matrix_average(e, face_number) =  Q  / P  ;
                }else{
                    es_flux_limiter_matrix_average(e, face_number)
                         = es_flux_limiter_matrix_average(e_neighbor, neighbor_face);
                }
            }

        // unsigned int e = local_dof_indices[4]- n_cg;
        //    double Q = es_Q_matrix_average(e, face_number);
        //    if(Q < tol) es_flux_limiter_matrix_average(e, face_number) =  1  ;
        }
    }
    triangulation.clear_user_flags();

    if(bOutput)
    {
        dof_handler_DGQ1.distribute_dofs(fe_DGQ0);
        Vector<double> entroy_production_U;
        // Vector<double> es_Q_output_average;
            entroy_production_U.reinit(dof_handler_DGQ0.n_dofs());
            // es_Q_output_cg.reinit(dof_handler_DGQ0.n_dofs());
        const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const unsigned int dofs_per_cell_DGQ0 = fe_DGQ0.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices_DGQ0(dofs_per_cell_DGQ0);
        typename DoFHandler<dim>::active_cell_iterator
            cell   = dof_handler_transp.begin_active(),
            endc   = dof_handler_transp.end(),
            cell0  = dof_handler_DGQ0.begin_active();
        for(; cell != endc; ++cell, ++cell0 )
        {
            cell->get_dof_indices(local_dof_indices);
            cell0->get_dof_indices(local_dof_indices_DGQ0);
            unsigned int e = local_dof_indices[4];
            for(unsigned int face_number = 0;
            face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
            {
                entroy_production_U(local_dof_indices_DGQ0[0]) += es_P_matrix_average(e, face_number)  / 4.;
            }
        }
        DataOut<dim> data_out_DGQ0;
        data_out_DGQ0.attach_dof_handler(dof_handler_DGQ0);
        std::vector<std::string> es_P_names_average;
            es_P_names_average.push_back("P_average");
            data_out_DGQ0.add_data_vector(entroy_production_U,es_P_names_average);
        // std::vector<std::string> es_Q_names_cg;
        //     es_Q_names_cg.push_back("Q_ei_ED");
        //     data_out_DGQ1.add_data_vector(es_Q_output_cg, es_Q_names_cg);
        data_out_DGQ0.build_patches();
        data_out_DGQ0.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        const std::string filename_cg =
            task_name 
            +Utilities::int_to_string(RefineTime,1)
            + "_P_Q_Average_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_cg(filename_cg);
        data_out_DGQ0.write_vtk(output_cg);
    }
}


template<int dim>
void ScalarHyperbolic<dim>::Clip_and_Scale_u_and_U()
{
    clip_and_scale_flux_limiter_matrix_cg      = 1;
    clip_and_scale_flux_limiter_matrix_average = 1;

    BlockSparseMatrix<double> raw_correction_flux_cg;
    raw_correction_flux_cg.reinit(sparsity_pattern_transp);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(0,1).end(i);
        for(; index!=index_end; ++index)
        {

            unsigned int e = index->column();
            DOF.add_index(e+n_cg);
        }
        for(auto e : DOF)
        {
            double high_approx = low_high_matrix_cg(i,e);
            double low_approx = low_high_matrix_cg(e,i);
            raw_correction_flux_cg.set(i, e, high_approx - low_approx);
        }
    }/*End of i-loop: assemble raw_correction_flux*/



    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        double f_tilde_plus = 0;
        double f_tilde_minus = 0;
        // fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        unsigned int e = local_dof_indices[4];
        for(unsigned i = 0; i < 4; ++i)
        {
            double     f_ei = raw_correction_flux_cg(local_dof_indices[i], e);
            double  beta_bp = bp_flux_limiter_matrix_cg(local_dof_indices[i], e);
            double  beta_es = es_flux_limiter_matrix_cg(local_dof_indices[i], e);


            double R_ei = 1;
            if(task_number == 2)       R_ei = beta_bp;
            else if (task_number == 3) R_ei = beta_es;
            else if (task_number == 4) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 5) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 6) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 7) R_ei = beta_bp;

            double f_tilde_ei = R_ei * f_ei;
            f_tilde_plus  += std::max(0., f_tilde_ei);
            f_tilde_minus += std::min(0., f_tilde_ei); 
        }
        for(unsigned i = 0; i < 4; ++i)
        {
            double     f_ei = raw_correction_flux_cg(local_dof_indices[i], e);
            double  beta_bp = bp_flux_limiter_matrix_cg(local_dof_indices[i], e);
            double  beta_es = es_flux_limiter_matrix_cg(local_dof_indices[i], e);
            double R_ei = 1;
            if(task_number == 2)       R_ei = beta_bp;
            else if (task_number == 3) R_ei = beta_es;
            else if (task_number == 4) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 5) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 6) R_ei = std::min(beta_bp, beta_es);
            else if (task_number == 7) R_ei = beta_bp;

            double f_tilde_ei = R_ei * f_ei;
            double coeff = 1;
            if(cell->at_boundary())
            {

            }else{
                if(f_tilde_ei > 1e-10)                              coeff = std::min(1., - f_tilde_minus / (f_tilde_plus  + 1e-20) );
                else if(f_tilde_ei < -1e-10 )                       coeff = std::min(1., - f_tilde_plus /  (f_tilde_minus - 1e-20));
                else if(f_tilde_ei < 1e-10 || f_tilde_ei > - 1e-10) coeff = 1.;
            }
            clip_and_scale_flux_limiter_matrix_cg.set(local_dof_indices[i],
                                                      e,
                                                      R_ei * coeff);
        }
    }/*End of cell-loop*/

    for(unsigned int e = 0; e < n_dg; ++e)
    {
        for(unsigned int face_number = 0; face_number < 4; ++face_number)
        {
            double beta_average_bp = bp_flux_limiter_matrix_average(e, face_number);
            double beta_average_es = es_flux_limiter_matrix_average(e, face_number);

            if(task_number == 2)         clip_and_scale_flux_limiter_matrix_average(e, face_number) = beta_average_bp;
            else if(task_number == 3)    clip_and_scale_flux_limiter_matrix_average(e, face_number) = beta_average_es;
            else if(task_number == 4)    clip_and_scale_flux_limiter_matrix_average(e, face_number) = std::min(beta_average_bp, beta_average_es);
            else if(task_number == 5)    clip_and_scale_flux_limiter_matrix_average(e, face_number) = std::min(beta_average_bp, beta_average_es);
            else if(task_number == 6)    clip_and_scale_flux_limiter_matrix_average(e, face_number) = std::min(beta_average_bp, beta_average_es);
            else if(task_number == 7)    clip_and_scale_flux_limiter_matrix_average(e, face_number) = std::min(beta_average_bp, beta_average_es);
                    
        }
    }


}

template <int dim>
void ScalarHyperbolic<dim>::Extra_ES_Limiting_cg(BlockVector<double> &solution_input)
{
    dof_handler_DGQ0.distribute_dofs(fe_DGQ0);
    extra_es_limiting_P_cg.reinit(dof_handler_DGQ0.n_dofs());
    extra_es_limiting_Q_cg.reinit(dof_handler_DGQ0.n_dofs());
    extra_es_limiting_P_cg = 0;
    extra_es_limiting_Q_cg = 0;
    
    extra_es_limiter_cg.reinit(dof_handler_DGQ0.n_dofs());
    extra_es_limiter_cg = 1;    

    BlockSparseMatrix<double> modified_correction_flux;
    modified_correction_flux.reinit(sparsity_pattern_transp);
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        IndexSet DOF(dof_handler_transp.n_dofs());
        SparseMatrix<double>::const_iterator
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.block(0,1).end(i);
        for(; index!=index_end; ++index)
        {

            unsigned int e = index->column();
            DOF.add_index(e+n_cg);
        }
        for(auto e : DOF)
        {
            double high_approx = low_high_matrix_cg(i,e);
            double low_approx = low_high_matrix_cg(e,i);
            double alpha      = clip_and_scale_flux_limiter_matrix_cg(i,e);
            modified_correction_flux.set(i, e, alpha *( high_approx - low_approx) );
        }
    }/*End of i-loop: assemble raw_correction_flux*/

    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<types::global_dof_index> cell_index              (1);

    FullMatrix<double> cell_matrix_d_cg     (dofs_per_cell, dofs_per_cell); /*d^e_{ij}*/      
    FullMatrix<double> cell_matrix_lambda(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_adv_comp0      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/
    FullMatrix<double> cell_matrix_adv_comp1      (dofs_per_cell, dofs_per_cell); /*k^e_{ij}*/

    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_transp.begin_active(),
        endc = dof_handler_transp.end(),
        dg_cell = dof_handler_DGQ0.begin_active();
    for(; cell!=endc; ++cell, ++dg_cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
            unsigned int e = local_dof_indices[4];
        dg_cell->get_dof_indices(cell_index);
        cell_matrix_adv_comp0 = 0; // -> c_e_ij, the first  component
        cell_matrix_adv_comp1 = 0; // -> c_e_ij, the second component
        cell_matrix_lambda    = 0;
        cell_matrix_d_cg = 0;
        for(unsigned int i = 0; i <4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                for(const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    // Calculate c_e_ij, x-component
                    cell_matrix_adv_comp0(i,j) += fe_values.shape_grad(j, q_index)[0] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);
                    // Calculate c_e_ij, y-component
                    cell_matrix_adv_comp1(i,j) += fe_values.shape_grad(j, q_index)[1] *
                                                  fe_values.shape_value(i, q_index)   *
                                                  fe_values.JxW(q_index);     
                }/*End of quadrature*/
                Tensor<1,dim> n_ij;
                        n_ij[0] = cell_matrix_adv_comp0(i,j);
                        n_ij[1] = cell_matrix_adv_comp1(i,j);
                        n_ij = n_ij / n_ij.norm();
                        double u_i = solution_input(local_dof_indices[i]);
                        double u_j = solution_input(local_dof_indices[j]);
                        
                        cell_matrix_lambda(i,j) = Hyperbolic_max_wave_speed(u_i, u_j, n_ij);
            }
        }/*End of i-loop*/
        for(unsigned int i  = 0; i < 4; ++i)
        {
            for(unsigned int j = 0; j < 4; ++j)
            {
                if(j!=i)
                {
                    Tensor<1,dim> c_e_ij;
                    c_e_ij[0] = cell_matrix_adv_comp0(i,j);
                    c_e_ij[1] = cell_matrix_adv_comp1(i,j);
                    double lambda_e_ij = cell_matrix_lambda(i,j);
                    
                    Tensor<1,dim> c_e_ji;
                    c_e_ji[0] = cell_matrix_adv_comp0(j,i);
                    c_e_ji[1] = cell_matrix_adv_comp1(j,i);
                    double lambda_e_ji = cell_matrix_lambda(j,i);

                    cell_matrix_d_cg(i,j) = std::max(lambda_e_ij * c_e_ij.norm(),
                                                     lambda_e_ji * c_e_ji.norm());
                    cell_matrix_d_cg(i,i) += -cell_matrix_d_cg(i,j);
                }/*Only for CG-components*/
            }/*End of j-loop*/
        }/*End of i-loop*/

        for(unsigned int i = 0; i < 4; ++i)
        {
            double u_i = solution_input(local_dof_indices[i]);
            double v_i = entropy_square_variable(u_i);
            double f_star_ei = modified_correction_flux(local_dof_indices[i],e);

            extra_es_limiting_P_cg[cell_index[0]] += v_i * f_star_ei;
            for(unsigned int j = 0; j < 4; ++j)
            {
                // if(j!=i)
                {
                    double d_e_ij = cell_matrix_d_cg(i,j);
                    double u_j = solution_input(local_dof_indices[j]);
                    double v_j = entropy_square_variable(u_j);
                    extra_es_limiting_Q_cg[cell_index[0]] -= v_i * d_e_ij * u_j;
                }
                /*Concern: should this be
                v_j = entropy_square_variable(u_j);
                extra_es_limiting_Q_cg[cell_index[0]] -= v_j * d_e_ij * u_j;
                 */
            }
            
        }/*End of i-loop*/
        
        double Q = std::max(0.,extra_es_limiting_Q_cg[cell_index[0]]);
        double P = extra_es_limiting_P_cg[cell_index[0]];
        if(Q < 0.) std::cout << "extra_es_limiting_Q_cg value is negative!\n";
        else{
            if( P > Q ) extra_es_limiter_cg(cell_index[0]) = Q / P;
        }
    }/*End of cell-loop*/
    
    if(bOutput)
    {
        DataOut<dim> data_out_DGQ0;
        data_out_DGQ0.attach_dof_handler(dof_handler_DGQ0);
        std::vector<std::string> extra_es_fluxlimiter_cg_names_average;
        std::vector<std::string> extra_es_P_cg_names_average;
        std::vector<std::string> extra_es_Q_cg_names_average;

            extra_es_fluxlimiter_cg_names_average.push_back("extra_es");
            extra_es_P_cg_names_average.push_back("P");
            extra_es_Q_cg_names_average.push_back("Q");
            
            data_out_DGQ0.add_data_vector(extra_es_limiter_cg, extra_es_fluxlimiter_cg_names_average);
            data_out_DGQ0.add_data_vector(extra_es_limiting_P_cg, extra_es_P_cg_names_average);
            data_out_DGQ0.add_data_vector(extra_es_limiting_Q_cg, extra_es_Q_cg_names_average);
        data_out_DGQ0.build_patches();
        data_out_DGQ0.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        const std::string filename_average =
            task_name + 
            Utilities::int_to_string(RefineTime,1)+
            "_Extra_ES_FluxLimiter_CG_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_average(filename_average);
        data_out_DGQ0.write_vtk(output_average);
    }
    
}

template<int dim>
void ScalarHyperbolic<dim>::Construct_LowOrder_Solution(BlockVector<double> &solution_input)
{
    loworder_solution_transp_EG = solution_input;
    for(unsigned int i = 0; i < n_cg;++i)
    {
        SparseMatrix<double>::const_iterator 
            index = mass_matrix_transp.block(0,1).begin(i);
        SparseMatrix<double>::const_iterator 
            index_end = mass_matrix_transp.block(0,1).end(i);
        double m_i = mass_matrix_transp(i,i);
        double CFL = time_step / m_i;
        for(; index!=index_end; ++index)
        {
            unsigned int e_support = index->column() + n_cg;
            loworder_solution_transp_EG(i) += CFL * low_high_matrix_cg(e_support, i);
        }   
    }
    for(unsigned int e = 0; e < n_dg; ++e)
    {
        unsigned int e_dof = e + n_cg;
        double m_e = mass_matrix_transp(e_dof,e_dof);
        double CFL = time_step / m_e;
        for(unsigned int face_number = 0; face_number < 4; ++face_number)
            loworder_solution_transp_EG(e_dof) += CFL * low_matrix_average(e,face_number);
    }
    if(bOutput)
    {
      DataOut<dim> data_out;
      std::vector<std::string> EG_names;
      EG_names.push_back("u_c");
      EG_names.push_back("ubar");
      data_out.attach_dof_handler(dof_handler_transp);
      data_out.add_data_vector(loworder_solution_transp_EG, EG_names);
      data_out.build_patches();
      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
      const std::string filename = 
        // Utilities::int_to_string(task_number, 1)
        task_name
        +Utilities::int_to_string(RefineTime,1)
        +"_LowOrder_TestCase_" + std::to_string(PrescribedSolution::test_case)
        +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);
    }
}

template<int dim>
BlockVector<double> ScalarHyperbolic<dim>::Construct_Solution(BlockVector<double> &solution_input)
{
    BlockVector<double> solution_output;
        solution_output.reinit(2);
        solution_output.block(0).reinit(n_cg);
        solution_output.block(1).reinit(n_dg);
        solution_output.collect_sizes();
    solution_output = solution_input;

    if(task_number == 0)
    {
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support, i);
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number);
        }
    }
    else if(task_number == 1)
    {
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(i,e_support);
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
                solution_output(e_dof) += CFL * high_matrix_average(e,face_number);
        }
    }else if(task_number == 2)
    {
        //  bp_flux_limiter_matrix_cg  =1;
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )   
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support);
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number))
                                          * clip_and_scale_flux_limiter_matrix_average(e,face_number);

            }
                                      
        }
    }else if(task_number == 3)
    {
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )  
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support);
                                      
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number))
                                          *  clip_and_scale_flux_limiter_matrix_average(e,face_number);

            }
                                      
        }
    }else if(task_number == 4)
    {
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )   
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support);
                                      
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number)) 
                                          * clip_and_scale_flux_limiter_matrix_average(e,face_number);

            }
                                      
        }
    }
    else if(task_number == 5)
    {
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )   
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support)
                                      *  extra_es_limiter_cg(e_support - n_cg);
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number)) 
                                          * clip_and_scale_flux_limiter_matrix_average(e,face_number)
                                          ;

            }
                                      
        }
    }else if(task_number == 6)
    {
        // solution_output = loworder_solution_transp_EG;
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )   
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support);
                                      
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number)) 
                                          * clip_and_scale_flux_limiter_matrix_average(e,face_number);

            }
                                      
        }
    }else if(task_number == 7)
    {
        // solution_output = loworder_solution_transp_EG;
        for(unsigned int i = 0; i < n_cg;++i)
        {
            SparseMatrix<double>::const_iterator 
                index = mass_matrix_transp.block(0,1).begin(i);
            SparseMatrix<double>::const_iterator 
                index_end = mass_matrix_transp.block(0,1).end(i);
            double m_i = mass_matrix_transp(i,i);
            double CFL = time_step / m_i;
            for(; index!=index_end; ++index)
            {
                unsigned int e_support = index->column() + n_cg;
                solution_output(i) += CFL * low_high_matrix_cg(e_support,i) ;
                solution_output(i) += CFL * (low_high_matrix_cg(i,e_support)  - low_high_matrix_cg(e_support,i) )   
                                      *  clip_and_scale_flux_limiter_matrix_cg(i, e_support)
                                      *  extra_es_limiter_cg(e_support - n_cg);
            }   
        }
        for(unsigned int e = 0; e < n_dg; ++e)
        {
            unsigned int e_dof = e + n_cg;
            double m_e = mass_matrix_transp(e_dof,e_dof);
            double CFL = time_step / m_e;
            for(unsigned int face_number = 0; face_number < 4; ++face_number)
            {
                solution_output(e_dof) += CFL * low_matrix_average(e,face_number) ;
                solution_output(e_dof) += CFL * (high_matrix_average(e,face_number) - low_matrix_average(e,face_number)) 
                                          * clip_and_scale_flux_limiter_matrix_average(e,face_number)
                                          ;

            }
        }      
    }
    // return loworder_solution_transp_EG;
    return solution_output;
}






template<int dim>
BlockVector<double> ScalarHyperbolic<dim>::EulerUpdate(BlockVector<double> &solution_input,
                                                          int stage)
{
    Low_High_Approx_u(solution_input,stage);
    Low_High_Approx_U(solution_input,stage);

    // Limiting_Condition_Check();
    if(task_number == 2)
    {
        MCL_BP_Limiting_u(solution_input);
        MCL_BP_Limiting_U(solution_input);
        Clip_and_Scale_u_and_U();

    }else if(task_number == 3)
    {
        ES_Limiting_u(solution_input);
        ES_Limiting_U(solution_input);
        Clip_and_Scale_u_and_U();
    }else if(task_number == 4)
    {
        MCL_BP_Limiting_u(solution_input);
        MCL_BP_Limiting_U(solution_input);
        ES_Limiting_u(solution_input);
        ES_Limiting_U(solution_input);
        Clip_and_Scale_u_and_U();

    }else if(task_number == 5)
    {
        MCL_BP_Limiting_u(solution_input);
        MCL_BP_Limiting_U(solution_input);
        ES_Limiting_u(solution_input);
        ES_Limiting_U(solution_input);
        Clip_and_Scale_u_and_U();
        Extra_ES_Limiting_cg(solution_input);
    }else if(task_number == 6)
    {
        // Construct_LowOrder_Solution(solution_input);
        // std::cout << "after loworder_solution\n";
        // FCT_BP_Limiting_u(solution_input);
        // // std::cout << "after FCT_BP_u\n";
        // FCT_BP_Limiting_U(solution_input);
        // // std::cout << "after FCT_BP_U\n";
        // ES_Limiting_u(solution_input);
        // // std::cout << "after ES_u\n";
        // ES_Limiting_U(solution_input);
        // // std::cout << "after ES_U\n";
        Clip_and_Scale_u_and_U();
        // std::cout << "after Clip\n";
    }else if(task_number == 7)
    {
        MCL_BP_Limiting_u(solution_input);
        MCL_BP_Limiting_U(solution_input);
        // ES_Limiting_u(solution_input);
        ES_Limiting_U(solution_input);
        Clip_and_Scale_u_and_U();
        Extra_ES_Limiting_cg(solution_input);
    }
    return Construct_Solution(solution_input);
}

template<int dim>
void ScalarHyperbolic<dim>::Compute_EG_Solution()
{
    EG_solution.block(0) = solution_transp_EG.block(0);
    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    const unsigned int n_q_points      = quadrature_formula.size();

    FEValues<dim>     fe_values(fe_transp,
                              quadrature_formula,
                              update_values | update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);
    std::vector<double>  solution_values_transp_cg(n_q_points);
     const FEValuesExtractors::Scalar cg(0);

    typename DoFHandler<dim>::active_cell_iterator
        cell  = dof_handler_transp.begin_active(),
        endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        fe_values[cg].get_function_values(solution_transp_EG,
                                            solution_values_transp_cg);
        double cell_area = mass_matrix_transp(local_dof_indices[4],
                                              local_dof_indices[4]);
        double average_value = 0;
        for(const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            double u_cg_q = solution_values_transp_cg[q_index];
            average_value += u_cg_q * fe_values.JxW(q_index);
        }
        CG_Average(local_dof_indices[4]) = average_value / cell_area;
        EG_solution(local_dof_indices[4]) = solution_transp_EG(local_dof_indices[4]) - CG_Average(local_dof_indices[4]);
    }  /*End of cell-loop*/ 
    if(bOutput)
    {
      DataOut<dim> data_out;
      std::vector<std::string> EG_names;
      EG_names.push_back("u_c");
      EG_names.push_back("u_d");
      data_out.attach_dof_handler(dof_handler_transp);
      data_out.add_data_vector(EG_solution, EG_names);
      data_out.build_patches();
      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
      const std::string filename = 
        "EG_solution_"
        +task_name
        +Utilities::int_to_string(RefineTime,1)
        +"_TestCase_" + std::to_string(PrescribedSolution::test_case)
        +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);
    }
    
}




template<int dim>
void ScalarHyperbolic<dim>::Error()
{
    PrescribedSolution::Transp_EG::Exact_transp<dim>  exact;
          exact.set_time(time);
    QGauss<dim>     quadrature_formula(fe_transp.degree + 2);
    const unsigned int n_q_points      = quadrature_formula.size(); 
    FEValues<dim>     fe_values_transp(fe_transp,
                                quadrature_formula,
                                update_values | update_gradients |
                                update_quadrature_points |
                                update_JxW_values);   
    double L1_norm_EG = 0;
    double L2_norm_EG = 0;
    double local_error_EG_L1 = 0;
    double local_error_EG_L2 = 0;
    std::vector<double> exact_transp_values(n_q_points);
    std::vector<Vector<double>> solution_values_transp(n_q_points, Vector<double>(2));

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_transp.begin_active(),
      endc = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
      fe_values_transp.reinit(cell);
      fe_values_transp.get_function_values(EG_solution,
                                           solution_values_transp);
                                            
      exact.value_list(fe_values_transp.get_quadrature_points(),
                       exact_transp_values);
      for(unsigned int q = 0; q<n_q_points; ++q)
      {
        // const auto &x_q = fe_values_transp.quadrature_point(q);
        // Tensor<1,dim> tmp = 
        //   exact.gradient(x_q) - solution_grads_transp[q][0] - solution_grads_transp[q][1];
        //     // exact.gradient(x_q) - solution_grads_transp[q];
        // local_error_EG_H1 = tmp.norm_square();

        local_error_EG_L1 = exact_transp_values[q] 
                            - solution_values_transp[q][0]
                            - solution_values_transp[q][1];
        local_error_EG_L2 = exact_transp_values[q] 
                            - solution_values_transp[q][0]
                            - solution_values_transp[q][1];

        L1_norm_EG += abs(local_error_EG_L1) * fe_values_transp.JxW(q);
        L2_norm_EG += local_error_EG_L2 * local_error_EG_L2 * fe_values_transp.JxW(q);
        // H1_seminorm_EG += local_error_EG_H1 * fe_values_transp.JxW(q);
      }
    }
    L1_error_transp = L1_norm_EG;
    L2_error_transp = sqrt(L2_norm_EG);
    if (L2_error_transp > sup_L2_error_transp)
    {
      sup_L2_error_transp = L2_error_transp;
    }

    if(L1_error_transp > sup_L1_error_transp)
    {
        sup_L1_error_transp = L1_error_transp;
    }
}


// template <int dim>
// void ScalarHyperbolic<dim>::Entropy_Production_Output()
// {
//     // BlockSparseMatrix<double> raw_correction_flux_cg;
//     // raw_correction_flux_cg.reinit(sparsity_pattern_transp);
//     // for(unsigned int i = 0; i < n_cg; ++i)
//     // {
//     //     IndexSet DOF(dof_handler_transp.n_dofs());
//     //     SparseMatrix<double>::const_iterator
//     //         index = mass_matrix_transp.block(0,1).begin(i);
//     //     SparseMatrix<double>::const_iterator
//     //         index_end = mass_matrix_transp.block(0,1).end(i);
//     //     for(; index!=index_end; ++index)
//     //     {

//     //         unsigned int e = index->column();
//     //         DOF.add_index(e+n_cg);
//     //     }
//     //     for(auto e : DOF)
//     //     {
//     //         double high_approx = low_high_matrix_cg(i,e);
//     //         double low_approx = low_high_matrix_cg(e,i);
//     //         raw_correction_flux_cg.set(i, e, high_approx - low_approx);
//     //     }
//     // }/*End of i-loop: assemble raw_correction_flux*/
//     FullMatrix<double> raw_correction_flux_average;
//     raw_correction_flux_average = FullMatrix<double>(n_dg, 4);
//     for(unsigned int e = 0; e < n_dg; ++e)
//     {
//         for(unsigned int face_number = 0; face_number < 4; ++face_number)
//         {
//             raw_correction_flux_average(e, face_number) 
//                     = high_matrix_average(e, face_number) 
//                     - low_matrix_average(e, face_number);
//         }
//     }

//     if(bOutput)
//     {
//         dof_handler_DGQ0.distribute_dofs(fe_DGQ0);
//         dof_handler_DGQ1.distribute_dofs(fe_DGQ1);
//         Vector<double> entropy_production_output_cg;
//         Vector<double> entropy_production_output_eg;
//     }
// }


template<int dim>
void ScalarHyperbolic<dim>::FluxLimiter_Output()
{
    if(bOutput)
    {
        dof_handler_DGQ0.distribute_dofs(fe_DGQ0);
        dof_handler_DGQ1.distribute_dofs(fe_DGQ1);

        Vector<double> bp_flux_limiter_output_cg;
        Vector<double> es_flux_limiter_output_cg;
        Vector<double> flux_limiter_output_cg;
            bp_flux_limiter_output_cg.reinit(dof_handler_DGQ1.n_dofs());
            es_flux_limiter_output_cg.reinit(dof_handler_DGQ1.n_dofs());
            flux_limiter_output_cg.reinit(dof_handler_DGQ1.n_dofs());
            
        Vector<double> bp_flux_limiter_output_average;
        Vector<double> es_flux_limiter_output_average;
        Vector<double> flux_limiter_output_average;
       
            bp_flux_limiter_output_average.reinit(dof_handler_DGQ0.n_dofs());
            es_flux_limiter_output_average.reinit(dof_handler_DGQ0.n_dofs());
            flux_limiter_output_average.reinit(dof_handler_DGQ0.n_dofs());

        const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const unsigned int dofs_per_cell_DGQ1 = fe_DGQ1.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices_DGQ1(dofs_per_cell_DGQ1);

        const unsigned int dofs_per_cell_DGQ0 = fe_DGQ0.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dof_indices_DGQ0(dofs_per_cell_DGQ0);

        typename DoFHandler<dim>::active_cell_iterator
            cell   = dof_handler_transp.begin_active(),
            endc   = dof_handler_transp.end(),
            cell0  = dof_handler_DGQ0.begin_active(),
            cell1  = dof_handler_DGQ1.begin_active();
        for(; cell != endc; ++cell, ++cell0, ++cell1 )
        {
            cell->get_dof_indices(local_dof_indices);



            cell1->get_dof_indices(local_dof_indices_DGQ1);
            unsigned int e = local_dof_indices[4];
            for(unsigned int i = 0; i < 4; ++i)
            {
                double alpha_bp_ei = bp_flux_limiter_matrix_cg(local_dof_indices[i], e);
                double alpha_es_ei = es_flux_limiter_matrix_cg(local_dof_indices[i], e);
                double alpha_ei    = clip_and_scale_flux_limiter_matrix_cg(local_dof_indices[i], e);
                    bp_flux_limiter_output_cg(local_dof_indices_DGQ1[i]) = alpha_bp_ei;
                    es_flux_limiter_output_cg(local_dof_indices_DGQ1[i]) = alpha_es_ei;
                    flux_limiter_output_cg(local_dof_indices_DGQ1[i])    = alpha_ei;
            }


            cell0->get_dof_indices(local_dof_indices_DGQ0);
            for(unsigned int face_number = 0;
                face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
            {
                double alpha_bp_eface = bp_flux_limiter_matrix_average(e - n_cg, face_number);
                double alpha_es_eface = es_flux_limiter_matrix_average(e - n_cg, face_number);
                double alpha_eface    = clip_and_scale_flux_limiter_matrix_average(e - n_cg, face_number);
                    bp_flux_limiter_output_average(local_dof_indices_DGQ0[0]) += alpha_bp_eface / 4;
                    es_flux_limiter_output_average(local_dof_indices_DGQ0[0]) += alpha_es_eface / 4;
                    flux_limiter_output_average(local_dof_indices_DGQ0[0])    += alpha_eface    / 4;
            }
        }/*End of cell-loop*/
        // std::cout << bp_flux_limiter_output_average << std::endl;
        DataOut<dim> data_out_DGQ0;
        data_out_DGQ0.attach_dof_handler(dof_handler_DGQ0);
        std::vector<std::string> bp_fluxlimiter_names_average;
        std::vector<std::string> es_fluxlimiter_names_average;
        std::vector<std::string> fluxlimiter_names_average;
            bp_fluxlimiter_names_average.push_back("bp");
            data_out_DGQ0.add_data_vector(bp_flux_limiter_output_average, bp_fluxlimiter_names_average);

            es_fluxlimiter_names_average.push_back("es");
            data_out_DGQ0.add_data_vector(es_flux_limiter_output_average, es_fluxlimiter_names_average);

            fluxlimiter_names_average.push_back("alpha_bar");
            data_out_DGQ0.add_data_vector(flux_limiter_output_average, fluxlimiter_names_average);
        
        data_out_DGQ0.build_patches();
        data_out_DGQ0.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        const std::string filename_average =
            task_name + 
            Utilities::int_to_string(RefineTime,1)+
            "_FluxLimiter_Average_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_average(filename_average);
        data_out_DGQ0.write_vtk(output_average);


        DataOut<dim> data_out_DGQ1;
        data_out_DGQ1.attach_dof_handler(dof_handler_DGQ1);
        std::vector<std::string> bp_fluxlimiter_names_cg;
        std::vector<std::string> es_fluxlimiter_names_cg;
        std::vector<std::string> fluxlimiter_names_cg;
            bp_fluxlimiter_names_cg.push_back("bp");
            data_out_DGQ1.add_data_vector(bp_flux_limiter_output_cg, bp_fluxlimiter_names_cg);

            es_fluxlimiter_names_cg.push_back("es");
            data_out_DGQ1.add_data_vector(es_flux_limiter_output_cg, es_fluxlimiter_names_cg);

            fluxlimiter_names_cg.push_back("alpha");
            data_out_DGQ1.add_data_vector(flux_limiter_output_cg, fluxlimiter_names_cg);
        
        data_out_DGQ1.build_patches();
        data_out_DGQ1.set_flags(DataOutBase::VtkFlags(time, timestep_number));
        const std::string filename_cg =
            task_name +
            Utilities::int_to_string(RefineTime,1)+
             "_FluxLimiter_CG_TestCase_" + std::to_string(PrescribedSolution::test_case)
            +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
        std::ofstream output_cg(filename_cg);
        data_out_DGQ1.write_vtk(output_cg);
    }   
}

template<int dim>
void ScalarHyperbolic<dim>::Output()
{
    if(bOutput)
    {
      DataOut<dim> data_out;
      std::vector<std::string> EG_names;
      EG_names.push_back("u_c");
      EG_names.push_back("ubar");
      data_out.attach_dof_handler(dof_handler_transp);
      data_out.add_data_vector(solution_transp_EG, EG_names);
      data_out.build_patches();
      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
      const std::string filename = 
        // Utilities::int_to_string(task_number, 1)
        task_name
        +Utilities::int_to_string(RefineTime,1)
        +"_TestCase_" + std::to_string(PrescribedSolution::test_case)
        +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);
    }
    

}



template<int dim>
std::vector<double> Relative_Error_Last_Time_Step(ScalarHyperbolic<dim> &set1,ScalarHyperbolic<dim> &set2)
{
    QGauss<dim> quadrature_formula(set2.fe_transp.degree + 2);
    const unsigned int n_q_points      = quadrature_formula.size();

    FEValues<dim>     fe_values_transp(set2.fe_transp,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    double local_error_EG_L1 = 0;
    double local_error_EG_L2 = 0;
    double L1_norm_EG = 0;
    double L2_norm_EG = 0;
    double L1_error_u = 0;
    double L1_error_U = 0;

    std::vector<double> exact_transp_values(n_q_points);
    std::vector<Vector<double>> exact_transp_gradients(n_q_points, Vector<double>(dim));

    std::vector<Vector<double>> solution_values_transp(n_q_points, Vector<double>(2));
    std::vector<Vector<double>> solution_values_transp_u(n_q_points, Vector<double>(2));
    std::vector<Vector<double>> solution_values_transp_U(n_q_points, Vector<double>(2));
    std::vector<std::vector<Tensor<1,dim>>> solution_grads_transp(n_q_points, std::vector<Tensor<1,dim>> (2));

    BlockVector<double> temp_u_set2 = set2.solution_transp_EG; temp_u_set2.block(1) = 0;
    BlockVector<double> temp_U_set2 = set2.solution_transp_EG; temp_u_set2.block(0) = 0;
    
    BlockVector<double> temp_u_set1 = set1.solution_transp_EG; temp_u_set1.block(1) = 0;
    BlockVector<double> temp_U_set1 = set1.solution_transp_EG; temp_U_set1.block(0) = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = set2.dof_handler_transp.begin_active(),
      endc = set2.dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        fe_values_transp.reinit(cell);
      fe_values_transp.get_function_values(
                                           set2.EG_solution,
                                           solution_values_transp);
      fe_values_transp.get_function_gradients(
                                             set2.EG_solution,
                                              solution_grads_transp);
        
        
        fe_values_transp.get_function_values(
                                           temp_u_set2,
                                           solution_values_transp_u);

        
        fe_values_transp.get_function_values(
                                           temp_U_set2,
                                           solution_values_transp_U);
        
      for(unsigned int q = 0; q<n_q_points; ++q)
      {
        const auto &x_q = fe_values_transp.quadrature_point(q);
        double coarse_solution_q = VectorTools::point_value(set1.dof_handler_transp, 
                                                          set1.EG_solution,
                                                          x_q);
         

        local_error_EG_L1 = coarse_solution_q
                            - solution_values_transp[q][0]
                            - solution_values_transp[q][1];
        local_error_EG_L2 = coarse_solution_q
                            - solution_values_transp[q][0]
                            - solution_values_transp[q][1];

        L1_norm_EG += abs(local_error_EG_L1) * fe_values_transp.JxW(q);
        L2_norm_EG += std::pow(local_error_EG_L2,2.) * fe_values_transp.JxW(q);

        double coarse_solution_q_set1_u = VectorTools::point_value(set1.dof_handler_transp,
                                                                 temp_u_set1,
                                                                 x_q);
        double coarse_solution_q_set1_U = VectorTools::point_value(set1.dof_handler_transp,
                                                                 temp_U_set1,
                                                                 x_q);
        L1_error_u += abs(coarse_solution_q_set1_u - solution_values_transp_u[q][0]) * fe_values_transp.JxW(q);

        L1_error_U += abs(coarse_solution_q_set1_U - solution_values_transp_U[q][1]) * fe_values_transp.JxW(q);
      }
    }
    double L1_error_transp = L1_norm_EG;
    double L2_error_transp = sqrt(L2_norm_EG);
    std::vector<double> return_value = {L1_error_transp, L2_error_transp, L1_error_u, L1_error_U};
    std::cout << "L1_error_transp = " << L1_error_transp << std::endl;
    std::cout << "L2_error_transp = " << L2_error_transp << std::endl;
    std::cout << std::endl;

    return return_value;

    
};

template<int dim>
void ScalarHyperbolic<dim>::FCR_Process()
{
    BlockSparseMatrix<double> system_matrix_EG;
    system_matrix_EG.reinit(sparsity_pattern_transp);
    BlockVector<double> system_rhs_EG;
    system_rhs_EG.reinit(2);
    system_rhs_EG.block(0).reinit(n_cg);
    system_rhs_EG.block(1).reinit(n_dg);
    system_rhs_EG.collect_sizes();



    QGauss<dim>     quadrature_formula(fe_transp.degree + 3);
    const unsigned int n_q_points      = quadrature_formula.size();

    FEValues<dim>     fe_values(fe_transp,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_JxW_values);
    const unsigned int dofs_per_cell = fe_transp.n_dofs_per_cell();
    std::vector<double>  solution_values_transp_cg(n_q_points);
    
    std::vector<types::global_dof_index> local_dof_indices     (dofs_per_cell);

    const FEValuesExtractors::Scalar cg(0);
    const FEValuesExtractors::Scalar dg(0);  

    system_matrix_EG           = 0;
    system_rhs_EG              = 0;
    std::vector<Vector<double>>  solution_values_transp(n_q_points, Vector<double>(2));
    FullMatrix<double> cell_matrix_mass     (dofs_per_cell, dofs_per_cell); /*m^e_{ij}*/
    Vector<double>     cell_rhs(dofs_per_cell);

    BlockVector<double> uH_solution;
        uH_solution.reinit(2);
        uH_solution.block(0).reinit(n_cg);
        uH_solution.block(1).reinit(n_dg);
        uH_solution.collect_sizes();
    typename DoFHandler<dim>::active_cell_iterator
    cell  = dof_handler_transp.begin_active(),
    endc  = dof_handler_transp.end();
    for(; cell!=endc; ++cell)
    {
        cell_matrix_mass = 0;
        cell_rhs         = 0;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        fe_values.get_function_values(EG_solution, solution_values_transp);
        for(const unsigned int i : fe_values.dof_indices())
        {
            for(const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                cell_rhs(i) += fe_values[cg].value(i, q_index)  *
                                (solution_values_transp[q_index][0] + solution_values_transp[q_index][1])  *
                                fe_values.JxW(q_index);
            }/*End of quadrature*/

            for(const unsigned int j : fe_values.dof_indices())
            {
                for(const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    cell_matrix_mass(i,j) += fe_values[cg].value(j, q_index) *
                                                fe_values[cg].value(i, q_index) *
                                                fe_values.JxW(q_index);
                }/*End of quadrature*/
            }/*End of j-loop*/
        }/*End of i-loop*/

        for(const unsigned int i : fe_values.dof_indices())
        {
            system_rhs_EG(local_dof_indices[i]) += cell_rhs(i);
            for(const unsigned int j : fe_values.dof_indices())
            {
                system_matrix_EG.add(local_dof_indices[i], 
                                    local_dof_indices[j], 
                                    cell_matrix_mass(i,j));
            }
        }
    }/*End of cell-loop*/

    SolverControl            solver_control(100000, 1e-12);
    PreconditionSSOR<SparseMatrix<double> > precondition;
    precondition.initialize(system_matrix_EG.block(0,0), 
                            PreconditionSSOR<SparseMatrix<double>>::AdditionalData(.6));
    SolverGMRES<Vector<double>> solver(solver_control);
    solver.solve(system_matrix_EG.block(0,0), 
                    uH_solution.block(0), 
                    system_rhs_EG.block(0), precondition);
    // std::cout << "uH_solution.block(0) = \n\t" << uH_solution.block(0) << std::endl;

    BlockVector<double> uL_solution;
            uL_solution.reinit(2);
            uL_solution.block(0).reinit(n_cg);
            uL_solution.block(1).reinit(n_dg);
            uL_solution.collect_sizes();
    for(unsigned int i = 0; i < n_cg; ++i)
    {
        double m_i = mass_matrix_transp(i,i);

        IndexSet DOF(dof_handler_transp.n_dofs());
        BlockSparseMatrix<double>::const_iterator
            index = mass_matrix_transp.begin(i);
        BlockSparseMatrix<double>::const_iterator
            index_end = mass_matrix_transp.end(i);
        for(; index!=index_end; ++index)
        {
            unsigned int e = index->column();
            if(e >= n_cg) DOF.add_index(e);
        }
        for(auto e : DOF)
        {
            double m_ei = mass_matrix_transp(i,e);
            uL_solution(i) += (1./m_i) * m_ei * solution_transp_EG(e);
        }
    }

    ////// Third Step: calculated ftilde_ei
        cell = dof_handler_transp.begin_active();
        Vector<double>     ftilde_ei(dofs_per_cell);
        for(; cell!=endc; ++cell)
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            ftilde_ei = 0;
            unsigned int e = local_dof_indices[4];

            fe_values.get_function_values(EG_solution, solution_values_transp);
            fe_values[cg].get_function_values(uH_solution, solution_values_transp_cg);
            for(const unsigned int i : fe_values.dof_indices())
            {
                if(i < 4)
                {
                    double uH_i     = uH_solution(local_dof_indices[i]);
                    double utilde_e = solution_transp_EG(e);
                    for(const unsigned int q_index : fe_values.quadrature_point_indices())
                    {
                        ftilde_ei(i) += fe_values[cg].value(i, q_index) * 
                                        ( uH_i- solution_values_transp_cg[q_index] 
                                        + solution_values_transp[q_index][0] + solution_values_transp[q_index][1] - utilde_e )
                                        * fe_values.JxW(q_index);
                    }/*End of quadrature*/
                }
            }/*End of i-loop*/
            for(const unsigned int i : fe_values.dof_indices())
            {
                system_matrix_EG.set(e, local_dof_indices[i], ftilde_ei(i));
            }
        }
        
        
        BlockVector<double> FCT_max;
        BlockVector<double> FCT_min;
            FCT_max.reinit(2);
            FCT_max.block(0).reinit(n_cg);
            FCT_max.block(1).reinit(n_dg);
            FCT_max.collect_sizes();

            FCT_min.reinit(2);
            FCT_min.block(0).reinit(n_cg);
            FCT_min.block(1).reinit(n_dg);
            FCT_min.collect_sizes();
       for(unsigned int i = 0; i < n_cg; ++i)
        {
            IndexSet DOF(dof_handler_transp.n_dofs());
            BlockSparseMatrix<double>::const_iterator index     = mass_matrix_transp.begin(i);
            BlockSparseMatrix<double>::const_iterator index_end = mass_matrix_transp.end(i);
            std::vector<double> u_solution;
            for(; index!=index_end; ++index)
            {
                unsigned int j = index->column();
                DOF.add_index(j);
            }
            for(auto k : DOF)
            {
                u_solution.push_back(solution_transp_EG(k));
            }
            auto MAX_dof = std::max_element(std::begin(u_solution),
                                        std::end(u_solution));
            double MAX_value = *MAX_dof;
            /////
            auto MIN_dof = std::min_element(std::begin(u_solution),
                                        std::end(u_solution));
            double MIN_value = *MIN_dof;
            FCT_max(i) = MAX_value;
            FCT_min(i) = MIN_value;
        }
        


        BlockVector<double> gamma;
            gamma.reinit(2);
            gamma.block(0).reinit(n_cg);
            gamma.block(1).reinit(n_dg);
            gamma.collect_sizes();
        for(unsigned int e = n_cg; e < n_cg + n_dg; e++)
        {
            BlockSparseMatrix<double>::const_iterator index = mass_matrix_transp.begin(e);
            BlockSparseMatrix<double>::const_iterator index_end = mass_matrix_transp.end(e);
            IndexSet DOF(dof_handler_transp.n_dofs());
            for(; index!=index_end; ++index)
            {
                unsigned int i = index->column();
                if(i < n_cg)
                {
                    DOF.add_index(i);
                }
            }
            std::vector<double> gamma_temp;
            for(auto i : DOF)
            {
                double ftilde_ei = system_matrix_EG(e,i);
                double m_ei      = mass_matrix_transp(i,e);
                double uL_i      = uL_solution(i);
                if(ftilde_ei > 0)
                {
                    double value = 1;
                    value = std::min(1., m_ei * (FCT_max(i) - uL_i)/(ftilde_ei + 1e-12));
                    gamma_temp.push_back(value);
                }else if(ftilde_ei < 0)
                {
                    double value = 1;
                    value = std::min(1., m_ei * (FCT_min(i) - uL_i)/(ftilde_ei - 1e-12));
                    gamma_temp.push_back(value);
                }else{
                    double value = 1;
                    gamma_temp.push_back(value);
                }
            }
            auto MIN_gamma = std::min_element(std::begin(gamma_temp),
                                                std::end(gamma_temp));
            gamma(e) = *MIN_gamma;
        }
        ustar_solution = uL_solution;
        for(unsigned int i = 0; i < n_cg; ++i)
        {
            double m_i = mass_matrix_transp(i,i);
            BlockSparseMatrix<double>::const_iterator
                index = mass_matrix_transp.begin(i);
            BlockSparseMatrix<double>::const_iterator
                index_end = mass_matrix_transp.end(i);
            IndexSet DOF(dof_handler_transp.n_dofs());
            for(; index!=index_end; ++index)
            {
                unsigned int e = index->column();
                if(e >= n_cg) DOF.add_index(e);
            }
            for(auto e : DOF)
            {
                ustar_solution(i) += (1./m_i) * gamma(e) * system_matrix_EG(e,i);
            }
        }
        // std::cout << "ustar_solution.block(0) = \n\t" << ustar_solution.block(0) << std::endl;
        if(bOutput){
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler_transp);
            std::vector<std::string> names;
            names.push_back("u");
            names.push_back("z");
            data_out.add_data_vector(ustar_solution, names);
            data_out.build_patches();
            data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
            const std::string filename = 
        // Utilities::int_to_string(task_number, 1)
                task_name+
                Utilities::int_to_string(RefineTime,1)+
                +"_FCR_TestCase" + std::to_string(PrescribedSolution::test_case)
                +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
            std::ofstream output(filename);
            data_out.write_vtk(output);
        }
        
}

template<int dim>
void ScalarHyperbolic<dim>::run_KPP()
{
    make_grid_KPP();
    setup_system();
    Assemble_Mass_Matrix();

    time =          0.0;
    timestep_number = 0;
    PrescribedSolution::Transp_EG::Exact_transp<dim> exact_transp;
    exact_transp.set_time(time);
    std::cout << "timestep_number = " << timestep_number << std::endl;
    std::cout << "time = " << time << std::endl;
    VectorTools::interpolate(dof_handler_transp,
                             exact_transp,
                             solution_transp_EG);
    old_old_solution_transp_EG = old_solution_transp_EG;
    old_solution_transp_EG     = solution_transp_EG;


    BlockVector<double> Exact_Value;
    Exact_Value = solution_transp_EG;
    // if(bOutput)
    // {
    //   DataOut<dim> data_out;
    //   std::vector<std::string> EG_names;
    //   EG_names.push_back("u_c");
    //   EG_names.push_back("ubar");
    //   data_out.attach_dof_handler(dof_handler_transp);
    //   data_out.add_data_vector(Exact_Value, EG_names);
    //   data_out.build_patches();
    //   data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    //   const std::string filename = 
    //     // Utilities::int_to_string(task_number, 1)
    //     Utilities::int_to_string(RefineTime,1)+
    //     "Exact_TestCase_" + std::to_string(PrescribedSolution::test_case)
    //     +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
    //   std::ofstream output(filename);
    //   data_out.write_vtk(output);
    // }

    while(
        time < tmax && time + time_step <= tmax + time_step / 2.
        // timestep_number < 5
    ){
        time += time_step;
        ++timestep_number;
        if(timestep_number % 100 == 0 ) bOutput =true;
        else bOutput = false;

        std::cout << "task_name = " << task_name << std::endl;
        std::cout << "Function: " << PrescribedSolution::test_case << std::endl;
        std::cout << "\ttimestep_number = " << timestep_number << std::endl;
        std::cout << "\ttime = " << time << std::endl;
        
        {/*SSP-RK2*/
            BlockVector<double> solution_transp_EG_RK0;
            solution_transp_EG_RK0 = old_solution_transp_EG;

            BlockVector<double> solution_transp_EG_RK1;
            solution_transp_EG_RK1 = EulerUpdate(solution_transp_EG_RK0,1);

            BlockVector<double> solution_transp_EG_RK2;
            solution_transp_EG_RK2 = EulerUpdate(solution_transp_EG_RK1,2);

            solution_transp_EG = 0.5 * (solution_transp_EG_RK0 + solution_transp_EG_RK2);
        }
        Compute_EG_Solution();
        FCR_Process();
        Error();
        Output();
        FluxLimiter_Output();

        old_old_solution_transp_EG = old_solution_transp_EG;
        old_solution_transp_EG = solution_transp_EG;


        exact_transp.set_time(time);
        VectorTools::interpolate(dof_handler_transp,
                             exact_transp,
                             	// ZeroFunction<dim>(2),
                             Exact_Value);
        if(bOutput)
        {
            DataOut<dim> data_out;
            std::vector<std::string> EG_names;
            EG_names.push_back("u_c");
            EG_names.push_back("ubar");
            data_out.attach_dof_handler(dof_handler_transp);
            data_out.add_data_vector(Exact_Value, EG_names);
            data_out.build_patches();
            data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
            const std::string filename = 
                // Utilities::int_to_string(task_number, 1)
                Utilities::int_to_string(RefineTime,1)+
                "Exact_TestCase_" + std::to_string(PrescribedSolution::test_case)
                +"_"+Utilities::int_to_string(timestep_number, 5) + ".vtk";
            std::ofstream output(filename);
            data_out.write_vtk(output);
        }
    }
    
}

int main()
{



    const int dim = 2;
    double Initial_time_step  =  0.001;
    int Initial_RefineTime = 9;
    double tmax               = 1.0;

    if(PrescribedSolution::test_case == 10) tmax = 0.5;


    bool   bOutput    = true;
    int task_number = 5;
    std::string task_name = "Debug_";
    {
        if(task_number == 0) task_name += "low_order";
        else if(task_number == 1) task_name += "high_order";
        else if(task_number == 2) task_name += "bp_MCL";
        else if(task_number == 3) task_name += "es";
        else if(task_number == 4) task_name += "bp_es_clip";
        else if(task_number == 5) task_name += "bpes_cg_bp_average_extra_ES_CG";
        else if(task_number == 6) task_name += "FCTBP_ES";
        else if(task_number == 7) task_name += "Nov04";
        else if(task_number == 8) task_name += "Nov04_es_stabilization_cg";
    }


    double time_step  = Initial_time_step ;
    int RefineTime  = Initial_RefineTime;
    ScalarHyperbolic<dim> Exp1(time_step, tmax, RefineTime,
                                bOutput, 
                                task_number,
                                task_name );
    Exp1.run_KPP();


    
}

