#include <iomanip>
#include <fstream>
#include <iostream>
#include <Kokkos_Core.hpp>

using namespace std;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using DeviceView = Kokkos::View<float*>;
using HostView = DeviceView::host_mirror_type;

const int NEIGHBOR = 9;
// const  Kokkos::Array<int, NEIGHBOR> CX = {0, 1, 0, -1, 0, 1, 1, -1, -1};
// const  Kokkos::Array<int, NEIGHBOR> CY = {0, 0, -1, 0, 1, -1, 1, -1, 1};
// const  Kokkos::Array<int, NEIGHBOR> OPP = {0, 3, 4, 1, 2, 8, 7, 6, 5};
// const  Kokkos::Array<float, NEIGHBOR> WEIGHT = {4/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/36.0f, 1/36.0f, 1/36.0f, 1/36.0f};
static float velx_inlet = 0.0f;
static float relax_time = 0.0f;
static float omega = 0.0f;


struct D2Q9
{
    Kokkos::Array<int, NEIGHBOR> cx;
    Kokkos::Array<int, NEIGHBOR> cy;
    Kokkos::Array<int, NEIGHBOR> opp;
    Kokkos::Array<float, NEIGHBOR> weight;
};

constexpr D2Q9 make_d2q9()
{
    return {
        {0, 1, 0, -1, 0, 1, 1, -1, -1},
        {0, 0, -1, 0, 1, -1, 1, -1, 1},
        {0, 3, 4, 1, 2, 8, 7, 6, 5},
        {4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f}
    };
}

void create_cylinder(Kokkos::View<bool*> solid, const int Nx, const int Ny);
void run_case2(const int steps, Kokkos::View<float*> cur_F, Kokkos::View<float*> new_F, Kokkos::View<float*> density,
    Kokkos::View<float*> velocity_x, Kokkos::View<float*> velocity_y, Kokkos::View<bool*> solid, const int Nx, const int Ny);

void save_frame(const int case_id, const int frame, HostView& density, HostView& velx, HostView& vely, const int N)
{
    std::stringstream ss;
    ss << "./data_GPU_" << case_id << "/frame_" << std::setw(6) << std::setfill('0') << frame << ".bin";
    std::ofstream fout(ss.str(), std::ios::binary);
    fout.write(reinterpret_cast<const char*>(density.data()), N * sizeof(float));
    fout.write(reinterpret_cast<const char*>(velx.data()), N * sizeof(float));
    fout.write(reinterpret_cast<const char*>(vely.data()), N * sizeof(float));
    fout.close();
}

    
int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: "
                  << argv[0]
                  << " Nx Ny steps\n";
        return 1;
    }

    const int Nx = std::atoi(argv[1]);
    const int Ny = std::atoi(argv[2]);
    const int steps = std::atoi(argv[3]);

    std::cout << "Nx = " << Nx;
    std::cout << " Ny = " << Ny;
    std::cout << " steps = " << steps << '\n';

    Kokkos::initialize(argc, argv);
    {

        printf("ExecSpace = %s\n", ExecSpace::name());
        
        const int N = Nx*Ny;
        Kokkos::View<float*> cur_F("cur_F", N*NEIGHBOR);
        Kokkos::View<float*> new_F("new_F", N*NEIGHBOR);
        Kokkos::View<float*> density("density", N);
        Kokkos::View<float*> velocity_x("velocity_x", N);
        Kokkos::View<float*> velocity_y("velocity_y", N);
        Kokkos::View<bool*> solid("solid", N);

        using ViewType = decltype(density);
        printf("MemSpace = %s\n", typeid(typename decltype(density)::memory_space).name());

        run_case2(steps, cur_F, new_F, density, velocity_x, velocity_y, solid, Nx, Ny);
    }
    Kokkos::finalize();

    return 0;
}

void create_cylinder(Kokkos::View<bool*> solid, const int Nx, const int Ny)
{
    int cx = 150, cy = 100;
    int R = 20;
    int N = Nx*Ny;
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int node)
    {
        int i = node % Nx;
        int j = node / Nx;
        int dx = i - cx;
        int dy = j - cy;
        if ((dx * dx + dy * dy <= R * R) || // cylinder
            (j == 0) || (j == Ny-1))  // wind tunnel wall
        {
            solid(node) = true;
        }
        else
        {
            solid(node) = false;
        }
    });
}

KOKKOS_INLINE_FUNCTION
float cal_feq(int cx, int cy, float w, float density, float velx, float vely)
{
    float dot_c_u = velx*cx + vely*cy;
    float vel2 =velx*velx + vely*vely;
    float f_eq = w * density * (1.0f + 3.0f*dot_c_u + 9.0f/2.0f*dot_c_u*dot_c_u - 3.0f/2.0f*vel2);

    return f_eq;
}

void init_Karman_vortex_street(Kokkos::View<float*> F, Kokkos::View<float*> density,
    Kokkos::View<float*> velocity_x, Kokkos::View<float*> velocity_y, Kokkos::View<bool*> solid, const int N)
{
    const float inlet = velx_inlet;
    constexpr D2Q9 lattice = make_d2q9();
    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int node)
    {
        density(node) = 1.0f;
        velocity_x(node) = solid(node) ? 0.0f : inlet;
        velocity_y(node) = 0.0f;
        for (int k = 0; k < NEIGHBOR; ++k)
        {
            F(k*N + node) = cal_feq(lattice.cx[k], lattice.cy[k], lattice.weight[k], density(node), velocity_x(node), velocity_y(node)); //init
            // F(k*N + node) = k*N + node;
        }

    });
}

void collision(Kokkos::View<float*> cur_F, Kokkos::View<float*> density,
    Kokkos::View<float*> velocity_x, Kokkos::View<float*> velocity_y,
    Kokkos::View<bool*> solid, const int Nx, const int Ny)
{
    const int N = Nx*Ny;
    const float local_omega = omega;
    constexpr D2Q9 lattice = make_d2q9();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int node)
    {
        if (solid(node))
        {
            velocity_x(node) = 0.0f;
            velocity_y(node) = 0.0f;
        }
        else
        {
            float rho = 0.0f;
            float momentum_x = 0.0f;
            float momentum_y = 0.0f;
            for (int k=0; k<NEIGHBOR; k++)
            {                
                int pdf_idx = k*N + node;
                rho += cur_F[pdf_idx];
                momentum_x += lattice.cx[k] * cur_F[pdf_idx];
                momentum_y += lattice.cy[k] * cur_F[pdf_idx];
            }
            density(node) = rho;
            velocity_x(node) = momentum_x/rho;
            velocity_y(node) = momentum_y/rho;

            // calculate f_eq
            for (int k=0; k<9; k++)
            {
                int pdf_idx = k*N + node;
                float f_eq =  cal_feq(lattice.cx[k], lattice.cy[k], lattice.weight[k], density(node), velocity_x(node), velocity_y(node));
                cur_F(pdf_idx) += local_omega * (f_eq - cur_F(pdf_idx));
            }
        }
    });
}

KOKKOS_INLINE_FUNCTION
bool check_boundary(int x, int y, int Nx, int Ny)
{
    return x >= 0 && x < Nx && y >= 0 && y < Ny;
}

void streaming(Kokkos::View<float*> F_prime, Kokkos::View<float*> new_F, Kokkos::View<bool*> solid, int Nx, int Ny)
{
    int N = Nx * Ny;
    constexpr D2Q9 lattice = make_d2q9();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int node)
    {
        int i = node % Nx;
        int j = node / Nx;
        for (int k = 0; k < 9; k++)
        {
            int pdf_idx = k*N + node;
            int pdf_OP_idx = lattice.opp[k]*N + node;

            int ip = i - lattice.cx[k];
            int jp = j - lattice.cy[k];
            int neighbor_node = jp*Nx + ip;
            int pdf_neighbor_node = k*N + neighbor_node;
            if (check_boundary(ip, jp, Nx, Ny))
            {
                new_F[pdf_idx] = solid[neighbor_node] ? F_prime[pdf_OP_idx] : F_prime[pdf_neighbor_node];
            }
            else
            {
                // update by boudary condition
            }
        }
    });
}

void wind_tunnel_boundary_condition(Kokkos::View<float*> F, Kokkos::View<float*> density,
    Kokkos::View<float*> velx, Kokkos::View<float*> vely, Kokkos::View<bool*> solid, const int Nx, const int Ny)
{
    const float velx_in = velx_inlet;
    constexpr D2Q9 lattice = make_d2q9();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(1, Ny-1), KOKKOS_LAMBDA(int j)
    {
        // inlet
        // priority: solid > boudary condition
        int N = Nx*Ny;
        int inlet_node_idx = j*Nx + 0; // first column
        if (solid(inlet_node_idx))
        {
            //continue;
        }
        else //fluid
        {
            float rho = 0.0f;
            float momentum_y = 0.0f;
            for (int k = 0; k < NEIGHBOR; ++k)
            {   
                int inlet_pdf_idx = k*N + inlet_node_idx;
                // recalculate rho for inlet
                rho += F[inlet_pdf_idx];
                // recalculate vely for inlet
                momentum_y += lattice.cy[k] * F[inlet_pdf_idx];

                // Outlet: last column = second last column
                F[k*N + j*Nx + (Nx - 1)] = F[k*N + j*Nx + (Nx-2)];
            }

            // Inlet
            float uy = momentum_y / rho;
            F[inlet_node_idx + 1*Nx*Ny] = cal_feq(lattice.cx[1], lattice.cy[1], lattice.weight[1], rho, velx_in, uy);
            F[inlet_node_idx + 5*Nx*Ny] = cal_feq(lattice.cx[5], lattice.cy[5], lattice.weight[5], rho, velx_in, uy);
            F[inlet_node_idx + 6*Nx*Ny] = cal_feq(lattice.cx[6], lattice.cy[6], lattice.weight[6], rho, velx_in, uy);
        }
    });
}

void run_case2(const int steps, Kokkos::View<float*> cur_F, Kokkos::View<float*> new_F, Kokkos::View<float*> density,
    Kokkos::View<float*> velocity_x, Kokkos::View<float*> velocity_y, Kokkos::View<bool*> solid, const int Nx, const int Ny)
{
    std::ofstream fout("./data_GPU_2/info.txt");
    fout << Nx << " " << Ny << "\n";
    fout.close();
    // create view for output frame
    auto density_host = Kokkos::create_mirror_view(density);
    auto velocity_x_host = Kokkos::create_mirror_view(velocity_x);
    auto velocity_y_host = Kokkos::create_mirror_view(velocity_y);
    auto solid_host = Kokkos::create_mirror_view(solid);

    // set up initial settings
    velx_inlet = 0.1f;
    relax_time = 0.58f;
    omega = (1.0f/relax_time);
    const int N = Nx*Ny;
    create_cylinder(solid, Nx, Ny);
    init_Karman_vortex_street(cur_F, density, velocity_x, velocity_y, solid, N);
    // for(int i=0; i<N; i++)
    // {
    // for(int j=0; j<NEIGHBOR; j++)
    //     printf(" %d, %.2f\n", j, cur_F[j*N + 803]);
    // // }
    // printf(" %d\n", solid[803]);

    int frame = 0;
    for (int step=0; step<steps; step++)
    {
        collision(cur_F, density, velocity_x, velocity_y, solid, Nx, Ny);
        // Kokkos::fence();
        // for(int k=0; k<NEIGHBOR; k++)
        //     printf(" %d, %d, %.0f, %.2f\n", step, k, density(803), cur_F(k*N + 803));

        streaming(cur_F, new_F, solid, Nx, Ny);
        wind_tunnel_boundary_condition(new_F, density, velocity_x, velocity_y, solid, Nx, Ny);

        if (step % 25 == 0){
            Kokkos::fence();
            
            // device to host
            Kokkos::deep_copy(density_host, density);
            Kokkos::deep_copy(velocity_x_host, velocity_x);
            Kokkos::deep_copy(velocity_y_host, velocity_y);
            save_frame(2, frame, density_host, velocity_x_host, velocity_y_host, N);
            frame++;
            printf(" %d, %d\n", step, frame);
            // for(int i=0; i<N; i++)
            // {
            //     for(int j=0; j<NEIGHBOR; j++)
            //         printf(" %d, %d, %.2f\n", i, j, velocity_x_host[j*N + i]);
            // }
        }

        std::swap(cur_F, new_F);
        
    }
}
