#include <iostream>
#include <random>
#include <iomanip>
#include <fstream>
using namespace std;

mt19937 rng(42);
const int NEIGHBOR = 9;
const int CX[NEIGHBOR] ={0, 1, 0, -1, 0, 1, 1, -1, -1};
const int CY[NEIGHBOR] ={0, 0, -1, 0, 1, -1, 1, -1, 1}; //add negative sign for matching array coordinate
const int OPP[NEIGHBOR] ={0, 3, 4, 1, 2, 8, 7, 6, 5}; //add negative sign for matching array coordinate
const float weight[NEIGHBOR] ={4/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/36.0f, 1/36.0f, 1/36.0f, 1/36.0f};
float velx_inlet = 0.0f;
float relax_time = 0.0f;


void init_array(float*** cur_F, float*** new_F, float** density, float** velocity_x, float** velocity_y, bool**  solid, int Nx, int Ny);
void delete_array(float*** cur_F, float*** new_F, float** density, float** velocity_x, float** velocity_y, bool**  solid, int Nx, int Ny);
float cal_feq(int cx, int cy, float w, float density, float velx, float vely);
void run_case1(const int steps, float*** cur_F, float*** new_F, float** density,
    float** velocity_x, float** velocity_y, bool** solid, const int Nx, const int Ny, float avg_pressure);
void run_case2(const int steps, float*** cur_F, float*** new_F, float** density,
    float** velocity_x, float** velocity_y, bool** solid, const int Nx, const int Ny, float avg_pressure);
void save_frame(const int case_id, const int step, float* const* density, float* const* velx, float* const* vely, const int Nx, const int Ny);
void wind_tunnel_boundary_condition(const float velx_in, float*** F, float* const* density, float** velx, float** vely, bool* const* solid, const int Nx, const int Ny);

void collision(float*** new_F, float*** cur_F, float** density, float& avg_pressure, float** velocity_x, float** velocity_y, bool* const* solid, int Nx, int Ny);
void create_cylinder(bool** solid, const int Nx, const int Ny);
void init_Acoustic_pulse(float*** F, float** density, float** velocity_x, float** velocity_y, int Nx, int Ny);
void init_Karman_vortex_street(float*** F, float** density, float** velocity_x, float** velocity_y, bool* const* solid, int Nx, int Ny);

void streaming_periodic(float*** F_prime, float*** new_F, bool* const* solid, int Nx, int Ny);
void streaming(float*** F_prime, float*** new_F, bool* const* solid, int Nx, int Ny);

void run_case1(const int steps, float*** cur_F, float*** new_F, float** density,
    float** velocity_x, float** velocity_y, bool** solid, const int Nx, const int Ny, float avg_pressure)
{
    init_Acoustic_pulse(cur_F, density, velocity_x, velocity_y, Nx, Ny);
    cout << "run case 1" << endl;

    for (int step=0; step<steps; step++)
    {
        collision(new_F, cur_F, density, avg_pressure, velocity_x, velocity_y, solid, Nx, Ny);
        streaming_periodic(cur_F, new_F, solid, Nx, Ny);
        // // cal_vortivity(vortivity, velocity_x, velocity_y, Nx, Ny);
        save_frame(1, step, density, velocity_x, velocity_y, Nx, Ny);

        std::swap(cur_F, new_F);
    }

    std::ofstream fout("./data1/info.txt");
    fout << Nx << " " << Ny << "\n";
    fout.close();
}

void run_case2(const int steps, float*** cur_F, float*** new_F, float** density,
    float** velocity_x, float** velocity_y, bool** solid, const int Nx, const int Ny, float avg_pressure)
{   
    std::ofstream fout("./data2/info.txt");
    fout << Nx << " " << Ny << "\n";
    fout.close();

    const float velx_in = 0.1f;
    create_cylinder(solid, Nx, Ny);
    init_Karman_vortex_street(cur_F, density, velocity_x, velocity_y, solid, Nx, Ny);
    cout << "run case2" << endl;

    int frame = 0;
    for (int step=0; step<steps; step++)
    {
        collision(new_F, cur_F, density, avg_pressure, velocity_x, velocity_y, solid, Nx, Ny);
        streaming(cur_F, new_F, solid, Nx, Ny);
        wind_tunnel_boundary_condition(velx_in, new_F, density, velocity_x, velocity_y, solid, Nx, Ny);

        // // cal_vortivity(vortivity, velocity_x, velocity_y, Nx, Ny);
        if (step % 25 ==0){
            save_frame(2, frame, density, velocity_x, velocity_y, Nx, Ny);
            frame++;
            cout << "step: " << step << "frame: " << frame << endl;
        }

        std::swap(cur_F, new_F);
    }



}

float cal_feq(int cx, int cy, float w, float density, float velx, float vely)
{
    float dot_c_u = velx*cx + vely*cy;
    float vel2 =velx*velx + vely*vely;
    float f_eq = w * density * (1 + 3*dot_c_u + 9.0f/2.0f*dot_c_u*dot_c_u - 3.0f/2.0f*vel2);
    return f_eq;
}

void create_cylinder(bool** solid, const int Nx, const int Ny)
{
    int cx = 150, cy = 100;
    int R = 20;

    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            int dx = i - cx;
            int dy = j - cy;

            if ((dx * dx + dy * dy <= R * R) || // cylinder
                (j == 0) || (j == Ny-1))  // wind tunnel wall
            {
                solid[j][i] = true;
            }
        }
    }    
}

void init_Acoustic_pulse(float*** F, float** density, float** velocity_x, float** velocity_y, int Nx, int Ny){
    // configutation
    relax_time = 1.2f;

    double xc = 0.5 * Nx;
    double yc = 0.5 * Ny;

    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            int idx = j * Nx + i;
            double dx = i - xc;
            double dy = j - yc;
            double r2 = dx*dx + dy*dy;
            density[j][i] = 1.0f + 0.05f * std::exp(-r2/(2.0f*15.0f*15.0f));
            velocity_x[j][i] = 0.0f;
            velocity_y[j][i] = 0.0f;
            for (int k=0; k<9; k++)
            {   
                F[j][i][k] = cal_feq(CX[k], CY[k], weight[k], density[j][i], velocity_x[j][i], velocity_y[j][i]); //init
            }
        }
    }
}

void init_Karman_vortex_street(float*** F, float** density, float** velocity_x, float** velocity_y, bool* const* solid, int Nx, int Ny){
    // configutation
    velx_inlet = 0.1f;
    relax_time = 0.58f;

    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            density[j][i] = 1.0f;
            velocity_x[j][i] = solid[j][i] ? 0.0f : velx_inlet;
            velocity_y[j][i] = 0.0f;
            for (int k = 0; k < NEIGHBOR; ++k)
            {
                F[j][i][k] = cal_feq(CX[k], CY[k], weight[k], density[j][i], velocity_x[j][i], velocity_y[j][i]); //init
                if ((i==0) && (j==0 || j==1)) 
                    cout << j << ", s:" << solid[j][i] << ", "<< CX[k] << ", " <<  CY[k] << ", " << weight[k] << ", " << density[j][i] << ", "  << velocity_x[j][i] << ", "  <<velocity_y[j][i] << endl;

            }
        }
    }
}

void collision(float*** new_F, float*** cur_F, float** density, float& avg_pressure, float** velocity_x, float** velocity_y, bool* const* solid, int Nx, int Ny)
{
    for (int j=0; j<Ny; j++)
    { 
        for (int i=0; i<Nx; i++)
        {
            if (solid[j][i]) {
                velocity_x[j][i] = 0;
                velocity_y[j][i] = 0;
                continue;
            }
            density[j][i] = 0;
            velocity_x[j][i] = 0;
            velocity_y[j][i] = 0;

            float momentum_x = 0.0f;
            float momentum_y = 0.0f;
            for (int k=0; k<9; k++)
            {
                int cx = CX[k];
                int cy = CY[k];
                density[j][i] += cur_F[j][i][k];
                momentum_x += cx * cur_F[j][i][k];
                momentum_y += cy * cur_F[j][i][k];
            }
            velocity_x[j][i] = momentum_x/density[j][i];
            velocity_y[j][i] = momentum_y/density[j][i];

            // calculate f_eq
            for (int k=0; k<9; k++)
            {
                float omega = (1.0f/relax_time);
                float f_eq =  cal_feq(CX[k], CY[k], weight[k], density[j][i], velocity_x[j][i], velocity_y[j][i]);
                cur_F[j][i][k] += omega * (f_eq - cur_F[j][i][k]);
            }
        }
    }
}

bool check_boundary(int x, int y, int Nx, int Ny)
{
    return x >= 0 && x < Nx && y >= 0 && y < Ny;
}

void streaming_periodic(float*** F_prime, float*** new_F, bool* const* solid, int Nx, int Ny)
{
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            for (int k = 0; k < 9; k++) {

                int ip = (i - CX[k] + Nx) % Nx;
                int jp = (j - CY[k] + Ny) % Ny;
                new_F[j][i][k] = F_prime[jp][ip][k];
            }
        }
    }
}

void streaming(float*** F_prime, float*** new_F, bool* const* solid, int Nx, int Ny)
{
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            for (int k = 0; k < 9; k++) {
                int ip = i - CX[k];
                int jp = j - CY[k];
                if (check_boundary(ip, jp, Nx, Ny))
                {
                    new_F[j][i][k] = solid[jp][ip] ? F_prime[j][i][OPP[k]] : F_prime[jp][ip][k];
                }
            }
        }
    }
}

void save_frame(const int case_id, const int frame, float* const* density, float* const* velx, float* const* vely, const int Nx, const int Ny)
{
    std::stringstream ss;
    ss << "./data" << case_id << "/frame_" << std::setw(6) << std::setfill('0') << frame << ".bin";
    std::ofstream fout(ss.str(), std::ios::binary);
    for (int y = 0; y < Ny; ++y) {
        fout.write(
        reinterpret_cast<const char*>(density[y]),
        Nx * sizeof(float)
    );
    }
    for (int y = 0; y < Ny; ++y) {
        fout.write(
            reinterpret_cast<const char*>(velx[y]),
            Nx * sizeof(float)
        );
    }
    for (int y = 0; y < Ny; ++y) {
        fout.write(
            reinterpret_cast<const char*>(vely[y]),
            Nx * sizeof(float)
        );
    }

    fout.close();
}

void wind_tunnel_boundary_condition(const float velx_in, float*** F, float* const* density, float** velx, float** vely, bool* const* solid, const int Nx, const int Ny)
{
    // inlet
    // priority: solid > boudary condition
    for (int j = 1; j < Ny-1; ++j) {
        if (solid[j][0]) continue;
        float rho = 0.0f;
        for (int k = 0; k < 9; ++k)
            rho += F[j][0][k];

        float uy = (F[j][0][4] + F[j][0][6] + F[j][0][8] - F[j][0][2] - F[j][0][5] - F[j][0][7] ) / rho;
        F[j][0][1] = cal_feq(CX[1], CY[1], weight[1], rho, velx_in,  uy);
        F[j][0][5] = cal_feq(CX[5], CY[5], weight[5], rho, velx_in,  uy);
        F[j][0][6] = cal_feq(CX[6], CY[6], weight[6], rho, velx_in,  uy);
    }

    // outlet
    for (int j = 1; j < Ny-1; ++j) {
        for (int k = 0; k < NEIGHBOR; ++k) {
            F[j][Nx-1][k] = F[j][Nx-2][k];
        }
    }
}

int main(int argc, char* argv[]) {
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

    std::cout << "Nx    = " << Nx;
    std::cout << " Ny    = " << Ny;
    std::cout << " steps = " << steps << '\n';

    float*** cur_F = new float**[Ny];
    float*** new_F = new float**[Ny];
    float** density    = new float*[Ny];
    float** vortivity  = new float*[Ny];
    float** velocity_x = new float*[Ny];
    float** velocity_y = new float*[Ny];
    float** pressure   = new float*[Ny];
    bool**  solid      = new bool*[Ny];
    init_array(new_F, cur_F, density, velocity_x, velocity_y, solid, Nx, Ny);
    float avg_pressure = 0.0f;
    // run_case1(steps, cur_F, new_F, density, velocity_x, velocity_y, solid, Nx, Ny, avg_pressure);
    run_case2(steps, cur_F, new_F, density, velocity_x, velocity_y, solid, Nx, Ny, avg_pressure);

    delete_array(new_F, cur_F, density, velocity_x, velocity_y, solid, Nx, Ny);
    return 0;
}

void init_array(float*** cur_F, float*** new_F, float** density, float** velocity_x, float** velocity_y, bool**  solid, int Nx, int Ny)
{
    for (int y = 0; y < Ny; ++y) {
        cur_F[y] = new float*[Nx];
        new_F[y] = new float*[Nx];

        for (int x = 0; x < Nx; ++x) {
            cur_F[y][x] = new float[NEIGHBOR]();
            new_F[y][x] = new float[NEIGHBOR]();
        }
    }

    // Macroscopic variables: [y][x]
    // float** density    = new float*[Ny];
    // // float** vortivity  = new float*[Ny];
    // float** velocity_x = new float*[Ny];
    // float** velocity_y = new float*[Ny];
    // // float** pressure   = new float*[Ny];
    // bool**  solid      = new bool*[Ny];

    for (int y = 0; y < Ny; ++y) {
        density[y]    = new float[Nx]();
        // vortivity[y]  = new float[Nx]();
        velocity_x[y] = new float[Nx]();
        velocity_y[y] = new float[Nx]();
        // pressure[y]   = new float[Nx]();
        solid[y]      = new bool[Nx]();
    }
}

void delete_array(float*** cur_F, float*** new_F, float** density, float** velocity_x, float** velocity_y, bool**  solid, int Nx, int Ny)
{
    for (int y = 0; y < Ny; ++y)
    {
        for (int x = 0; x < Nx; ++x)
        {
            delete[] cur_F[y][x];
            delete[] new_F[y][x];
        }

        delete[] cur_F[y];
        delete[] new_F[y];
    
        delete[] density[y];
        delete[] velocity_x[y];
        delete[] velocity_y[y];
        delete[] solid[y];
    }

    delete[] cur_F;
    delete[] new_F;
}