// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project


// EXERCISE mdrange Goals:
//   Use EXERCISE 2 Solution as a starting point
//   Initialize Views by using parallel_for pattern.
//   For the rank 2 View "A", use an MDRangePolicy

#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

double ans[4][4] = {{0.091667, 0.058333, 0.058333, -0.208333},
                    {0.058333, 0.091667, 0.058333, -0.208333},
                    {0.058333, 0.058333, 0.091667, -0.208333},
                    {-0.208333, -0.208333, -0.208333, 0.625000}};

template <class Layout, class Scalar>
void run(int n_min, int n_max, int nrepeat);
int main( int argc, char* argv[] )
{
  int nrepeat = 1;  // number of repeats of the test
  int layout = 0; // 0: right, 1:left
  int scalar = 0; // 0: float, 1:double

  int n_min = 1;
  int n_max = 1;

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( strcmp( argv[ i ], "-max" ) == 0 ) {
      n_max = atof( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-min" ) == 0 ) {
      n_min = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-layout" ) == 0 ) {
      layout = atoi( argv[ ++i ] );
    }
    else if ( strcmp( argv[ i ], "-scalar" ) == 0 ) {
      scalar = atoi( argv[ ++i ] );
    }
  }
  printf("n min=%d, n max=%d, repeat=%d\n", n_min, n_max, nrepeat);

  Kokkos::initialize( argc, argv );
  {
  
    if (layout == 0) // LayoutRight
    {
      using Layout = Kokkos::LayoutRight;
      if (scalar == 0){ // double
        printf("Layout: LayoutRight, precision: double\n");
        run<Layout, double>(n_min, n_max, nrepeat);
      }
      else if (scalar == 1){ // float
        printf("Layout: LayoutRight, precision: float\n");
        run<Layout, float>(n_min, n_max, nrepeat);
      }
    }
    else if (layout == 1) // LayoutLeft
    {
      using Layout = Kokkos::LayoutLeft;
      if (scalar == 0){ // double
        printf("Layout: LayoutLeft, precision: double\n");
        run<Layout, double>(n_min, n_max, nrepeat);
      }
      else if (scalar == 1){ // float
        printf("Layout: LayoutLeft, precision: float\n");
        run<Layout, float>(n_min, n_max, nrepeat);
      }
    }
    else
    {
      printf("Undefine layout");
    }
  }
  Kokkos::finalize();
  
  return 0;

}

template <class Layout, class Scalar>
void run(int n_min, int n_max, int nrepeat) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace = typename ExecSpace::memory_space;
  ExecSpace exec;

  printf("size of scalar: %d byte, ", sizeof(Scalar));
  printf("Execution Space: %s ", ExecSpace::name());
  printf("Concurrency: %u\n", exec.concurrency());
  printf("Memory Space: %s\n", MemSpace::name());
  printf("OpenMP available: %d\n", Kokkos::is_initialized());

  for (int N=n_min; N<n_max; N*=2){
    Kokkos::View<Scalar***, Layout, Kokkos::HostSpace> J("J", N, 3, 3); // host
    Kokkos::View<Scalar*, Kokkos::HostSpace> inv_J_det("inv_J_det", N); // host
    Kokkos::View<Scalar*, MemSpace> d_inv_J_det("d_inv_J_det", N); // device
    Kokkos::View<Scalar***, Layout, MemSpace> A("A", N, 4, 4); // device
    Kokkos::View<Scalar***, Layout, MemSpace> d_J("d_J", N, 3, 3); // device
    auto h_A = Kokkos::create_mirror_view(A); // host 

    // Initialize J matrix on host, note 2D indexing.
    Kokkos::parallel_for("J_init", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA (const int k){
        J( k, 0, 0 ) = 3;
        J( k, 0, 1 ) = 1;
        J( k, 0, 2 ) = 1;
        J( k, 1, 0 ) = 1;
        J( k, 1, 1 ) = 3;
        J( k, 1, 2 ) = 1;
        J( k, 2, 0 ) = 1;
        J( k, 2, 1 ) = 1;
        J( k, 2, 2 ) = 3;
    });

    Kokkos::parallel_for("inv_J_det", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA (int k){
        Scalar C0 = J(k,  1, 1) * J(k,  2, 2) - J(k,  1, 2) * J(k,  2, 1);
        Scalar C1 = J(k,  1, 2) * J(k,  2, 0) - J(k,  1, 0) * J(k,  2, 2);
        Scalar C2 = J(k,  1, 0) * J(k,  2, 1) - J(k,  1, 1) * J(k,  2, 0);
        inv_J_det(k) = J(k,  0, 0) * C0 + J(k,  0, 1) * C1 + J(k,  0, 2) * C2;
      });

    Kokkos::parallel_for("h_A_init",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>,ExecSpace>({0 ,0, 0}, {N, 4, 4}),
      KOKKOS_LAMBDA (int k, int j, int i){
        h_A( k, j, i ) = 0;
      });
    
    Kokkos::fence();
    Kokkos::Timer timer_h2d;
    Kokkos::deep_copy(A, h_A); // device to host
    Kokkos::deep_copy(d_inv_J_det, inv_J_det); // host to device
    Kokkos::deep_copy(d_J, J); // host to device
    // Wait for initalization to finish (after they are converted)
    Kokkos::fence();
    double tot_h2d_time = timer_h2d.seconds();

    // Timer products.
    Kokkos::Timer timer_compute;
    for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
      // execute on device
      Kokkos::parallel_for("cal_A", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int k){
        const Scalar c = (1./6.);
        Scalar d = c / d_inv_J_det(k);
        Scalar G0 = d * (d_J(k,  0,0) * d_J(k,  0,0) + d_J(k,  1,0) * d_J(k,  1,0) + d_J(k,  2,0) * d_J(k,  2,0));
        Scalar G1 = d * (d_J(k,  0,0) * d_J(k,  0,1) + d_J(k,  1,0) * d_J(k,  1,1) + d_J(k,  2,0) * d_J(k,  2,1));
        Scalar G2 = d * (d_J(k,  0,0) * d_J(k,  0,2) + d_J(k,  1,0) * d_J(k,  1,2) + d_J(k,  2,0) * d_J(k,  2,2));
        Scalar G3 = d * (d_J(k,  0,1) * d_J(k,  0,1) + d_J(k,  1,1) * d_J(k,  1,1) + d_J(k,  2,1) * d_J(k,  2,1));
        Scalar G4 = d * (d_J(k,  0,1) * d_J(k,  0,2) + d_J(k,  1,1) * d_J(k,  1,2) + d_J(k,  2,1) * d_J(k,  2,2));
        Scalar G5 = d * (d_J(k,  0,2) * d_J(k,  0,2) + d_J(k,  1,2) * d_J(k,  1,2) + d_J(k,  2,2) * d_J(k,  2,2));
        // 37 FLOPs
        // read 1+9 scalar

        A(k, 0, 0) = G0;
        A(k, 0, 1) = A(k, 1, 0) = G1;
        A(k, 0, 2) = A(k, 2, 0) = G2;
        A(k, 0, 3) = A(k, 3, 0) = -G0 - G1 - G2;
        A(k, 1, 1) = G3;
        A(k, 1, 2) = A(k, 2, 1) = G4;
        A(k, 1, 3) = A(k, 3, 1) = -G1 - G3 - G4;
        A(k, 2, 2) = G5;
        A(k, 2, 3) = A(k, 3, 2) = -G2 - G4 - G5;
        A(k, 3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5;
        // 6 + 8 = 14 FLOPs
        // total: 51 FLOPs
        // write 4*4 scalar
        // total (read + write) 26 scalar
      });
    }
    Kokkos::fence();
    double tot_compute_time = timer_compute.seconds();


    Kokkos::Timer timer_d2h;
    Kokkos::deep_copy(h_A, A); // copy value to host
    Kokkos::fence();
    double tot_d2h_time = timer_d2h.seconds();

    // check result
    // for (int k=0;k<N;k++){
    //   for (int i=0;i<4;i++){
    //     for (int j=0;j<4;j++){
    //       double round_output = std::round(h_A(k, i, j) * 1e6) / 1e6;
    //       double round_ans = std::round(ans[i][j] * 1e6) / 1e6;
    //       if (round_output != round_ans)
    //         printf("(k, i, j) = (%d, %d, %d) %.6f != %.6f", k, i, j, h_A(k, i, j), ans[i][j]);
    //     }
    //   }
    // }

    double Gbytes = double(sizeof(Scalar)*26.0f*N ); //1.0e-9 * 
    double GFLOP =  double(51.0f * N); //1.0e-9 *
    double MElements = double(16.0f * N); //1.0e-6 * 
    double avg_time = tot_compute_time / float(nrepeat);

    // Print results (problem size, time and bandwidth in GB/s).
    // N, time cal, time host to device, time device to host, MElements/s, GB/s, GFlop/s
    printf("%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", N, avg_time, tot_h2d_time, tot_d2h_time, MElements/avg_time, Gbytes/avg_time, GFLOP/avg_time);
  }



}
