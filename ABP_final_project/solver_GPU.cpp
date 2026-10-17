#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    {
        Kokkos::parallel_for(
            "Hello",
            10,
            KOKKOS_LAMBDA(const int i)
            {
                printf("Hello from %d\n", i);
            });
    }

    Kokkos::finalize();

    return 0;
}