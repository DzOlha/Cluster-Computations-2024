#include <mpi.h>
#include <stdio.h>

int main(int args, char** argv) {
        MPI_Init(&args, &argv);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size); // get the whole processes size

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // get the number of the current process

        printf("Hello from the process of number %d out of %d processes.\n", world_rank, world_size);

        MPI_Finalize();

        return 0;
}