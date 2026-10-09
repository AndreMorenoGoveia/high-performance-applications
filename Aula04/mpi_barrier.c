#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Processo %d: antes da barreira\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Processo %d: depois da barreira\n", rank);

    MPI_Finalize();

    return 0;
}