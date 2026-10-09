#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    int n;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        n = 100;
        MPI_Send(&n, 1, MPI_INT, 1, 11, MPI_COMM_WORLD);
        printf("Processo 0: enviei %d\n", n);
    }

    else if (rank == 1) {
        MPI_Recv(&n, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &status);

        printf("Processo 1: recebi %d\n", n);

        printf("MPI_SOURCE = %d\n", status.MPI_SOURCE);
        printf("MPI_TAG    = %d\n", status.MPI_TAG);
        printf("MPI_ERROR  = %d\n", status.MPI_ERROR);
    }

    MPI_Finalize();

    return 0;
}
