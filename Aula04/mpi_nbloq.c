#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    int n;

    MPI_Request request;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        n = 100;
        MPI_Isend(&n, 1, MPI_INT, 1, 11, MPI_COMM_WORLD, &request);
        printf("Processo 0: enviei %d\n", n);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    else if (rank == 1) {
        MPI_Irecv(&n, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        printf("Processo 1: recebi %d\n", n);
    }

    MPI_Finalize();

    return 0;
}