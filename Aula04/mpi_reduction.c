#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    int vetor[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int elementos[2];
    int soma_parcial;
    int soma_total;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        MPI_Finalize();
        return 0;
    }

    // Distribui 2 elementos para cada processo
    MPI_Scatter(vetor, 2, MPI_INT, elementos, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada processo calcula sua soma parcial
    soma_parcial = elementos[0] + elementos[1];

    printf("Processo %d: %d + %d = %d\n", rank, elementos[0], elementos[1], soma_parcial);

    // Soma todas as somas parciais no processo 0 por meio do Reduction
    MPI_Reduce(&soma_parcial, &soma_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nSoma total = %d\n", soma_total);
    }

    MPI_Finalize();

    return 0;
}
