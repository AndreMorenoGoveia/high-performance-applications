#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    int vetor[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int elementos[2];
    int soma_parcial;
    int somas_parciais[4];
    int soma_total;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verifica se np = 4
    if (size != 4) {
        MPI_Finalize();
        return 0;
    }

    // Distribui 2 elementos para cada processo
    MPI_Scatter(vetor, 2, MPI_INT, elementos, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Cada processo calcula sua soma parcial
    soma_parcial = elementos[0] + elementos[1];

    printf("Processo %d: %d + %d = %d\n", rank, elementos[0], elementos[1], soma_parcial);

    // Reúne as somas parciais no processo 0
    MPI_Gather(&soma_parcial, 1, MPI_INT, somas_parciais, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        soma_total = 0;

        for (int i = 0; i < 4; i++) {
            soma_total += somas_parciais[i];
        }

        printf("\nSoma total = %d\n", soma_total);
    }

    MPI_Finalize();

    return 0;
}