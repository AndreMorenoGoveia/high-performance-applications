#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TIMER_START \
    struct timespec t1; \
    clock_gettime(CLOCK_MONOTONIC, &t1)

#define TIMER_STOP \
    struct timespec t2; \
    clock_gettime(CLOCK_MONOTONIC, &t2); \
    double t3 = (t2.tv_sec - t1.tv_sec) + \
                (t2.tv_nsec - t1.tv_nsec) / 1e9; \
    printf("Tempo: %.5f s\n", t3)

#define SIZE 1000

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

void multiplica(int inicio, int fim) {
    for (int i = inicio; i < fim; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {

    int i, j;

    srand(42);

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
            C[i][j] = 0;
        }
    }

    TIMER_START;
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        multiplica(0, SIZE / 4);

        #pragma omp section
        multiplica(SIZE / 4, SIZE / 2);

        #pragma omp section
        multiplica(SIZE / 2, 3 * SIZE / 4);

        #pragma omp section
        multiplica(3 * SIZE / 4, SIZE);
    }
    TIMER_STOP;

    printf("Matriz[%d][%d]: %d\n",0,0,C[0][0]);
    printf("Matriz[%d][%d]: %d\n",1,1,C[1][1]);
    printf("Matriz[%d][%d]: %d\n",SIZE-1,SIZE-1,C[SIZE-1][SIZE-1]);

    return 0;
}