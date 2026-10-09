#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 100000000

#define TIMER_START \
    struct timespec t1; \
    clock_gettime(CLOCK_MONOTONIC, &t1)

#define TIMER_STOP \
    struct timespec t2; \
    clock_gettime(CLOCK_MONOTONIC, &t2); \
    double t3 = (t2.tv_sec - t1.tv_sec) + \
                (t2.tv_nsec - t1.tv_nsec) / 1e9; \
    printf("Tempo: %.5f s\n", t3)

int vetor[SIZE];

int main(){
    long long soma = 0;
    int maior, menor, pares = 0;
    int i;

    srand(42);

    for(i = 0; i < SIZE; i++){
        vetor[i] = rand() % 1000000 + 1;
    }

    TIMER_START;

    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            long long s = 0;
            for(int j = 0; j < SIZE; j++)
                s += vetor[j];
            soma = s;
        }

        #pragma omp section
        {
            int m = vetor[0];
            for(int j = 0; j < SIZE; j++)
                if(vetor[j] > m) m = vetor[j];
            maior = m;
        }

        #pragma omp section
        {
            int m = vetor[0];
            for(int j = 0; j < SIZE; j++)
                if(vetor[j] < m) m = vetor[j];
            menor = m;
        }

        #pragma omp section
        {
            int p = 0;
            for(int j = 0; j < SIZE; j++)
                if(vetor[j] % 2 == 0) p++;
            pares = p;
        }
    }

    TIMER_STOP;

    printf("Soma: %lld\n", soma);
    printf("Maior: %d\n", maior);
    printf("Menor: %d\n", menor);
    printf("Pares: %d\n", pares);
    return 0;
}
