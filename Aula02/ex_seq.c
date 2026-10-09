// Esse código apresenta uma implementacao sequencial da soma dos elementos de um vetor

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000

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
    int soma = 0;
    int i;

    srand(42);

    for(i = 0; i < SIZE; i++){
        vetor[i] = rand() % 100;
    }
    
    TIMER_START;

    for(i = 0; i < SIZE; i++){
        soma += vetor[i];
    }
    
    TIMER_STOP;

    printf("A soma dos elementos do vetor é: %d\n", soma);
    return 0;
}