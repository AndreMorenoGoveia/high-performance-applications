// Esse código apresenta uma implementacao sequencial da soma dos elementos de um vetor

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define SIZE 1000
#define NUM_THREADS 4
#define SIZE_PER_THREAD (SIZE / NUM_THREADS)

#define TIMER_START \
    struct timespec t1; \
    clock_gettime(CLOCK_MONOTONIC, &t1)

#define TIMER_STOP \
    struct timespec t2; \
    clock_gettime(CLOCK_MONOTONIC, &t2); \
    double t3 = (t2.tv_sec - t1.tv_sec) + \
                (t2.tv_nsec - t1.tv_nsec) / 1e9; \
    printf("Tempo: %.5f s\n", t3)


typedef struct {
    int id;
    int* vetor;
    int* vetor_soma;
} Dados;


void *calcular(void *arg)
{
    Dados *dados = (Dados *)arg;
    int id = dados->id;
    int start = id * SIZE_PER_THREAD;
    int end = (id + 1) * SIZE_PER_THREAD;
    int parcial = 0;

    for(int i = start; i < end; i++){
        parcial += dados->vetor[i];
    }
    dados->vetor_soma[id] = parcial;

    return NULL;
}

int vetor[SIZE];

int main(){
    int soma = 0;
    int i;
    int vetor_soma[NUM_THREADS] = {0};
    pid_t pid;
    int ids[4] = {0, 1, 2, 3};
    pthread_t threads[NUM_THREADS];
    Dados entradas[NUM_THREADS];


    srand(42);

    for(i = 0; i < SIZE; i++){
        vetor[i] = rand() % 100;
    }
    
    TIMER_START;


    entradas[0].id = ids[0];
    entradas[0].vetor = vetor;
    entradas[0].vetor_soma = vetor_soma;
    pthread_create(&threads[0], NULL, calcular, &entradas[0]);

    entradas[1].id = ids[1];
    entradas[1].vetor = vetor;
    entradas[1].vetor_soma = vetor_soma;
    pthread_create(&threads[1], NULL, calcular, &entradas[1]);

    entradas[2].id = ids[2];
    entradas[2].vetor = vetor;
    entradas[2].vetor_soma = vetor_soma;
    pthread_create(&threads[2], NULL, calcular, &entradas[2]);

    entradas[3].id = ids[3];
    entradas[3].vetor = vetor;
    entradas[3].vetor_soma = vetor_soma;
    pthread_create(&threads[3], NULL, calcular, &entradas[3]);

    for(i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
        soma += vetor_soma[i];
    }
    
    TIMER_STOP;

    printf("A soma dos elementos do vetor é: %d\n", soma);
    return 0;
}