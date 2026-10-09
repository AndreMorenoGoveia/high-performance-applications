#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define N 5
#define DURACAO 3.0

volatile int garfos[N];
int refeicoes[N];
double ultima[N];

int main() {
    double fim = omp_get_wtime() + DURACAO;

    #pragma omp parallel num_threads(N)
    {
        int id = omp_get_thread_num();
        int esq = id;
        int dir = (id + 1) % N;
        unsigned int semente = id;
        ultima[id] = omp_get_wtime();

        while (omp_get_wtime() < fim) {
            usleep(rand_r(&semente) % 1000);

            while (garfos[esq] && omp_get_wtime() < fim);
            garfos[esq] = 1;
            usleep(1000);

            while (garfos[dir] && omp_get_wtime() < fim);
            if (omp_get_wtime() >= fim) break;
            garfos[dir] = 1;

            refeicoes[id]++;
            ultima[id] = omp_get_wtime();
            usleep(500);

            garfos[dir] = 0;
            garfos[esq] = 0;
        }
    }

    double termino = omp_get_wtime();

    for (int i = 0; i < N; i++)
        printf("Filosofo %d comeu %d vezes e ficou %.3f s sem comer no final\n", i, refeicoes[i], termino - ultima[i]);

    return 0;
}
