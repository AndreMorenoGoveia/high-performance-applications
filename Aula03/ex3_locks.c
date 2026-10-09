#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define N 5
#define DURACAO 3.0

omp_lock_t garfos[N];
int refeicoes[N];
double ultima[N];

int refeicoes_de(int id) {
    int r;
    #pragma omp atomic read
    r = refeicoes[id];
    return r;
}

int main() {
    double fim = omp_get_wtime() + DURACAO;

    for (int i = 0; i < N; i++)
        omp_init_lock(&garfos[i]);

    #pragma omp parallel num_threads(N)
    {
        int id = omp_get_thread_num();
        int esq = id;
        int dir = (id + 1) % N;
        int primeiro = esq < dir ? esq : dir;
        int segundo = esq < dir ? dir : esq;
        int viz_esq = (id + N - 1) % N;
        int viz_dir = (id + 1) % N;
        unsigned int semente = id;
        ultima[id] = omp_get_wtime();

        while (omp_get_wtime() < fim) {
            usleep(rand_r(&semente) % 1000);

            while ((refeicoes_de(id) > refeicoes_de(viz_esq) || refeicoes_de(id) > refeicoes_de(viz_dir))
                   && omp_get_wtime() < fim)
                usleep(10);

            omp_set_lock(&garfos[primeiro]);
            usleep(1000);
            omp_set_lock(&garfos[segundo]);

            #pragma omp atomic
            refeicoes[id]++;
            ultima[id] = omp_get_wtime();
            usleep(500);

            omp_unset_lock(&garfos[segundo]);
            omp_unset_lock(&garfos[primeiro]);
        }
    }

    double termino = omp_get_wtime();

    for (int i = 0; i < N; i++) {
        omp_destroy_lock(&garfos[i]);
        printf("Filosofo %d comeu %d vezes e ficou %.3f s sem comer no final\n", i, refeicoes[i], termino - ultima[i]);
    }

    return 0;
}
