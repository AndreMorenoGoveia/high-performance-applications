#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int id;
    int a;
    int b;
    int resultado;
} Dados;

void *calcular(void *arg)
{
    Dados *dados = (Dados *)arg;
    dados->resultado = dados->a * dados->b;
    return NULL;
}

int main()
{
    pthread_t threads[2];

    Dados dados[2] = {
        {1, 10, 20, 0},
        {2, 30, 40, 0}
    };

    // Cria as duas threads
    pthread_create(&threads[0], NULL, calcular, &dados[0]);
    pthread_create(&threads[1], NULL, calcular, &dados[1]);

    // Espera as duas threads terminarem
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    // Mostra os resultados
    printf("Resultados:\n");
    printf("Thread %d: %d x %d = %d\n", dados[0].id, dados[0].a, dados[0].b, dados[0].resultado);
    printf("Thread %d: %d x %d = %d\n", dados[1].id, dados[1].a, dados[1].b, dados[1].resultado);

    return 0;
}