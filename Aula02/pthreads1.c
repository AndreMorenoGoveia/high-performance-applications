#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *funcao(void *arg)
{
    // Transformando os argumentos recebidos em inteiro
    int id = *(int *)arg;

    // OBS: A funcao pthread_seft() retorna um numero equivalente a um "id"
    // da thread. O problema eh que o numero retornado condiz a um mapeamento
    // em memoria da thread, e nao a um identificador simples e sequencial
    pthread_t thread_id = pthread_self();
    printf("Thread %d: executando = %lu\n", id, (unsigned long)thread_id);
    return NULL;
}

int main()
{
    pthread_t threads[2];
    int ids[2] = {1, 2};

    // Cria as duas threads
    pthread_create(&threads[0], NULL, funcao, &ids[0]);
    pthread_create(&threads[1], NULL, funcao, &ids[1]);

    // Espera as duas threads terminarem
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    printf("Fim da execucao \n");

    return 0;
}