// Soma paralela: cada filho soma um trecho e envia o resultado ao pai por um pipe.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#ifndef SIZE
#define SIZE 1000
#endif
#ifndef NUM_PROCESSES
#define NUM_PROCESSES 8
#endif


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

static int ler_soma(int fd, long long *soma){
    size_t recebidos = 0;
    while(recebidos < sizeof(*soma)){
        ssize_t n = read(fd, (char *)soma + recebidos,
                         sizeof(*soma) - recebidos);
        if(n < 0){
            if(errno == EINTR) continue;
            perror("read");
            return -1;
        }
        if(n == 0){
            fprintf(stderr, "Pipe fechado antes de receber todas as somas.\n");
            return -1;
        }
        recebidos += (size_t)n;
    }
    return 0;
}

int main(void){
    long long soma = 0;
    int i;
    int fd[2];
    int criados = 0;
    int erro = 0;
    const int processos = NUM_PROCESSES < SIZE ? NUM_PROCESSES : SIZE;

    srand(42);

    for(i = 0; i < SIZE; i++){
        vetor[i] = rand() % 100;
    }
    
    TIMER_START;

    if(pipe(fd) == -1){
        perror("pipe");
        return EXIT_FAILURE;
    }

    for(int p = 0; p < processos; p++){
        pid_t pid = fork();
        if(pid < 0){
            perror("fork");
            erro = 1;
            break;
        }

        if(pid == 0){
            close(fd[0]);
            int start = (int)((long long)p * SIZE / processos);
            int end = (int)(((long long)p + 1) * SIZE / processos);
            long long parcial = 0;
            for(i = start; i < end; i++){
                parcial += vetor[i];
            }

            ssize_t enviados;
            do {
                enviados = write(fd[1], &parcial, sizeof(parcial));
            } while(enviados < 0 && errno == EINTR);
            if(enviados != (ssize_t)sizeof(parcial)){
                if(enviados < 0) perror("write");
                else fprintf(stderr, "Escrita incompleta no pipe.\n");
                close(fd[1]);
                _exit(EXIT_FAILURE);
            }
            close(fd[1]);
            _exit(EXIT_SUCCESS);
        }
        criados++;
    }

    close(fd[1]);
    for(i = 0; i < criados; i++){
        long long parcial;
        if(ler_soma(fd[0], &parcial) != 0){
            erro = 1;
            break;
        }
        soma += parcial;
    }
    close(fd[0]);

    for(i = 0; i < criados; i++){
        int status;
        pid_t terminado;
        do {
            terminado = waitpid(-1, &status, 0);
        } while(terminado < 0 && errno == EINTR);
        if(terminado < 0){
            perror("waitpid");
            erro = 1;
            break;
        }
        if(!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS){
            fprintf(stderr, "Um processo filho falhou.\n");
            erro = 1;
        }
    }
    if(erro){
        return EXIT_FAILURE;
    }
    
    TIMER_STOP;

    printf("A soma dos elementos do vetor é: %lld\n", soma);
    return 0;
}
