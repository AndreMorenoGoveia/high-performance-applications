#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main()
{
    int fd[2];
    pid_t pid;

    // Cria o pipe
    // OBS: a posição fd[0] é a extremidade de leitura, enquanto fd[1] é a extremidade de escrita
    if (pipe(fd) == -1) {
        perror("pipe");
        exit(1);
    }

    pid = fork();

    if (pid < 0) {
        perror("fork");
        exit(1);
    }

    // Processo pai
    if (pid > 0) {
        close(fd[0]);  // O processo pai não realiza a leitura
        char mensagem[] = "Hello world!";
        write(fd[1], mensagem, sizeof(mensagem));
        close(fd[1]); // O processo pai fecha a extremidade de escrita
        wait(NULL); // O processo pai espera algum filho finalizar a execução
    }
    // Processo filho
    else {
        close(fd[1]);  // O processo filho não realiza a escrita
        char mensagem[20];
        read(fd[0], mensagem, sizeof(mensagem));
        printf("Filho recebeu: %s\n", mensagem);
        close(fd[0]); // O processo pai fecha a extremidade de leitura
    }

    return 0;
}
