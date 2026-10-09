#include <stdio.h>
#include <unistd.h>         // fork() e getpid()
#include <sys/types.h>      // Tipo pid_t

int main()
{
    pid_t pid;

    pid = fork();

    // A chamada fork() retorna -1 quando ocorre algum erro
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    printf("PID: %d | retorno de fork(): %d\n", getpid(), pid);

    return 0;
}