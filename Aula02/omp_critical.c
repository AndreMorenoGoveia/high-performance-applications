#include <stdio.h>
#include <omp.h>

int main()
{
    int contador = 0;
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        #pragma omp critical
        {
            contador = contador + 1;
            printf("Thead %d | contador = %d \n", omp_get_thread_num(), contador);
        }
    }

    printf("Valor final do contador: %d \n", contador);
    return 0;
}