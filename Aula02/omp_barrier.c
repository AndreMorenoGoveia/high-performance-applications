#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        printf("Thead id: %d \n", omp_get_thread_num());
        #pragma omp barrier
        printf("Numero de threads: %d \n", omp_get_num_threads());
    }

    return 0;
}