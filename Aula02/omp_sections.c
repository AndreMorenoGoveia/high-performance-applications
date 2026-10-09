#include <stdio.h>
#include <omp.h>

int main() {

    omp_set_max_active_levels(2);
    #pragma omp parallel sections num_threads(4)
    {
        #pragma omp section
        {
            #pragma omp parallel num_threads(2)
            {
                printf("Secao 1 - Thread %d\n", omp_get_thread_num());
            }
        }

        #pragma omp section
        {
            #pragma omp parallel num_threads(2)
            {
                printf("Secao 2 - Thread %d\n", omp_get_thread_num());
            }
        }
    }

    return 0;
}
