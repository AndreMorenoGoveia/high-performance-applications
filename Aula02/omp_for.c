#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for (int i = 0; i < 8; i++) {
            printf("Thread %d executou i = %d\n", omp_get_thread_num(), i);
        }
    }

    return 0;
}