#include <stdio.h>
#include <omp.h>

int main()
{
    omp_lock_t lock;
    int contador = 0;

    omp_init_lock(&lock);

    #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 1000; i++) {
            omp_set_lock(&lock);
            contador++;
            omp_unset_lock(&lock);
        }

    omp_destroy_lock(&lock);

    printf("contador = %d\n", contador);

    return 0;
}