#include <stdio.h>
#include <cuda_runtime.h>

// Kernel executado na GPU
__global__ void somaVetores(int *a, int *b, int *c, int n)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    int n = 5;
    int size = n * sizeof(int);

    // Dados no HOST (CPU)
    int h_a[] = {1, 2, 3, 4, 5};
    int h_b[] = {10, 20, 30, 40, 50};
    int h_c[5];

    // 1. Aloca memória no DEVICE (GPU)
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 2. Copia os dados do HOST para o DEVICE
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 3. Chama o KERNEL na GPU
    somaVetores<<<1, n>>>(d_a, d_b, d_c, n);

    // Espera o kernel terminar
    cudaDeviceSynchronize();

    // 4. Copia o resultado do DEVICE para o HOST
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Mostra o resultado
    for (int i = 0; i < n; i++)
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);

    // 5. Libera a memória do DEVICE
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
