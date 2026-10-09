#include <stdio.h>
#include <mpi.h>

#define BUFSIZE	64

int	buf[BUFSIZE];

int main(int argc, char *argv[])
{
	int	size, rank;
	MPI_Status status;
	int i;

	// Inicializa MPI
	MPI_Init(&argc, &argv);

	// A variavel MPI_COMM_WORLD eh uma variavel interna do MPI
	// A variavel size eh o argumento passado no momento de executar o programa
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Neste exemplo o numero de processos deve ser 2
	if (size != 2){
		MPI_Finalize();
		return 1;
	}

	// Atribui o um rank para cada processo comecando em 0 e terminando em size-1
	// Aqui o rank 0 ira enviar e o rank 1 ira receber a mensagem
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Cada processo imprime o seu rank
	printf("Processo com MPI rank %d \n",rank);

	// Rank 0 envia mensagem para rank 1
	if (rank == 0) {
    	for (i = 0; i < 64; i++)
			buf[i] = i;	
		MPI_Send(buf, BUFSIZE, MPI_INT, 1, 11, MPI_COMM_WORLD);
	}

	// Rank 1 recebe mensagem de rank 0
	else{
		MPI_Recv(buf, BUFSIZE, MPI_INT, 0, 11, MPI_COMM_WORLD,&status);
		for(i = 0; i < 64; i++) 
			printf("%d  ",buf[i]);
		printf("\n");
	}

	MPI_Finalize();
	return 0;
}