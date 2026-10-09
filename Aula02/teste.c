#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TIMER_START \
    struct timespec t1; \
    clock_gettime(CLOCK_MONOTONIC, &t1)

#define TIMER_STOP \
    struct timespec t2; \
    clock_gettime(CLOCK_MONOTONIC, &t2); \
    double t3 = (t2.tv_sec - t1.tv_sec) + \
                (t2.tv_nsec - t1.tv_nsec) / 1e9; \
    printf("Tempo: %.5f s\n", t3)

#define SIZE 1000

int main(){
	
	int a;
  
  TIMER_START;
	for(int i = 0; i < 1000; i++)
		a = a + 1;
  TIMER_STOP;
  
	return 0;
}