#include <sys/time.h>
#include <sys/resource.h>


/* Ejemplo de uso	
	struct rusage ru_begin;
	struct rusage ru_end;
	struct timeval tv_elapsed;
	getrusage(RUSAGE_SELF, &ru_begin);

	int result = dot(a,b,N);

	getrusage(RUSAGE_SELF, &ru_end);
	timeval_subtract(&tv_elapsed, &ru_end.ru_utime, &ru_begin.ru_utime);
	printf("El cpu tardo %g ms\n", (tv_elapsed.tv_sec + (tv_elapsed.tv_usec/1000000.0))*1000.0);
*/
int timeval_subtract (struct timeval * result, struct timeval *x, struct timeval * y){
	// Perform the carry for the later subtraction by updating y.
	if(x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if(x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	// Compute the time remaining to wait. tv_usec is certainly positive.
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	// Return 1 if result is negative.
	return x->tv_sec < y->tv_sec;
}
