/**
 * Created by Bryce Melander on 10/21/22.
 * Re-implementation of pthread barriers
 * for use on MacOS
 */

#include <pthread.h>

#ifndef SOBEL_FILTER_PTHREAD_BARRIER_H
#define SOBEL_FILTER_PTHREAD_BARRIER_H

// Check if barriers are already defined
#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD (-1)

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition_variable;
    int threads_required;
    int threads_left;
    unsigned int cycle;
} pthread_barrier_t;

int pthread_barrier_init(pthread_barrier_t *barrier, __attribute__((unused)) void *attr, int count);

int pthread_barrier_wait(pthread_barrier_t *barrier);

int pthread_barrier_destroy(pthread_barrier_t *barrier);

#endif //PTHREAD_BARRIER_SERIAL_THREAD

#endif //SOBEL_FILTER_PTHREAD_BARRIER_H
