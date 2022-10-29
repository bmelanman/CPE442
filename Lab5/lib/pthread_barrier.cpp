//
// Created by Bryce Melander on 10/21/22.
//

#include "pthread_barrier.h"

int pthread_barrier_init(pthread_barrier_t *barrier, __attribute__((unused)) void *attr, int count) {
    barrier->threads_required = count;
    barrier->threads_left = count;
    barrier->cycle = 0;
    pthread_mutex_init(&barrier->mutex, nullptr);
    pthread_cond_init(&barrier->condition_variable, nullptr);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);

    if (--barrier->threads_left == 0) {
        barrier->cycle++;
        barrier->threads_left = barrier->threads_required;

        pthread_cond_broadcast(&barrier->condition_variable);
        pthread_mutex_unlock(&barrier->mutex);

        return PTHREAD_BARRIER_SERIAL_THREAD;
    } else {
        unsigned int cycle = barrier->cycle;

        while (cycle == barrier->cycle)
            pthread_cond_wait(&barrier->condition_variable, &barrier->mutex);

        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}

int pthread_barrier_destroy(pthread_barrier_t *barrier) {
    pthread_cond_destroy(&barrier->condition_variable);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}
