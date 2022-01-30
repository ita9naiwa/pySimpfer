#ifndef UTIL_H
#define UTIL_H

// We need to get the thread number to figure out which RNG to use,
// but this will fail on OSX etc if we have no openmp enabled compiler.
// Cython won't let me #ifdef this, so I'm doing it here

#ifdef _OPENMP
#include <omp.h>
#endif


#ifdef _OPENMP
inline int get_tid() { return omp_get_thread_num(); }
#else
inline int get_tid() { return 0; }
#endif

#endif  // UTIL_H
