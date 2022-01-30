#ifndef UTIL_H
#define UTIL_H

#ifdef _OPENMP
#include <omp.h>
#endif


#ifdef _OPENMP
inline int get_tid() { return omp_get_thread_num(); }
#else
inline int get_tid() { return 0; }
#endif

#endif  // UTIL_H
