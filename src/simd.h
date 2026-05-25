#pragma once
#include <stdint.h>

/*
 * scoreRequestSIMD — complete IVF nearest-neighbour search in a single CGo
 * transition (1 call per HTTP request).
 *
 * Steps performed entirely in C:
 *   1. SIMD scan of k centroids → nProbeRepair nearest indices (sorted)
 *   2. Scan initial nProbe clusters → top-nNeigh heap
 *   3. If result is uncertain (0 < fraudCount < nNeigh), scan remaining
 *      nProbeRepair-nProbe clusters for the repair path.
 *   4. Return fraud count (# of top-nNeigh neighbours whose label==1).
 *
 * Parameters
 * ----------
 * centroids    : k * vecstride int16, last (vecstride-dims) elements zero
 * k            : number of IVF clusters
 * vecs         : n * vecstride int16 in cluster-sorted order
 * offsets      : k uint32s — start index of each cluster in vecs[] / labels[]
 * counts       : k uint32s — number of vectors per cluster
 * labels       : n uint8s (0=legit, 1=fraud)
 * query        : vecstride int16 (last elements zero-padded to vecstride)
 * vecstride    : int16 elements per vector (must be a multiple of 8)
 * nProbe       : initial clusters to scan
 * nProbeRepair : total clusters to scan on repair path (>= nProbe)
 * nNeigh       : K for KNN (size of the top-K heap)
 */
int scoreRequestSIMD(
    const short   *centroids,
    int            k,
    const short   *vecs,
    const unsigned *offsets,
    const unsigned *counts,
    const unsigned char *labels,
    const short   *query,
    int            vecstride,
    int            nProbe,
    int            nProbeRepair,
    int            nNeigh);
