/*
 * simd.c — IVF nearest-neighbour search in one C call per HTTP request.
 *
 * On x86-64 with SSE4.1: uses _mm_madd_epi16 (SSE2) + _mm_hadd_epi32 (SSSE3).
 * Two 128-bit loads cover all vecstride=16 int16 elements (14 real + 2 zero).
 * On other architectures: plain C scalar loop (auto-vectorisable by GCC/Clang).
 *
 * Compiled with -O3 [-msse4.1 on amd64] via CGo CFLAGS in simd_cgo.go.
 */
#include "simd.h"
#include <string.h>
#include <limits.h>

/* ── Squared L2 distance ──────────────────────────────────────────────────── */

#if defined(__SSE4_1__)
#include <smmintrin.h>   /* SSE4.1 implies SSE2, SSE3, SSSE3 */

static inline long long sqDist16(const short *a, const short *b) {
    __m128i d0 = _mm_sub_epi16(_mm_loadu_si128((const __m128i *)a),
                                _mm_loadu_si128((const __m128i *)b));
    __m128i d1 = _mm_sub_epi16(_mm_loadu_si128((const __m128i *)(a + 8)),
                                _mm_loadu_si128((const __m128i *)(b + 8)));
    __m128i sq = _mm_add_epi32(_mm_madd_epi16(d0, d0), _mm_madd_epi16(d1, d1));
    sq = _mm_hadd_epi32(sq, sq);
    sq = _mm_hadd_epi32(sq, sq);
    return (long long)_mm_cvtsi128_si32(sq);
}

#else  /* scalar fallback — auto-vectorised as NEON/AVX by the compiler */

static inline long long sqDist16(const short *a, const short *b) {
    long long s = 0;
    for (int i = 0; i < 14; i++) {
        long long d = (long long)a[i] - b[i];
        s += d * d;
    }
    return s;
}

#endif

/* ── Internal helpers ─────────────────────────────────────────────────────── */

static void scanCluster(const short *vecs, unsigned cnt, int vecstride,
                        const unsigned char *labels,
                        const short *query,
                        long long *topDists, unsigned char *topLabels,
                        long long *maxDist, int *maxPos, int nNeigh)
{
    const short *v = vecs;
    for (unsigned i = 0; i < cnt; i++, v += vecstride) {
        long long d = sqDist16(query, v);
        if (d < *maxDist) {
            topDists[*maxPos]  = d;
            topLabels[*maxPos] = labels[i];
            *maxDist = topDists[0];
            *maxPos  = 0;
            for (int j = 1; j < nNeigh; j++) {
                if (topDists[j] > *maxDist) {
                    *maxDist = topDists[j];
                    *maxPos  = j;
                }
            }
        }
    }
}

static int countFraud(const unsigned char *topLabels, int nNeigh) {
    int cnt = 0;
    for (int i = 0; i < nNeigh; i++) cnt += topLabels[i];
    return cnt;
}

/* ── Public API: one call per HTTP request ────────────────────────────────── */

int scoreRequestSIMD(
    const short    *centroids,
    int             k,
    const short    *vecs,
    const unsigned *offsets,
    const unsigned *counts,
    const unsigned char *labels,
    const short    *query,
    int             vecstride,
    int             nProbe,
    int             nProbeRepair,
    int             nNeigh)
{
    /* 1. Find top-nProbeRepair nearest centroids */
    long long bestDists[128];
    int       bestIdx[128];

    for (int i = 0; i < nProbeRepair; i++) {
        bestDists[i] = sqDist16(query, centroids + (long long)i * vecstride);
        bestIdx[i]   = i;
    }

    long long maxDist = bestDists[0];
    int       maxPos  = 0;
    for (int i = 1; i < nProbeRepair; i++) {
        if (bestDists[i] > maxDist) { maxDist = bestDists[i]; maxPos = i; }
    }

    for (int c = nProbeRepair; c < k; c++) {
        long long d = sqDist16(query, centroids + (long long)c * vecstride);
        if (d < maxDist) {
            bestDists[maxPos] = d;
            bestIdx[maxPos]   = c;
            maxDist = bestDists[0]; maxPos = 0;
            for (int j = 1; j < nProbeRepair; j++) {
                if (bestDists[j] > maxDist) { maxDist = bestDists[j]; maxPos = j; }
            }
        }
    }

    /* Insertion sort (nProbeRepair <= 64) */
    for (int i = 1; i < nProbeRepair; i++) {
        long long kd = bestDists[i];
        int       ki = bestIdx[i];
        int j = i - 1;
        while (j >= 0 && bestDists[j] > kd) {
            bestDists[j + 1] = bestDists[j];
            bestIdx[j + 1]   = bestIdx[j];
            j--;
        }
        bestDists[j + 1] = kd;
        bestIdx[j + 1]   = ki;
    }

    /* 2. Initialise top-nNeigh heap */
    long long     topDists[16];
    unsigned char topLabels[16];
    for (int i = 0; i < nNeigh; i++) {
        topDists[i]  = LLONG_MAX;
        topLabels[i] = 0;
    }
    long long heapMax  = LLONG_MAX;
    int       heapMaxP = 0;

    /* 3. Scan initial nProbe clusters */
    for (int i = 0; i < nProbe; i++) {
        int      c   = bestIdx[i];
        unsigned off = offsets[c];
        unsigned cnt = counts[c];
        scanCluster(vecs + (long long)off * vecstride, cnt, vecstride,
                    labels + off, query,
                    topDists, topLabels, &heapMax, &heapMaxP, nNeigh);
    }

    int fraudCnt = countFraud(topLabels, nNeigh);

    /* 4. Repair path if result is uncertain */
    if (fraudCnt > 0 && fraudCnt < nNeigh) {
        for (int i = nProbe; i < nProbeRepair; i++) {
            int      c   = bestIdx[i];
            unsigned off = offsets[c];
            unsigned cnt = counts[c];
            scanCluster(vecs + (long long)off * vecstride, cnt, vecstride,
                        labels + off, query,
                        topDists, topLabels, &heapMax, &heapMaxP, nNeigh);
        }
        fraudCnt = countFraud(topLabels, nNeigh);
    }

    return fraudCnt;
}
