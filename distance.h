#pragma once
#include <cstddef> 
#include <omp.h>

namespace hnsw {
    inline float get_distance(const float* vec1, const float* vec2, size_t dim) {
        float distance = 0.0f;
        
        #pragma omp simd reduction(+:distance)
        // 循环展开，约归reduction
        for (size_t i = 0; i < dim; ++i) {
            distance += vec1[i] * vec2[i];
        }
        
        return 1.0f - distance;
    }

} // namespace hnsw