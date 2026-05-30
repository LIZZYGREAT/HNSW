#pragma once
#include <cstddef> 

namespace hnsw {
    //inline 内联，消除简单的distance平凡调用开销
    inline float get_distance(const float* vec1, const float* vec2, size_t dim) {
        float distance = 0.0f;
        
        for (size_t i = 0; i < dim; ++i) {
            distance += vec1[i] * vec2[i];
        }
        
        return 1.0f - distance;
    }

} // namespace hnsw