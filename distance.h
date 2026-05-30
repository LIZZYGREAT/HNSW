#pragma once
#include <cstddef> 

namespace my_hnsw {

inline float InnerProductDistance(const float* vec1, const float* vec2, size_t dim) {
    float dot_product = 0.0f;
    
    for (size_t i = 0; i < dim; ++i) {
        dot_product += vec1[i] * vec2[i];
    }
    
    return 1.0f - dot_product;
}

} // namespace my_hnsw