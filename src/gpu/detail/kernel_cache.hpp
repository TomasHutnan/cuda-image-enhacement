#pragma once

#include <cassert>
#include <cstring>
#include <iostream>

namespace tgpu::gpu::detail {

/// Generic kernel weight cache for constant memory uploads (e.g., Gaussian, PSF).
/// Caches weights to avoid redundant uploads and recomputation.
/// 
/// Template parameters:
/// - ElementType: Type of cache elements (float, int, etc.)
/// - MaxKernelSize: Maximum kernel weight array size (e.g., 129 for Gaussian)
template <typename ElementType, int MaxKernelSize>
class KernelCache {
public:
    KernelCache() : valid_(false), cached_sigma_(-1.0f), cached_radius_(-1) {}

    /// Check if cache contains valid weights for given parameters.
    /// @param sigma Filter parameter (e.g., Gaussian standard deviation)
    /// @param radius Filter radius (e.g., kernel half-width)
    /// @return true if cached values match; false if recomputation needed
    bool is_valid(float sigma, int radius) const {
        return valid_ && cached_sigma_ == sigma && cached_radius_ == radius;
    }

    /// Update cache with new weights via cudaMemcpyToSymbol.
    /// @param sigma Filter parameter
    /// @param radius Filter radius
    /// @param weights_host Pointer to host-side weight array
    /// @param symbol_name CUDA device symbol name (e.g., "kGaussianWeights")
    /// @param num_elements Number of elements to copy
    void update(float sigma, int radius, const ElementType* weights_host, 
                const void* device_symbol, int num_elements) {
        assert(weights_host != nullptr);
        assert(num_elements > 0 && num_elements <= MaxKernelSize);

        cudaError_t err = cudaMemcpyToSymbol(device_symbol, weights_host, 
                                             num_elements * sizeof(ElementType));
        if (err != cudaSuccess) {
            std::cerr << "KernelCache::update failed: " << cudaGetErrorString(err) << "\n";
            valid_ = false;
            return;
        }

        cached_sigma_ = sigma;
        cached_radius_ = radius;
        valid_ = true;
    }

    /// Validate cache state and log issues.
    /// @param sigma Current filter parameter
    /// @param radius Current filter radius
    /// @param stage_name Name of stage for diagnostic messages
    void check_valid(float sigma, int radius, const char* stage_name) const {
        if (!is_valid(sigma, radius)) {
            std::cerr << stage_name << ": kernel cache miss (sigma=" << sigma 
                      << ", radius=" << radius << ")\n";
        }
    }

    /// Clear cache state (used at pipeline cleanup or explicit resource release).
    void clear() {
        valid_ = false;
        cached_sigma_ = -1.0f;
        cached_radius_ = -1;
    }

private:
    bool valid_;
    float cached_sigma_;
    int cached_radius_;
};

}  // namespace tgpu::gpu::detail
