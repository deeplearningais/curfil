#include "ndarray_ops.h"

namespace cuv {

template<class V, class M, class L>
void fill(cuv::ndarray<V, M, L>& v, const V& p) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = p;
    }
}

}

namespace curfil
{

template<class value_type>
void add_ndarrays(
        cuv::ndarray<value_type, cuv::host_memory_space>& a,
        const cuv::ndarray<value_type, cuv::host_memory_space>& b,
        cuv::host_memory_space) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("can not add tensors with different shapes");
    }
    for (size_t i = 0; i < a.size(); i++) {
        a[i] += b[i];
    }
}

template<class value_type, class memory_space>
cuv::ndarray<value_type, memory_space>&
operator+=(cuv::ndarray<value_type, memory_space>& a, const cuv::ndarray<value_type, memory_space>& b) {
    add_ndarrays(a, b, memory_space());
    return a;
}

}

#define CUV_NDARRAY_OPS_INST(T, M) \
    template cuv::ndarray<T, M>& curfil::operator+=(cuv::ndarray<T, M>&, const cuv::ndarray<T, M>&); \
    template void cuv::fill(cuv::ndarray<T, M, cuv::row_major>&, const T&); \
    template void curfil::add_ndarrays(cuv::ndarray<T, M>&, const cuv::ndarray<T, M>&, M);

CUV_NDARRAY_OPS_INST(double, cuv::host_memory_space)
;
CUV_NDARRAY_OPS_INST(float, cuv::host_memory_space)
;
CUV_NDARRAY_OPS_INST(int, cuv::host_memory_space)
;
CUV_NDARRAY_OPS_INST(unsigned int, cuv::host_memory_space)
;
CUV_NDARRAY_OPS_INST(unsigned char, cuv::host_memory_space)
;
