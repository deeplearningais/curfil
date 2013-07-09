#ifndef CURFIL_NDARRAY_OPS_H
#define CURFIL_NDARRAY_OPS_H

#include <cuv/ndarray.hpp>

namespace curfil {

template<class value_type>
void add_ndarrays(cuv::ndarray<value_type, cuv::host_memory_space>& a,
        const cuv::ndarray<value_type, cuv::host_memory_space>& b, cuv::host_memory_space);

template<class value_type, class memory_space>
cuv::ndarray<value_type, memory_space>&
operator+=(cuv::ndarray<value_type, memory_space>& a, const cuv::ndarray<value_type, memory_space>& b);

}

#endif
