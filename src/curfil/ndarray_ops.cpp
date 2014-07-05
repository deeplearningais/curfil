#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif
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
