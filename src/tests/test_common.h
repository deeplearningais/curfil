#ifndef CURFIL_TEST_COMMON_H
#define CURFIL_TEST_COMMON_H

#include <cuv/ndarray.hpp>

template<class W>
std::vector<const W*> getPointers(const std::vector<W>& v) {
    std::vector<const W*> r(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        r[i] = &v[i];
    }
    return r;
}

template<class V>
bool operator==(const cuv::ndarray<V, cuv::host_memory_space>& a,
        const cuv::ndarray<V, cuv::host_memory_space>& b) {

    if (a.ndim() != b.ndim()) {
        return false;
    }

    if (a.shape() != b.shape()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); i++) {
        if (a.ptr()[i] != b.ptr()[i]) {
            return false;
        }
    }

    return true;
}

#endif
