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
