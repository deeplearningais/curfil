#include "random_features.h"

#include <curand_kernel.h>

namespace curfil {
namespace gpu {

__global__
void setup_kernel(int seed, curandState *state) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

__global__
void generate_uniform_kernel(curandState* state, unsigned int* result) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int count = 0;
    float x;

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    /* Generate pseudo-random uniforms */
    for (int n = 0; n < 10000; n++) {
        x = curand_uniform(&localState);
        /* Check if > .5 */
        if (x > .5) {
            count++;
        }
    }

    /* Copy state back to global memory */
    state[id] = localState;

    /* Store results */
    result[id] += count;
}

}
}
