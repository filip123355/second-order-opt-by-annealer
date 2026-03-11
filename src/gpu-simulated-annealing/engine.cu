#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Simple GPU simulated annealer for dense Ising / BQM problems.
// - Spins are represented as {-1, +1}
// - J is a dense symmetric matrix stored row-major (size N*N)
// - h is local fields vector (size N)
// - Runs R replicas in parallel, each performing `steps` sweeps

__device__ inline float replica_energy(const int8_t* spins, const float* J, const float* h, int N) {
    float E = 0.0f;
    for (int i = 0; i < N; ++i) {
        float local = h[i] * spins[i];
        // dot product J[i,:] * spins
        float ssum = 0.0f;
        int idx = i * N;
        for (int j = 0; j < N; ++j) {
            ssum += J[idx + j] * spins[j];
        }
        // each pair counted twice in full sum; we include full interaction then divide later
        E += 0.5f * spins[i] * ssum + local;
    }
    return E;
}

__global__ void sa_kernel(
    int8_t* d_spins, // R x N packed as replica-major contiguous (replica* N + i)
    const float* J,   // N x N
    const float* h,   // N
    float beta_start,
    float beta_end,
    int N,
    int steps,
    int R,
    unsigned long long seed,
    float* out_best_energy,
    int8_t* out_best_spins
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= R) return;

    // per-replica RNG
    curandState_t state;
    curand_init(seed ^ (unsigned long long)rid, 0, 0, &state);

    int base = rid * N;
    int8_t* spins = d_spins + base;

    // compute initial energy
    float bestE = replica_energy(spins, J, h, N);
    // copy current spins to best
    for (int i = 0; i < N; ++i) out_best_spins[base + i] = spins[i];

    for (int t = 0; t < steps; ++t) {
        float frac = (steps == 1) ? 1.0f : (float)t / (float)(steps - 1);
        float beta = beta_start + frac * (beta_end - beta_start);

        // single sweep: try flipping each spin sequentially
        for (int i = 0; i < N; ++i) {
            // compute local field sum: h[i] + sum_j J[i,j] s_j
            float local = h[i];
            int idx = i * N;
            for (int j = 0; j < N; ++j) {
                local += J[idx + j] * (float)spins[j];
            }
            // energy change for flipping s_i -> -s_i is dE = 2 * s_i * local
            float si = (float)spins[i];
            float dE = 2.0f * si * local;

            bool accept = false;
            if (dE <= 0.0f) {
                accept = true;
            } else {
                float u = curand_uniform(&state);
                if (u < expf(-beta * dE)) accept = true;
            }

            if (accept) {
                spins[i] = (int8_t)(-spins[i]);
            }
        }

        // optionally evaluate energy and track best
        float E = replica_energy(spins, J, h, N);
        if (E < bestE) {
            bestE = E;
            for (int i = 0; i < N; ++i) out_best_spins[base + i] = spins[i];
        }
    }

    out_best_energy[rid] = bestE;
}

// Host-side helper: simple example that sets up a random small Ising and runs the kernel.
int main(int argc, char** argv) {
    // small demo parameters
    int N = 64;               // number of spins
    int R = 256;              // replicas
    int steps = 200;          // sweeps per replica
    float beta_start = 0.1f;
    float beta_end = 5.0f;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) R = atoi(argv[2]);

    size_t Jbytes = sizeof(float) * N * N;
    size_t hbytes = sizeof(float) * N;
    size_t spins_bytes = sizeof(int8_t) * N * R;
    size_t energy_bytes = sizeof(float) * R;

    // allocate host
    float* h_J = (float*)malloc(Jbytes);
    float* h_h = (float*)malloc(hbytes);
    int8_t* h_spins = (int8_t*)malloc(spins_bytes);
    float* h_bestE = (float*)malloc(energy_bytes);
    int8_t* h_bestSpins = (int8_t*)malloc(spins_bytes);

    // random init (small couplings)
    srand(123);
    for (int i = 0; i < N; ++i) {
        h_h[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
        for (int j = 0; j < N; ++j) {
            if (i == j) h_J[i * N + j] = 0.0f;
            else {
                float v = 0.02f * ((float)rand() / RAND_MAX - 0.5f);
                h_J[i * N + j] = v;
            }
        }
    }

    // initial spins per replica
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < N; ++i) {
            h_spins[r * N + i] = (rand() & 1) ? 1 : -1;
        }
    }

    // device buffers
    float* d_J; float* d_h; int8_t* d_spins; float* d_bestE; int8_t* d_bestSpins;
    cudaMalloc((void**)&d_J, Jbytes);
    cudaMalloc((void**)&d_h, hbytes);
    cudaMalloc((void**)&d_spins, spins_bytes);
    cudaMalloc((void**)&d_bestE, energy_bytes);
    cudaMalloc((void**)&d_bestSpins, spins_bytes);

    cudaMemcpy(d_J, h_J, Jbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h_h, hbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spins, h_spins, spins_bytes, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (R + threads - 1) / threads;

    unsigned long long seed = 123456789ULL;

    sa_kernel<<<blocks, threads>>>(d_spins, d_J, d_h, beta_start, beta_end, N, steps, R, seed, d_bestE, d_bestSpins);
    cudaDeviceSynchronize();

    cudaMemcpy(h_bestE, d_bestE, energy_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bestSpins, d_bestSpins, spins_bytes, cudaMemcpyDeviceToHost);

    // find global best across replicas
    float globalE = 1e30f;
    int best_r = -1;
    for (int r = 0; r < R; ++r) {
        if (h_bestE[r] < globalE) {
            globalE = h_bestE[r];
            best_r = r;
        }
    }
    printf("Global best energy: %f (replica %d)\n", globalE, best_r);

    // cleanup
    cudaFree(d_J); cudaFree(d_h); cudaFree(d_spins); cudaFree(d_bestE); cudaFree(d_bestSpins);
    free(h_J); free(h_h); free(h_spins); free(h_bestE); free(h_bestSpins);
    return 0;
}
