#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Simple GPU simulated annealer for dense Ising / BQM problems.
// - Spins are represented as {-1, +1}
// - J is a dense symmetric matrix stored row-major (size N*N)
// - h is local fields vector (size N)
// - Runs R replicas in parallel, each performing `steps` sweeps

__device__ inline float replica_energy(const int8_t* spins, const float* J, const float* h, int N) {
    float E = 0.0f;
    for (int i = 0; i < N; ++i) {
        float local = h[i] * spins[i];
        float ssum = 0.0f;
        int idx = i * N;
        for (int j = 0; j < N; ++j) {
            ssum += J[idx + j] * spins[j];
        }
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

    // replica's rng
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

static int read_file_bytes(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open input file: %s\n", path);
        return 0;
    }
    size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes) {
        fprintf(stderr, "Failed to read %zu bytes from %s (read %zu)\n", bytes, path, got);
        return 0;
    }
    return 1;
}

static int write_file_bytes(const char* path, const void* src, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open output file: %s\n", path);
        return 0;
    }
    size_t wrote = fwrite(src, 1, bytes, f);
    fclose(f);
    if (wrote != bytes) {
        fprintf(stderr, "Failed to write %zu bytes to %s (wrote %zu)\n", bytes, path, wrote);
        return 0;
    }
    return 1;
}

static void make_path_in_dir(const char* src_file, const char* out_name, char* out_path, size_t out_path_size) {
    const char* slash = strrchr(src_file, '/');
    if (!slash) {
        snprintf(out_path, out_path_size, "%s", out_name);
        return;
    }
    size_t dir_len = (size_t)(slash - src_file + 1);
    if (dir_len + strlen(out_name) + 1 > out_path_size) {
        fprintf(stderr, "Output path too long\n");
        exit(1);
    }
    memcpy(out_path, src_file, dir_len);
    memcpy(out_path + dir_len, out_name, strlen(out_name) + 1);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s N j.bin h.bin num_reads [steps beta_start beta_end]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    const char* j_path = argv[2];
    const char* h_path = argv[3];
    int R = atoi(argv[4]);

    int steps = (argc >= 6) ? atoi(argv[5]) : 200;
    float beta_start = (argc >= 7) ? (float)atof(argv[6]) : 0.1f;
    float beta_end = (argc >= 8) ? (float)atof(argv[7]) : 5.0f;

    if (N <= 0 || R <= 0 || steps <= 0) {
        fprintf(stderr, "N, num_reads and steps must be positive\n");
        return 1;
    }

    size_t Jbytes = sizeof(float) * (size_t)N * (size_t)N;
    size_t hbytes = sizeof(float) * (size_t)N;
    size_t spins_bytes = sizeof(int8_t) * (size_t)N * (size_t)R;
    size_t energy_bytes = sizeof(float) * (size_t)R;

    float* h_J = (float*)malloc(Jbytes);
    float* h_h = (float*)malloc(hbytes);
    int8_t* h_spins = (int8_t*)malloc(spins_bytes);
    float* h_bestE = (float*)malloc(energy_bytes);
    int8_t* h_bestSpins = (int8_t*)malloc(spins_bytes);

    if (!h_J || !h_h || !h_spins || !h_bestE || !h_bestSpins) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    if (!read_file_bytes(j_path, h_J, Jbytes) || !read_file_bytes(h_path, h_h, hbytes)) {
        return 1;
    }

    srand(123);
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < N; ++i) {
            h_spins[r * N + i] = (rand() & 1) ? 1 : -1;
        }
    }

    float* d_J;
    float* d_h;
    int8_t* d_spins;
    float* d_bestE;
    int8_t* d_bestSpins;
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

    sa_kernel<<<blocks, threads>>>(
        d_spins,
        d_J,
        d_h,
        beta_start,
        beta_end,
        N,
        steps,
        R,
        seed,
        d_bestE,
        d_bestSpins
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_bestE, d_bestE, energy_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bestSpins, d_bestSpins, spins_bytes, cudaMemcpyDeviceToHost);

    char best_e_path[4096];
    char best_sample_path[4096];
    make_path_in_dir(j_path, "bestE.bin", best_e_path, sizeof(best_e_path));
    make_path_in_dir(j_path, "bestSample.bin", best_sample_path, sizeof(best_sample_path));

    if (!write_file_bytes(best_e_path, h_bestE, energy_bytes) || !write_file_bytes(best_sample_path, h_bestSpins, spins_bytes)) {
        return 1;
    }

    cudaFree(d_J);
    cudaFree(d_h);
    cudaFree(d_spins);
    cudaFree(d_bestE);
    cudaFree(d_bestSpins);
    free(h_J);
    free(h_h);
    free(h_spins);
    free(h_bestE);
    free(h_bestSpins);
    return 0;
}
