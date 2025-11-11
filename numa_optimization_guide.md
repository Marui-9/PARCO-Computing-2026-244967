# NUMA Optimization Guide for Sparse Matrix-Vector Multiplication

## Your System
- **NUMA nodes**: 1 (single socket)
- **Cores per socket**: 12
- **Threads**: 24 (with hyperthreading)

Since you have a single NUMA node, you won't see NUMA effects. However, here's how to optimize for multi-socket systems:

## 1. Thread Binding (Works on Single Socket Too)

### Environment Variables
```bash
# Keep threads on same socket (best for single NUMA node)
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Or spread across sockets (for multi-socket)
export OMP_PROC_BIND=spread
```

### Test with your program:
```bash
OMP_PLACES=cores OMP_PROC_BIND=close ./test_config 8 2k_0p52.mtx 50
```

## 2. First-Touch Initialization

Modify your code to initialize arrays in parallel with the same thread distribution:

```c
// In main(), after allocating x and y:
#pragma omp parallel for num_threads(thread_count) schedule(static)
for (int i = 0; i < m; i++) {
    y[i] = 0.0f;
}

#pragma omp parallel for num_threads(thread_count) schedule(static)
for (int i = 0; i < n; i++) {
    x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}
```

## 3. NUMA-Aware Allocation (Multi-Socket Only)

For systems with multiple NUMA nodes:

```c
#include <numa.h>

void* numa_alloc_interleaved_subset(size_t size, struct bitmask *nodemask) {
    if (numa_available() < 0) {
        return malloc(size);  // Fallback
    }
    
    // Allocate memory interleaved across all NUMA nodes
    return numa_alloc_interleaved(size);
}
```

Compile with: `gcc -lnuma ...`

## 4. Check NUMA Statistics

```bash
# Run with numactl to see memory allocation
numactl --hardware
numactl --show

# Monitor NUMA stats during execution
numastat -c ./test_config

# Force specific NUMA node (for testing)
numactl --cpunodebind=0 --membind=0 ./test_config 8 2k_0p52.mtx 50
```

## 5. CSR Format NUMA Considerations

### Option A: Distribute Rows Across NUMA Nodes
```c
void mat_vect_numa_aware(int thread_count, csr_matrix *csr_A, int chunk) {
    #pragma omp parallel num_threads(thread_count)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        // Each thread processes its chunk of rows
        int rows_per_thread = m / nthreads;
        int start_row = tid * rows_per_thread;
        int end_row = (tid == nthreads - 1) ? m : start_row + rows_per_thread;
        
        for (int i = start_row; i < end_row; i++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int j = csr_A->row_ptr[i]; j < csr_A->row_ptr[i + 1]; j++) {
                sum += csr_A->values[j] * x[csr_A->col_ind[j]];
            }
            y[i] = sum;
        }
    }
}
```

### Option B: Replicate Read-Only Data (x vector)
For multi-socket systems with large x vectors, replicate x on each NUMA node to reduce remote memory access.

## 6. Performance Monitoring

### Check NUMA-related metrics:
```bash
perf stat -e node-loads,node-load-misses,node-stores,node-store-misses \
    ./test_config 8 2k_0p52.mtx 50
```

### Expected results on single-socket:
- Low node-load-misses (all memory is local)
- No remote NUMA access

### On multi-socket systems:
- High node-load-misses indicates cross-socket traffic
- Optimize by ensuring first-touch locality

## 7. Recommended Settings for Your System

Since you have **1 NUMA node**, focus on:

1. **Thread binding**: Prevent thread migration
   ```bash
   export OMP_PLACES=cores
   export OMP_PROC_BIND=close
   ```

2. **Use physical cores first**: You have 12 cores + hyperthreading
   - For compute-intensive: Use 12 threads (one per core)
   - For memory-bound: Use up to 24 threads
   
3. **Test different thread counts**:
   ```bash
   for t in 1 2 4 6 8 10 12 16 20 24; do
       echo "Threads: $t"
       OMP_PLACES=cores ./test_config $t 5k_9p37.mtx 30
   done
   ```

## Summary

**For your single-socket system:**
- ✅ Use thread affinity (OMP_PLACES=cores)
- ✅ Initialize arrays in parallel (first-touch)
- ✅ Test 12 vs 24 threads (physical cores vs hyperthreading)
- ❌ NUMA interleaving not needed (single node)

**For multi-socket systems:**
- ✅ All of the above
- ✅ Use numa_alloc_interleaved() or first-touch policy
- ✅ Monitor node-loads/node-load-misses with perf
- ✅ Consider replicating read-only data (x vector)
- ✅ Use OMP_PROC_BIND=spread to distribute across sockets
