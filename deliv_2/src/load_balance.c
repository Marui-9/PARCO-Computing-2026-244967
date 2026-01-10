/* File:     
 *     load_balance.c 
 *
 * Purpose:  
 *     Load balancing strategies for distributed sparse matrix-vector multiplication
 *     Implements NNZ-based distribution to balance computational workload across MPI ranks
 *
 * Author: PARCO Computing Project
 * Date: January 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "load_balance.h"

/*------------------------------------------------------------------
 * Function:    calculate_nnz_based_distribution
 * Purpose:     Compute load-balanced row distribution based on NNZ per row
 * 
 * In args:     row_ptr       - CSR row pointer array (size: rows+1)
 *              rows          - Total number of matrix rows
 *              num_ranks     - Number of MPI ranks to distribute across
 *              rank          - Current MPI rank
 * 
 * Out args:    row_start     - Starting row index for this rank
 *              row_end       - Ending row index for this rank (exclusive)
 * 
 * Strategy:    
 *     1. Calculate total NNZ across all rows
 *     2. Determine target NNZ per rank (total_nnz / num_ranks)
 *     3. Assign contiguous row blocks to each rank targeting equal NNZ
 *     4. Handle remainder NNZ by adjusting last rank's range
 * 
 * Notes:       
 *     - Maintains row contiguity for cache efficiency
 *     - Balances computational work (NNZ-proportional)
 *     - Does not require global matrix data (works on row_ptr only)
 */
void calculate_nnz_based_distribution(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank,
    int *row_start,
    int *row_end
) {
    long long total_nnz = row_ptr[rows];  // Total non-zeros in matrix
    long long target_nnz_per_rank = total_nnz / num_ranks;
    
    if (rank == 0) {
        printf("  Load Balancing: NNZ-based distribution\n");
        printf("    Total NNZ: %lld, Target NNZ/rank: %lld\n", total_nnz, target_nnz_per_rank);
    }
    
    /* Calculate row ranges for all ranks */
    int *all_row_starts = (int *)malloc((num_ranks + 1) * sizeof(int));
    all_row_starts[0] = 0;
    
    long long accumulated_nnz = 0;
    int current_rank = 0;
    
    for (int r = 0; r < rows; r++) {
        long long row_nnz = row_ptr[r + 1] - row_ptr[r];
        accumulated_nnz += row_nnz;
        
        /* If accumulated NNZ exceeds target, assign this row to next rank */
        if (accumulated_nnz >= target_nnz_per_rank * (current_rank + 1) && 
            current_rank < num_ranks - 1) {
            current_rank++;
            all_row_starts[current_rank] = r + 1;
        }
    }
    
    /* Last rank gets all remaining rows */
    all_row_starts[num_ranks] = rows;
    
    /* Extract this rank's range */
    *row_start = all_row_starts[rank];
    *row_end = all_row_starts[rank + 1];
    
    /* Verify load balance (rank 0 prints distribution) */
    if (rank == 0) {
        printf("    Distribution:\n");
        for (int r = 0; r < num_ranks; r++) {
            int start = all_row_starts[r];
            int end = all_row_starts[r + 1];
            long long rank_nnz = row_ptr[end] - row_ptr[start];
            double load_pct = 100.0 * rank_nnz / total_nnz;
            
            printf("      Rank %d: rows [%d-%d) = %d rows, %lld NNZ (%.2f%% of total)\n",
                   r, start, end, end - start, rank_nnz, load_pct);
        }
    }
    
    free(all_row_starts);
}

/*------------------------------------------------------------------
 * Function:    calculate_hybrid_distribution
 * Purpose:     Hybrid strategy combining NNZ balance with row count balance
 * 
 * Strategy:    
 *     1. Calculate pure NNZ-based distribution (as above)
 *     2. Calculate pure row-based distribution (rows / num_ranks)
 *     3. Blend the two with a tunable alpha parameter:
 *        hybrid_target = alpha * nnz_target + (1-alpha) * row_target
 *     4. alpha=1.0: pure NNZ balance (best for irregular matrices)
 *        alpha=0.0: pure row balance (best for uniform matrices)
 *        alpha=0.5: balanced approach
 * 
 * Benefits:    
 *     - Prevents extreme row count imbalance
 *     - Maintains reasonable cache locality
 *     - Adapts to matrix characteristics
 */
void calculate_hybrid_distribution(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank,
    int *row_start,
    int *row_end,
    double alpha  /* 0.0 = row-based, 1.0 = NNZ-based */
) {
    long long total_nnz = row_ptr[rows];
    long long target_nnz_per_rank = total_nnz / num_ranks;
    int target_rows_per_rank = rows / num_ranks;
    
    if (rank == 0) {
        printf("  Load Balancing: Hybrid distribution (alpha=%.2f)\n", alpha);
        printf("    NNZ target: %lld, Row target: %d\n", target_nnz_per_rank, target_rows_per_rank);
    }
    
    int *all_row_starts = (int *)malloc((num_ranks + 1) * sizeof(int));
    all_row_starts[0] = 0;
    
    long long accumulated_nnz = 0;
    int accumulated_rows = 0;
    int current_rank = 0;
    
    for (int r = 0; r < rows; r++) {
        long long row_nnz = row_ptr[r + 1] - row_ptr[r];
        accumulated_nnz += row_nnz;
        accumulated_rows++;
        
        /* Hybrid threshold: weighted combination of NNZ and row targets */
        double nnz_ratio = (double)accumulated_nnz / target_nnz_per_rank;
        double row_ratio = (double)accumulated_rows / target_rows_per_rank;
        double hybrid_ratio = alpha * nnz_ratio + (1.0 - alpha) * row_ratio;
        
        if (hybrid_ratio >= (current_rank + 1) && current_rank < num_ranks - 1) {
            current_rank++;
            all_row_starts[current_rank] = r + 1;
            accumulated_nnz = 0;
            accumulated_rows = 0;
        }
    }
    
    all_row_starts[num_ranks] = rows;
    
    *row_start = all_row_starts[rank];
    *row_end = all_row_starts[rank + 1];
    
    /* Print distribution statistics */
    if (rank == 0) {
        printf("    Distribution:\n");
        for (int r = 0; r < num_ranks; r++) {
            int start = all_row_starts[r];
            int end = all_row_starts[r + 1];
            int rank_rows = end - start;
            long long rank_nnz = row_ptr[end] - row_ptr[start];
            double load_pct = 100.0 * rank_nnz / total_nnz;
            double row_pct = 100.0 * rank_rows / rows;
            
            printf("      Rank %d: rows [%d-%d) = %d rows (%.2f%%), %lld NNZ (%.2f%%)\n",
                   r, start, end, rank_rows, row_pct, rank_nnz, load_pct);
        }
    }
    
    free(all_row_starts);
}

/*------------------------------------------------------------------
 * Function:    calculate_block_cyclic_distribution
 * Purpose:     Block-cyclic distribution for better cache and load balance
 * 
 * Strategy:    
 *     1. Divide rows into B blocks (B >> num_ranks)
 *     2. Assign blocks cyclically: rank r gets blocks r, r+num_ranks, r+2*num_ranks, ...
 *     3. Smaller blocks improve load balance
 *     4. Cyclic pattern averages out irregularities
 * 
 * In args:     block_size    - Size of each block (rows per block)
 *                              Recommended: rows / (num_ranks * 10)
 * 
 * Benefits:    
 *     - Averages load imbalance across matrix
 *     - Good for matrices with localized dense regions
 *     - Maintains reasonable spatial locality
 * 
 * Drawbacks:   
 *     - More complex gather/scatter operations
 *     - Requires tracking multiple discontiguous ranges
 *     - Implementation complexity vs. performance gain trade-off
 * 
 * Note:        This returns only the FIRST block range for simplicity.
 *              Full implementation would return array of ranges.
 */
void calculate_block_cyclic_distribution(
    int rows,
    int num_ranks,
    int rank,
    int block_size,
    int *row_start,
    int *row_end
) {
    /* Simplified: return first block only */
    /* Full implementation would track all blocks for this rank */
    
    int num_blocks = (rows + block_size - 1) / block_size;
    int blocks_per_rank = num_blocks / num_ranks;
    int extra_blocks = num_blocks % num_ranks;
    
    /* This rank gets first block at position 'rank' */
    int first_block_id = rank;
    *row_start = first_block_id * block_size;
    *row_end = (*row_start + block_size < rows) ? *row_start + block_size : rows;
    
    if (rank == 0) {
        printf("  Load Balancing: Block-cyclic distribution\n");
        printf("    Block size: %d, Num blocks: %d, Blocks/rank: ~%d\n",
               block_size, num_blocks, blocks_per_rank);
        printf("    Note: Simplified implementation (first block only)\n");
    }
    
    /* TODO: Full implementation would store all block ranges for gather operations */
}

/*------------------------------------------------------------------
 * Function:    analyze_load_imbalance
 * Purpose:     Measure actual computational load imbalance across ranks
 * 
 * In args:     local_nnz      - This rank's NNZ count
 *              local_rows     - This rank's row count
 *              local_time_ms  - This rank's computation time (ms)
 *              rank           - Current MPI rank
 *              num_ranks      - Total number of ranks
 * 
 * Out args:    Prints load imbalance statistics to rank 0
 * 
 * Returns:     Imbalance factor (max_load / avg_load)
 *              1.0 = perfect balance, >1.2 = significant imbalance
 */
double analyze_load_imbalance(
    long long local_nnz,
    int local_rows,
    double local_time_ms,
    int rank,
    int num_ranks
) {
    /* Gather all ranks' data to rank 0 */
    long long *all_nnz = NULL;
    int *all_rows = NULL;
    double *all_times = NULL;
    
    if (rank == 0) {
        all_nnz = (long long *)malloc(num_ranks * sizeof(long long));
        all_rows = (int *)malloc(num_ranks * sizeof(int));
        all_times = (double *)malloc(num_ranks * sizeof(double));
    }
    
    MPI_Gather(&local_nnz, 1, MPI_LONG_LONG, all_nnz, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_rows, 1, MPI_INT, all_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_time_ms, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double imbalance_factor = 1.0;
    
    if (rank == 0) {
        long long total_nnz = 0;
        int total_rows = 0;
        double total_time = 0;
        long long max_nnz = 0, min_nnz = all_nnz[0];
        double max_time = 0, min_time = all_times[0];
        
        printf("\n  LOAD IMBALANCE ANALYSIS:\n");
        printf("  %-8s %-12s %-12s %-15s\n", "Rank", "Rows", "NNZ", "Compute (ms)");
        printf("  %s\n", "--------------------------------------------------------");
        
        for (int r = 0; r < num_ranks; r++) {
            printf("  %-8d %-12d %-12lld %-15.3f\n", 
                   r, all_rows[r], all_nnz[r], all_times[r]);
            
            total_nnz += all_nnz[r];
            total_rows += all_rows[r];
            total_time += all_times[r];
            
            if (all_nnz[r] > max_nnz) max_nnz = all_nnz[r];
            if (all_nnz[r] < min_nnz) min_nnz = all_nnz[r];
            if (all_times[r] > max_time) max_time = all_times[r];
            if (all_times[r] < min_time) min_time = all_times[r];
        }
        
        double avg_nnz = (double)total_nnz / num_ranks;
        double avg_time = total_time / num_ranks;
        
        double nnz_imbalance = (double)max_nnz / avg_nnz;
        double time_imbalance = max_time / avg_time;
        
        printf("  %s\n", "--------------------------------------------------------");
        printf("  Average:     %-12.0f %-12.0f %-15.3f\n", 
               (double)total_rows / num_ranks, avg_nnz, avg_time);
        printf("\n  Imbalance Factors:\n");
        printf("    NNZ:  max/avg = %.3f (max: %lld, min: %lld, avg: %.0f)\n",
               nnz_imbalance, max_nnz, min_nnz, avg_nnz);
        printf("    Time: max/avg = %.3f (max: %.3f ms, min: %.3f ms, avg: %.3f ms)\n",
               time_imbalance, max_time, min_time, avg_time);
        printf("\n  Load Balance Quality:\n");
        if (time_imbalance < 1.1) {
            printf("    ✓ EXCELLENT: Time imbalance < 1.1 (well balanced)\n");
        } else if (time_imbalance < 1.3) {
            printf("    ○ GOOD: Time imbalance < 1.3 (acceptable)\n");
        } else if (time_imbalance < 1.5) {
            printf("    △ FAIR: Time imbalance < 1.5 (moderate imbalance)\n");
        } else {
            printf("    ✗ POOR: Time imbalance >= 1.5 (significant imbalance)\n");
            printf("    Recommendation: Consider NNZ-based or hybrid distribution\n");
        }
        
        imbalance_factor = time_imbalance;
        
        free(all_nnz);
        free(all_rows);
        free(all_times);
    }
    
    /* Broadcast imbalance factor to all ranks */
    MPI_Bcast(&imbalance_factor, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return imbalance_factor;
}

/*------------------------------------------------------------------
 * Function:    select_distribution_strategy
 * Purpose:     Automatically select best distribution strategy for matrix
 * 
 * In args:     row_ptr       - CSR row pointer array
 *              rows          - Total number of rows
 *              num_ranks     - Number of MPI ranks
 * 
 * Returns:     LOAD_BALANCE_STRATEGY enum value
 * 
 * Heuristics:  
 *     1. Calculate coefficient of variation (CoV) of NNZ per row
 *     2. If CoV < 0.3: matrix is uniform → use ROW_BASED
 *     3. If 0.3 <= CoV < 0.7: moderate variation → use HYBRID
 *     4. If CoV >= 0.7: highly irregular → use NNZ_BASED
 */
LoadBalanceStrategy select_distribution_strategy(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank
) {
    /* Calculate NNZ per row statistics */
    long long total_nnz = row_ptr[rows];
    double mean_nnz = (double)total_nnz / rows;
    
    double variance = 0.0;
    for (int r = 0; r < rows; r++) {
        long long row_nnz = row_ptr[r + 1] - row_ptr[r];
        double diff = row_nnz - mean_nnz;
        variance += diff * diff;
    }
    variance /= rows;
    
    double std_dev = sqrt(variance);
    double cov = std_dev / mean_nnz;  // Coefficient of variation
    
    LoadBalanceStrategy strategy;
    
    if (cov < 0.3) {
        strategy = LB_ROW_BASED;
        if (rank == 0) {
            printf("  Matrix Analysis: CoV = %.4f (uniform) → ROW_BASED distribution\n", cov);
        }
    } else if (cov < 0.7) {
        strategy = LB_HYBRID;
        if (rank == 0) {
            printf("  Matrix Analysis: CoV = %.4f (moderate) → HYBRID distribution\n", cov);
        }
    } else {
        strategy = LB_NNZ_BASED;
        if (rank == 0) {
            printf("  Matrix Analysis: CoV = %.4f (irregular) → NNZ_BASED distribution\n", cov);
        }
    }
    
    return strategy;
}
