/* File:     
 *     load_balance.h 
 *
 * Purpose:  
 *     Header file for load balancing strategies in distributed SpMV
 *
 * Author: PARCO Computing Project
 * Date: January 2026
 */

#ifndef LOAD_BALANCE_H
#define LOAD_BALANCE_H

#include <math.h>

/* Load balancing strategies */
typedef enum {
    LB_ROW_BASED,      /* Static row-wise partitioning (current baseline) */
    LB_NNZ_BASED,      /* NNZ-weighted distribution */
    LB_HYBRID,         /* Hybrid: blend NNZ and row balance */
    LB_BLOCK_CYCLIC    /* Block-cyclic for averaged load balance */
} LoadBalanceStrategy;

/*------------------------------------------------------------------
 * Function:    calculate_nnz_based_distribution
 * Purpose:     Compute load-balanced row distribution based on NNZ count
 * 
 * In args:     row_ptr       - CSR row pointer array (size: rows+1)
 *              rows          - Total number of matrix rows
 *              num_ranks     - Number of MPI ranks to distribute across
 *              rank          - Current MPI rank
 * 
 * Out args:    row_start     - Starting row index for this rank
 *              row_end       - Ending row index for this rank (exclusive)
 */
void calculate_nnz_based_distribution(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank,
    int *row_start,
    int *row_end
);

/*------------------------------------------------------------------
 * Function:    calculate_hybrid_distribution
 * Purpose:     Hybrid strategy combining NNZ balance with row count balance
 * 
 * In args:     alpha         - Blend factor (0.0 = row-based, 1.0 = NNZ-based)
 */
void calculate_hybrid_distribution(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank,
    int *row_start,
    int *row_end,
    double alpha
);

/*------------------------------------------------------------------
 * Function:    calculate_block_cyclic_distribution
 * Purpose:     Block-cyclic distribution for better load averaging
 * 
 * In args:     block_size    - Size of each block (rows per block)
 */
void calculate_block_cyclic_distribution(
    int rows,
    int num_ranks,
    int rank,
    int block_size,
    int *row_start,
    int *row_end
);

/*------------------------------------------------------------------
 * Function:    analyze_load_imbalance
 * Purpose:     Measure actual computational load imbalance across ranks
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
);

/*------------------------------------------------------------------
 * Function:    select_distribution_strategy
 * Purpose:     Automatically select best distribution strategy for matrix
 * 
 * Returns:     LoadBalanceStrategy enum value based on matrix characteristics
 */
LoadBalanceStrategy select_distribution_strategy(
    const int *row_ptr,
    int rows,
    int num_ranks,
    int rank
);

#endif /* LOAD_BALANCE_H */
