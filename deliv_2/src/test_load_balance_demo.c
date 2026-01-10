/* File:     
 *     test_load_balance_demo.c 
 *
 * Purpose:  
 *     Demonstrate load balancing strategies for distributed SpMV
 *     Shows work distribution comparison between different strategies
 *
 * Compile:  mpicc -g -Wall -O3 -o test_lb_demo \
 *               test_load_balance_demo.c load_balance.c -lm
 * 
 * Usage:
 *     mpirun -np 4 ./test_lb_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "load_balance.h"

void simulate_matrix_distribution(int rows, long long total_nnz, int num_ranks);
void compare_strategies(const int *row_ptr, int rows, int num_ranks);

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        printf("\n");
        printf("========================================================================\n");
        printf("  LOAD BALANCING DEMONSTRATION FOR DISTRIBUTED SpMV\n");
        printf("========================================================================\n");
        printf("\n");
        printf("MPI Ranks: %d\n\n", size);
    }
    
    /* Simulate different matrix scenarios */
    
    /* Scenario 1: Uniform matrix (low CoV) */
    if (rank == 0) {
        printf("SCENARIO 1: UNIFORM MATRIX (Low Irregularity)\n");
        printf("--------------------------------------------------------------------\n");
    }
    
    int rows1 = 10000;
    long long nnz1 = 50000;  // 5 NNZ/row average, uniform
    
    int *row_ptr1 = (int *)malloc((rows1 + 1) * sizeof(int));
    row_ptr1[0] = 0;
    for (int i = 0; i < rows1; i++) {
        row_ptr1[i + 1] = row_ptr1[i] + 5;  // Exactly 5 NNZ per row
    }
    
    if (rank == 0) {
        printf("Matrix: %d rows, %lld NNZ, %.2f NNZ/row (uniform)\n", rows1, nnz1, (double)nnz1/rows1);
        printf("Expected: Row-based and NNZ-based should produce similar distributions\n\n");
    }
    
    compare_strategies(row_ptr1, rows1, size);
    free(row_ptr1);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Scenario 2: Irregular matrix (high CoV) */
    if (rank == 0) {
        printf("\n\nSCENARIO 2: IRREGULAR MATRIX (High Irregularity)\n");
        printf("--------------------------------------------------------------------\n");
    }
    
    int rows2 = 10000;
    long long nnz2 = 50000;
    
    int *row_ptr2 = (int *)malloc((rows2 + 1) * sizeof(int));
    row_ptr2[0] = 0;
    
    /* Create power-law distribution: first 20% of rows have 80% of NNZ */
    int dense_rows = rows2 / 5;  // 20% of rows
    long long dense_nnz = (nnz2 * 4) / 5;  // 80% of NNZ
    long long sparse_nnz = nnz2 - dense_nnz;
    int sparse_rows = rows2 - dense_rows;
    
    for (int i = 0; i < dense_rows; i++) {
        row_ptr2[i + 1] = row_ptr2[i] + (dense_nnz / dense_rows);
    }
    for (int i = dense_rows; i < rows2; i++) {
        row_ptr2[i + 1] = row_ptr2[i] + (sparse_nnz / sparse_rows);
    }
    row_ptr2[rows2] = nnz2;  // Fix total
    
    if (rank == 0) {
        printf("Matrix: %d rows, %lld NNZ, %.2f NNZ/row (average)\n", rows2, nnz2, (double)nnz2/rows2);
        printf("Distribution: First 20%% of rows contain 80%% of NNZ (power-law)\n");
        printf("Expected: NNZ-based should balance workload much better than row-based\n\n");
    }
    
    compare_strategies(row_ptr2, rows2, size);
    free(row_ptr2);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Scenario 3: Real-world example simulation */
    if (rank == 0) {
        printf("\n\nSCENARIO 3: REALISTIC SPARSE MATRIX (Moderate Irregularity)\n");
        printf("--------------------------------------------------------------------\n");
    }
    
    int rows3 = 36417;  // Similar to 36k_0p17.mtx
    long long nnz3 = 2190591;
    
    int *row_ptr3 = (int *)malloc((rows3 + 1) * sizeof(int));
    row_ptr3[0] = 0;
    
    /* Simulate realistic distribution with clusters */
    for (int i = 0; i < rows3; i++) {
        int base_nnz = 60;  // Average
        int cluster = i / 1000;  // Cluster every 1000 rows
        int variation = ((i * 17) % 100) - 50;  // Pseudo-random variation
        int row_nnz = base_nnz + variation + (cluster % 3) * 20;  // Cluster effect
        if (row_nnz < 1) row_nnz = 1;
        
        row_ptr3[i + 1] = row_ptr3[i] + row_nnz;
    }
    
    /* Normalize to actual NNZ */
    long long simulated_nnz = row_ptr3[rows3];
    for (int i = 1; i <= rows3; i++) {
        row_ptr3[i] = (row_ptr3[i] * nnz3) / simulated_nnz;
    }
    row_ptr3[rows3] = nnz3;
    
    if (rank == 0) {
        printf("Matrix: %d rows, %lld NNZ, %.2f NNZ/row (clustered distribution)\n", 
               rows3, nnz3, (double)nnz3/rows3);
        printf("Expected: Hybrid strategy provides good balance between workload and locality\n\n");
    }
    
    compare_strategies(row_ptr3, rows3, size);
    free(row_ptr3);
    
    if (rank == 0) {
        printf("\n");
        printf("========================================================================\n");
        printf("  SUMMARY AND RECOMMENDATIONS\n");
        printf("========================================================================\n");
        printf("\n");
        printf("Load Balance Quality Metrics:\n");
        printf("  • Imbalance Factor = max(NNZ) / avg(NNZ)\n");
        printf("    - < 1.1: Excellent (< 10%% imbalance)\n");
        printf("    - < 1.3: Good (< 30%% imbalance)\n");
        printf("    - > 1.5: Poor (> 50%% imbalance) → performance degradation\n");
        printf("\n");
        printf("Strategy Selection:\n");
        printf("  • UNIFORM matrices (CoV < 0.3):    ROW_BASED sufficient\n");
        printf("  • MODERATE matrices (CoV 0.3-0.7): HYBRID (α=0.5-0.7)\n");
        printf("  • IRREGULAR matrices (CoV > 0.7):  NNZ_BASED essential\n");
        printf("\n");
        printf("Performance Impact:\n");
        printf("  • Reducing imbalance 1.5 → 1.1 can improve speedup by 15-30%%\n");
        printf("  • Critical at 5-8 nodes where communication is high\n");
        printf("  • Enables better scaling beyond 8 nodes\n");
        printf("\n");
        printf("Next Steps:\n");
        printf("  1. Integrate load_balance.c into your SpMV application\n");
        printf("  2. Use analyze_load_imbalance() to measure current performance\n");
        printf("  3. Select strategy with select_distribution_strategy()\n");
        printf("  4. Apply distribution before computation\n");
        printf("  5. Benchmark and compare results\n");
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}

void compare_strategies(const int *row_ptr, int rows, int num_ranks) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    long long total_nnz = row_ptr[rows];
    
    if (rank == 0) {
        printf("Comparing distribution strategies:\n\n");
        
        /* Strategy 1: Row-based (current baseline) */
        printf("1. ROW-BASED Distribution (static, uniform row assignment):\n");
        printf("   %-8s %-15s %-15s %-15s %-15s\n", "Rank", "Row Range", "Num Rows", "NNZ", "Workload %%");
        printf("   --------------------------------------------------------------------------------\n");
        
        long long max_nnz_row = 0, min_nnz_row = total_nnz;
        
        for (int r = 0; r < num_ranks; r++) {
            int rows_per_rank = rows / num_ranks;
            int extra = rows % num_ranks;
            
            int start = r * rows_per_rank + (r < extra ? r : extra);
            int end = (r < extra) ? start + rows_per_rank + 1 : start + rows_per_rank;
            
            long long rank_nnz = row_ptr[end] - row_ptr[start];
            double workload_pct = 100.0 * rank_nnz / total_nnz;
            
            if (rank_nnz > max_nnz_row) max_nnz_row = rank_nnz;
            if (rank_nnz < min_nnz_row) min_nnz_row = rank_nnz;
            
            printf("   %-8d [%5d-%5d)   %10d      %12lld      %10.2f%%\n",
                   r, start, end, end - start, rank_nnz, workload_pct);
        }
        
        double avg_nnz_row = (double)total_nnz / num_ranks;
        double imbalance_row = (double)max_nnz_row / avg_nnz_row;
        
        printf("   --------------------------------------------------------------------------------\n");
        printf("   Imbalance Factor: %.3f (max: %lld, min: %lld, avg: %.0f)\n", 
               imbalance_row, max_nnz_row, min_nnz_row, avg_nnz_row);
        
        if (imbalance_row < 1.1) {
            printf("   Quality: ✓ EXCELLENT\n");
        } else if (imbalance_row < 1.3) {
            printf("   Quality: ○ GOOD\n");
        } else if (imbalance_row < 1.5) {
            printf("   Quality: △ FAIR\n");
        } else {
            printf("   Quality: ✗ POOR - significant imbalance detected!\n");
        }
        
        /* Strategy 2: NNZ-based */
        printf("\n2. NNZ-BASED Distribution (balanced by computational workload):\n");
        printf("   %-8s %-15s %-15s %-15s %-15s\n", "Rank", "Row Range", "Num Rows", "NNZ", "Workload %%");
        printf("   --------------------------------------------------------------------------------\n");
        
        long long target_nnz = total_nnz / num_ranks;
        long long accumulated_nnz = 0;
        int current_rank = 0;
        int *rank_starts = (int *)malloc((num_ranks + 1) * sizeof(int));
        rank_starts[0] = 0;
        
        for (int r = 0; r < rows; r++) {
            long long row_nnz = row_ptr[r + 1] - row_ptr[r];
            accumulated_nnz += row_nnz;
            
            if (accumulated_nnz >= target_nnz * (current_rank + 1) && current_rank < num_ranks - 1) {
                current_rank++;
                rank_starts[current_rank] = r + 1;
            }
        }
        rank_starts[num_ranks] = rows;
        
        long long max_nnz_nnzb = 0, min_nnz_nnzb = total_nnz;
        
        for (int r = 0; r < num_ranks; r++) {
            int start = rank_starts[r];
            int end = rank_starts[r + 1];
            long long rank_nnz = row_ptr[end] - row_ptr[start];
            double workload_pct = 100.0 * rank_nnz / total_nnz;
            
            if (rank_nnz > max_nnz_nnzb) max_nnz_nnzb = rank_nnz;
            if (rank_nnz < min_nnz_nnzb) min_nnz_nnzb = rank_nnz;
            
            printf("   %-8d [%5d-%5d)   %10d      %12lld      %10.2f%%\n",
                   r, start, end, end - start, rank_nnz, workload_pct);
        }
        
        double avg_nnz_nnzb = (double)total_nnz / num_ranks;
        double imbalance_nnzb = (double)max_nnz_nnzb / avg_nnz_nnzb;
        
        printf("   --------------------------------------------------------------------------------\n");
        printf("   Imbalance Factor: %.3f (max: %lld, min: %lld, avg: %.0f)\n", 
               imbalance_nnzb, max_nnz_nnzb, min_nnz_nnzb, avg_nnz_nnzb);
        
        if (imbalance_nnzb < 1.1) {
            printf("   Quality: ✓ EXCELLENT\n");
        } else if (imbalance_nnzb < 1.3) {
            printf("   Quality: ○ GOOD\n");
        } else if (imbalance_nnzb < 1.5) {
            printf("   Quality: △ FAIR\n");
        } else {
            printf("   Quality: ✗ POOR\n");
        }
        
        /* Comparison */
        printf("\n3. COMPARISON:\n");
        printf("   %-25s %15s %15s %20s\n", "Metric", "ROW-BASED", "NNZ-BASED", "Improvement");
        printf("   -------------------------------------------------------------------------------\n");
        printf("   %-25s %15.3f %15.3f %19.1f%%\n", "Imbalance Factor", 
               imbalance_row, imbalance_nnzb, 
               100.0 * (imbalance_row - imbalance_nnzb) / imbalance_row);
        
        /* Estimate speedup improvement */
        /* Assuming: speedup_loss = imbalance_factor - 1.0 */
        double efficiency_row = 1.0 / imbalance_row;
        double efficiency_nnzb = 1.0 / imbalance_nnzb;
        double speedup_gain = 100.0 * (efficiency_nnzb - efficiency_row) / efficiency_row;
        
        printf("   %-25s %14.1f%% %14.1f%% %18.1f%%\n", "Parallel Efficiency",
               100.0 * efficiency_row, 100.0 * efficiency_nnzb, speedup_gain);
        
        printf("\n   Recommendation: ");
        if (imbalance_row > 1.3 && imbalance_nnzb < 1.2) {
            printf("✓ NNZ-BASED distribution strongly recommended (%.0f%% speedup gain)\n", speedup_gain);
        } else if (imbalance_row > 1.15 && imbalance_nnzb < 1.15) {
            printf("○ NNZ-BASED provides moderate improvement (%.0f%% speedup gain)\n", speedup_gain);
        } else {
            printf("ROW-BASED is sufficient for this matrix\n");
        }
        
        free(rank_starts);
    }
}
