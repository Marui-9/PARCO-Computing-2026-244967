#!/bin/bash
# Download larger matrices from SuiteSparse Collection
# https://sparse.tamu.edu/
#
# Matrix selection strategy:
#   - 8 large matrices (1.5M-5M rows): Suitable for weak scaling
#   - Emphasis on matrices with 20M-70M NNZ for meaningful computation per process
#   - Includes social networks, structural FEA, web graphs, and synthetic matrices
#   - Total: 8 matrices focused on true weak scaling (200k rows/proc at 16-32 processes)
#
# Matrices are auto-renamed to match project convention: {rows_in_k}k_{density%}.mtx
# Example: A 1.5M row matrix with 0.0267% density -> 1500k_0p0267.mtx
#
# Data size: ~1.5-2 GB total
# Estimated download time: 3-5 minutes (depends on internet speed)
#
# Usage: ./download_matrices_bigger.sh

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/matrices"
TEMP_DIR="${PROJECT_ROOT}/.matrix_download_tmp"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "=============================================="
echo "SuiteSparse Matrix Downloader (Large Matrices)"
echo "Output directory: $OUTPUT_DIR"
echo "Naming convention: {rows_in_k}k_{density%}.mtx"
echo "Estimated total download size: ~1.5-2 GB"
echo "Estimated download time: 3-5 minutes (depending on internet speed)"
echo "=============================================="

# Base URL for SuiteSparse Matrix Collection
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# 8 large matrices: 1.5M-5M rows, 30M-500M NNZ
# Suitable for weak scaling with 200k rows/proc at 16-32 processes
declare -A MATRICES=(
    # Large sparse matrices (1.5M-5M rows with substantial NNZ)
    ["SNAP/soc-LiveJournal1"]="soc-LiveJournal1"          # 4.8M rows, 69M nnz - social network
    ["Janna/Flan_1565"]="Flan_1565"                       # 1.5M rows, 60M nnz - structural
    ["Janna/Geo_1438"]="Geo_1438"                         # 1.4M rows, 32M nnz - structural
    ["LAW/it-2004"]="it-2004"                             # 1.9M rows, 39M nnz - web graph
    ["DIMACS10/delaunay_n21"]="delaunay_n21"              # 2.1M rows, 6.3M nnz - Delaunay
    ["Schenk_AFE/af_shell10"]="af_shell10"                # 1.8M rows, 11M nnz - structural
    ["Kron/kron_g500-logn20"]="kron_g500-logn20"          # 1.06M rows, 22M nnz - Kronecker
    ["SNAP/web-Google"]="web-Google"                      # 916k rows, 5.1M nnz - web graph
)

# Function to generate filename from matrix properties
# Convention: {rows_in_k}k_{density_percent}.mtx
# Density = nnz / (rows * cols) * 100
# Decimal point replaced with 'p' (e.g., 0.03% -> 0p03)
generate_filename() {
    local mtxfile="$1"
    
    # Parse Matrix Market header to get dimensions
    # Format: rows cols nnz (first non-comment line after header)
    local dimensions=$(grep -v "^%" "$mtxfile" | head -1)
    local rows=$(echo "$dimensions" | awk '{print $1}')
    local cols=$(echo "$dimensions" | awk '{print $2}')
    local nnz=$(echo "$dimensions" | awk '{print $3}')
    
    if [[ -z "$rows" ]] || [[ -z "$cols" ]] || [[ -z "$nnz" ]]; then
        echo ""
        return 1
    fi
    
    # Calculate rows in thousands (rounded)
    local rows_k=$(echo "scale=0; ($rows + 500) / 1000" | bc)
    if [[ "$rows_k" -eq 0 ]]; then
        rows_k=1
    fi
    
    # Calculate density percentage: nnz / (rows * cols) * 100
    # Use bc for floating point, handle very small densities
    local density=$(echo "scale=10; $nnz / ($rows * $cols) * 100" | bc)
    
    # Format density: replace decimal point with 'p', remove trailing zeros
    # Examples: 0.03 -> 0p03, 1.5 -> 1p5, 0.008 -> 0p008
    local density_formatted=$(echo "$density" | sed 's/^\./0./' | \
        awk '{
            # Round to reasonable precision based on magnitude
            if ($1 < 0.01) {
                printf "%.4f", $1
            } else if ($1 < 0.1) {
                printf "%.3f", $1
            } else if ($1 < 1) {
                printf "%.2f", $1
            } else if ($1 < 10) {
                printf "%.1f", $1
            } else {
                printf "%.0f", $1
            }
        }' | sed 's/\./p/' | sed 's/p0*$//' | sed 's/$//' )
    
    # Handle edge case where density might just be "0"
    if [[ "$density_formatted" == "0" ]] || [[ -z "$density_formatted" ]]; then
        density_formatted="0p001"
    fi
    
    echo "${rows_k}k_${density_formatted}.mtx"
}

# Function to download and extract a single matrix
download_matrix() {
    local matrix_path="$1"
    local description="$2"
    local group=$(dirname "$matrix_path")
    local name=$(basename "$matrix_path")
    local url="${BASE_URL}/${matrix_path}.tar.gz"
    local tarfile="${TEMP_DIR}/${name}.tar.gz"
    local temp_mtxfile="${TEMP_DIR}/${name}.mtx"
    
    echo ""
    echo "Downloading: $name"
    echo "  Description: $description"
    echo "  URL: $url"
    
    # Download with wget (follow redirects, show progress)
    if wget -q --show-progress -O "$tarfile" "$url" 2>/dev/null || \
       curl -L -o "$tarfile" "$url" 2>/dev/null; then
        echo "  Extracting..."
        
        # Extract the .mtx file from the tarball
        # SuiteSparse tarballs contain: name/name.mtx
        tar -xzf "$tarfile" -C "$TEMP_DIR" 2>/dev/null || true
        
        # Find and move the .mtx file
        local found_mtx=$(find "$TEMP_DIR" -name "*.mtx" -type f 2>/dev/null | head -1)
        
        if [[ -n "$found_mtx" ]] && [[ -f "$found_mtx" ]]; then
            mv "$found_mtx" "$temp_mtxfile"
            
            # Clean up extracted directory
            find "$TEMP_DIR" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
            
            # Generate new filename based on matrix properties
            local new_filename=$(generate_filename "$temp_mtxfile")
            
            if [[ -n "$new_filename" ]]; then
                local final_path="${OUTPUT_DIR}/${new_filename}"
                
                # Check if file with same name already exists
                if [[ -f "$final_path" ]]; then
                    echo "  [SKIP] $new_filename already exists"
                    rm -f "$temp_mtxfile"
                else
                    mv "$temp_mtxfile" "$final_path"
                    local size=$(du -h "$final_path" | cut -f1)
                    echo "  [OK] Saved: $new_filename ($size)"
                    echo "       Original: $name"
                fi
            else
                echo "  [WARN] Could not parse matrix dimensions for $name"
                rm -f "$temp_mtxfile"
            fi
        else
            echo "  [WARN] No .mtx file found in archive for $name"
        fi
        
        # Clean up tarball
        rm -f "$tarfile"
    else
        echo "  [ERROR] Failed to download $name"
        rm -f "$tarfile"
        return 1
    fi
}

# Main script logic
main() {
    echo ""
    echo "Downloading 8 large matrices from SuiteSparse Collection..."
    echo ""
    
    # Download all matrices in MATRICES array
    for matrix_path in "${!MATRICES[@]}"; do
        download_matrix "$matrix_path" "${MATRICES[$matrix_path]}"
    done
    
    # Clean up temp directory
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "==============================================="
    echo "Matrix download complete!"
    echo ""
    echo "Downloaded matrices in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR"/*.mtx 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no matrices found)"
    
    # Calculate total size
    local total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    echo ""
    echo "Total download size: $total_size"
    echo "Matrix files ready for weak scaling benchmarks."
    echo "=============================================="
}

# Show usage
usage() {
    echo "Usage: $0"
    echo ""
    echo "Downloads 8 large matrices from SuiteSparse Collection"
    echo ""
    echo "Matrix categories:"
    echo "  - Large (8):     1.5M-5M rows, 20M-70M NNZ (weak scaling)"
    echo ""
    echo "Suitable for:"
    echo "  - Weak scaling benchmarks (200k rows/proc at 16-32 processes)"
    echo "  - Testing MPI communication modes at realistic scale"
    echo "  - Each process gets ~40M NNZ at 32 processes"
    echo ""
    echo "Total size: ~1.5-2 GB"
    echo "Estimated download time: 3-5 minutes (depending on internet speed)"
    echo ""
    echo "Output: Matrices saved to ./matrices/ with naming convention:"
    echo "        {rows_in_k}k_{density%}.mtx"
    echo ""
    echo "Examples:"
    echo "  1500k_0p0267.mtx = 1.5M rows, 0.0267% density"
    echo "  4800k_0p0003.mtx = 4.8M rows, 0.0003% density (soc-LiveJournal1)"
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

main "$@"
