#!/bin/bash
# Download matrices from SuiteSparse Collection
# https://sparse.tamu.edu/
#
# Matrix selection strategy:
#   - 3 small matrices (100k-300k rows): Quick testing, baseline comparison
#   - 5 sweet spot matrices (500k-1.8M rows, 10-50M nnz): Larger computation
#   - 4 big matrices (4.8M-10M rows, 69-405M nnz): Largest matrices
#   - 2 commented matrices (16.8M-18.5M rows): Available but not auto-downloaded
#
# Matrices are auto-renamed to match project convention: {rows_in_k}k_{density%}.mtx
# Example: A 1.4M row matrix with 0.015% density -> 1400k_0p015.mtx
#
# Data size: ~8-10 GB total
# Estimated download time: 20-40 minutes (depends on internet speed)
#
# Usage: ./download_matrices_mpi.sh

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/matrices"
TEMP_DIR="${PROJECT_ROOT}/.matrix_download_tmp"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "=============================================="
echo "SuiteSparse Matrix Downloader"
echo "Output directory: $OUTPUT_DIR"
echo "Naming convention: {rows_in_k}k_{density%}.mtx"
echo "Estimated total download size: ~8-10 GB"
echo "Estimated download time: 20-40 minutes (depending on internet speed)"
echo "=============================================="

# Base URL for SuiteSparse Matrix Collection
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# 12 matrices spanning 3 size categories
declare -A MATRICES=(
    # Small matrices: 100k-300k rows
    # Quick testing, baseline comparison
    ["Mycielski/mycielskian17"]="mycielskian17"           # 131k rows
    ["DIMACS10/delaunay_n18"]="delaunay_n18"             # 262k rows, 0.79M nnz
    ["Williams/pdb1HYS"]="pdb1HYS"                        # 36k rows
    
    # Medium: 500k-1.8M rows, 10-50M nnz
    # Larger computation
    ["SNAP/web-Google"]="web-Google"                      # 916k rows, 5M nnz
    ["Janna/Geo_1438"]="Geo_1438"                         # 1.4M rows, 15M nnz
    ["Schenk_AFE/af_shell10"]="af_shell10"                # 1.8M rows, 11M nnz
    ["AMD/G3_circuit"]="G3_circuit"                       # 1.6M rows, 4.7M nnz
    ["LAW/cnr-2000"]="cnr-2000"                           # 325k rows, 3M nnz
    
    # Big matrices: 4.8M-10M rows
    # Largest matrices
    ["SNAP/com-LiveJournal"]="com-LiveJournal"            # 4.8M rows, 69M nnz
    ["vanHeukelum/cage15"]="cage15"                       # 5.2M rows, 99M nnz
    ["SNAP/com-Youtube"]="com-Youtube"                    # 6.5M rows, 160M nnz
    ["LAW/twitter7"]="twitter7"                           # 10M rows, 405M nnz
   # ["DIMACS10/rgg_n_2_24_s0"]="rgg_n_2_24_s0"            # 16.8M rows, 265M nnz
   # ["LAW/uk-2002"]="uk-2002"                             # 18.5M rows, 298M nnz
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
    echo "Downloading 14 matrices..."
    echo "  (~8-10 GB total, estimated time: 20-40 minutes)"
    echo ""
    
    # Download all matrices in MATRICES array
    for matrix_path in "${!MATRICES[@]}"; do
        download_matrix "$matrix_path" "${MATRICES[$matrix_path]}"
    done
    
    # Clean up temp directory
    rm -rf "$TEMP_DIR"
    
    echo ""
    echo "=============================================="
    echo "Download complete!"
    echo ""
    echo "Downloaded matrices in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR"/*.mtx 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (no matrices found)"
    
    # Calculate total size
    local total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    echo ""
    echo "Total downloaded size: $total_size"
    echo "Naming convention: {rows_in_k}k_{density%}.mtx"
    echo ""
    echo "Download complete! Matrices ready for use."
    echo "=============================================="
}

# Show usage
usage() {
    echo "Usage: $0"
    echo ""
    echo "Downloads 14 matrices from SuiteSparse Collection"
    echo ""
    echo "Matrix categories:"
    echo "  - Small (3):     100k-300k rows"
    echo "  - Medium (5):    500k-1.8M rows"
    echo "  - Big (4):       4.8M-10M rows (automatically downloaded)"
    echo "  - Extra (2):     16.8M-18.5M rows (commented, available on demand)"
    echo ""
    echo "Total size: ~8-10 GB (for 14 matrices)"
    echo "Estimated download time: 20-40 minutes (depending on internet speed)"
    echo ""
    echo "Output: Matrices saved to ./matrices/ with naming convention:"
    echo "        {rows_in_k}k_{density%}.mtx"
    echo ""
    echo "Examples:"
    echo "  131k_0p0018.mtx  = 131k rows, 0.0018% density"
    echo "  1400k_0p015.mtx  = 1.4M rows, 0.015% density"
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

main "$@"
