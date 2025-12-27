#!/bin/bash
# Download 10 large square sparse matrices from SuiteSparse Matrix Collection
# https://sparse.tamu.edu/
#
# Selected matrices span 1M to 100M nnz for distributed SpMV benchmarking
# with 2-3 nodes (192-288 threads).
#
# Matrices are auto-renamed to match project convention: {rows_in_k}k_{density%}.mtx
# Example: A 1.5M row matrix with 0.03% density -> 1500k_0p03.mtx
#
# Data size: ~2.5 GB total (fits in 10 minutes at 50 Mbps)
# Estimated download time at 50 Mbps: ~7 minutes
#
# Usage: ./download_matrices.sh

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/matrices"
TEMP_DIR="${PROJECT_ROOT}/.matrix_download_tmp"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

echo "=============================================="
echo "SuiteSparse Matrix Downloader for Distributed SpMV"
echo "Output directory: $OUTPUT_DIR"
echo "Naming convention: {rows_in_k}k_{density%}.mtx"
echo "Estimated download size: ~2.5 GB"
echo "Estimated time (50 Mbps): ~7 minutes"
echo "=============================================="

# Base URL for SuiteSparse Matrix Collection
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# 10 square matrices spanning 1M to 100M nnz
# Selected for good compression, square structure, and representativeness
declare -A MATRICES=(
    # ~1-5M nnz: Small matrices
    ["Williams/cop20k_A"]="cop20k_A"
    ["SNAP/amazon0312"]="amazon0312"
    
    # ~5-15M nnz: Medium matrices
    ["SNAP/web-Google"]="web-Google"
    ["AMD/G3_circuit"]="G3_circuit"
    ["Janna/Geo_1438"]="Geo_1438"
    
    # ~15-50M nnz: Large matrices
    ["Freescale/circuit5M"]="circuit5M"
    ["Schenk_AFE/af_shell10"]="af_shell10"
    
    # ~50-100M nnz: Very large matrices
    ["SNAP/com-LiveJournal"]="com-LiveJournal"
    ["Janna/Flan_1565"]="Flan_1565"
    ["vanHeukelum/cage15"]="cage15"
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

# Function to download a subset of matrices by size category
download_by_category() {
    local category="$1"
    
    case "$category" in
        "medium")
            echo "Downloading medium matrices (1-10M nnz)..."
            download_matrix "Williams/cop20k_A" "${MATRICES[Williams/cop20k_A]}"
            download_matrix "SNAP/amazon0312" "${MATRICES[SNAP/amazon0312]}"
            download_matrix "SNAP/web-Google" "${MATRICES[SNAP/web-Google]}"
            download_matrix "AMD/G3_circuit" "${MATRICES[AMD/G3_circuit]}"
            ;;
        "large")
            echo "Downloading large matrices (10-50M nnz)..."
            download_matrix "Gleich/wikipedia-20070206" "${MATRICES[Gleich/wikipedia-20070206]}"
            download_matrix "Janna/Geo_1438" "${MATRICES[Janna/Geo_1438]}"
            download_matrix "Schenk_AFE/af_shell10" "${MATRICES[Schenk_AFE/af_shell10]}"
            ;;
        "xlarge")
            echo "Downloading extra-large matrices (50M+ nnz)..."
            download_matrix "SNAP/com-Orkut" "${MATRICES[SNAP/com-Orkut]}"
            download_matrix "LAW/hollywood-2009" "${MATRICES[LAW/hollywood-2009]}"
            download_matrix "vanHeukelum/cage15" "${MATRICES[vanHeukelum/cage15]}"
            download_matrix "Janna/Flan_1565" "${MATRICES[Janna/Flan_1565]}"
            ;;
        "all")
            for matrix in "${!MATRICES[@]}"; do
                download_matrix "$matrix" "${MATRICES[$matrix]}"
            done
            ;;
        "recommended")
            echo "Downloading recommended matrices for 2-3 node benchmarking..."
            # Good mix of sizes and patterns
            download_matrix "AMD/G3_circuit" "${MATRICES[AMD/G3_circuit]}"
            download_matrix "SNAP/web-Google" "${MATRICES[SNAP/web-Google]}"
            download_matrix "Janna/Geo_1438" "${MATRICES[Janna/Geo_1438]}"
            download_matrix "SNAP/com-LiveJournal" "${MATRICES[SNAP/com-LiveJournal]}"
            ;;
        *)
            echo "Unknown category: $category"
            echo "Valid categories: medium, large, xlarge, recommended, all"
            return 1
            ;;
    esac
}

# Print available matrices
print_available() {
    echo ""
    echo "Available matrices:"
    echo "==================="
    for matrix in "${!MATRICES[@]}"; do
        echo "  $(basename $matrix): ${MATRICES[$matrix]}"
    done
    echo ""
    echo "Categories:"
    echo "  medium     - 1-10M nnz (good for initial testing)"
    echo "  large      - 10-50M nnz (good for 2-node testing)"  
    echo "  xlarge     - 50M+ nnz (good for 3-node testing)"
    echo "  recommended - curated mix for benchmarking"
    echo "  all        - download everything"
}

# Main script logic
main() {
    echo ""
    echo "Downloading 10 square matrices (1M-100M nnz)..."
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
    echo "Total size: $total_size"
    echo "Naming convention: {rows_in_k}k_{density%}.mtx"
    echo "=============================================="
}

# Show usage
usage() {
    echo "Usage: $0"
    echo ""
    echo "Downloads 10 square matrices from SuiteSparse Collection"
    echo "Total size: ~2.5 GB"
    echo "Estimated time (50 Mbps): ~7 minutes"
    echo ""
    echo "Output: Matrices saved to ./matrices/ with naming convention:"
    echo "        {rows_in_k}k_{density%}.mtx"
    echo ""
    echo "Examples:"
    echo "  1500k_0p03.mtx  = 1.5M rows, 0.03% density"
    echo "  400k_0p8.mtx    = 400k rows, 0.8% density"
}

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

main "$@"
