"""
================================================================================
NLSY79 DATA PROCESSING: STEP 1 - VARIABLE CLASSIFICATION
================================================================================

PRE-CLEANING COMPLETED (regitze_figuring_out_shiit copy.ipynb & ML_cleaning_parquet.ipynb):

1. DATASET SOURCE & FORMAT
   - Original: nlsy79_all_1979-2022.csv (6.25 GB, 12,686 participants × 178,014 variables)
   - Converted to parquet for processing efficiency

2. MISSING VALUE STANDARDIZATION
   - All negative values converted to NA (encode non-measurements):
     * -1: Refusal
     * -2: Don't Know
     * -3: Invalid Skip
     * -4: SKIP
     * -5: NON-INTERVIEW

3. PARTICIPANT FILTERING (Rotter Score)
   - Removed 145 participants without Rotter Score in 1979 (R0153710)
   - Removed 5,525 participants without 2nd Rotter Score (all NA in T4998510, T5733800, T8162800)
   - Remaining: 7,016 participants

4. ROTTER SCORE 2 CONSOLIDATION
   - Created `rotter_score_2` from three timepoints:
     * 2014 (T4998510): n=6,438
     * 2016 (T5733800): n=253  
     * 2018 (T8162800): n=109
   - 8 participants with multiple measurements: averaged and rounded to nearest integer
   - Dropped original T4998510, T5733800, T8162800 columns

5. ROTTER SUB-QUESTIONS REMOVED
   - Dropped 32 sub-question variables (ROTTER_1A→ROTTER_4B) from 1979, 2014, 2016, 2018
   - These are redundant with final Rotter Score calculations

6. VARIABLE SPARSITY FILTERING  
   - Removed variables with >25% NAs
   - Dropped: 163,730 variables → Remaining: 14,284 variables

7. ZERO-VARIANCE FILTERING
   - Removed 13 variables with ≤1 unique value (no information)

8. DUPLICATE CHECK
   - Verified: No duplicate column names

9. FUTURE DATA REMOVAL
   - Removed 2,259 variables from 2016-2022 (measured after primary Rotter Score 2 in 2014)
   - Includes both direct year-suffixed variables and XRND variables containing 2015-2022 in titles

DATASET STATE AT STEP1 INPUT:
- Source file: nlsy79_cleaned_4_new.parquet
- Participants: 7,016 (before 25% missingness filter in Step 1)
- Variables: 11,981 (178,014 → 11,981 after pre-cleaning)
- Rotter Score 1: R0153710 (1979)
- Rotter Score 2: rotter_score_2 (consolidated 2014/2016/2018)

STEP 1 OBJECTIVES:
1. Parse SAS file (single pass) for rename mappings + format definitions
2. Rename columns from reference numbers to descriptive names
3. Clean missing values (embedded detection: 999, -9, etc.)
4. Classify variables by type using SAS formats + heuristics
5. Convert 2-digit years to 4-digit during classification (0-14→2000s, 15-99→1900s)
6. Save classifications + cleaned data for Step 2 aggregation

DEFERRED TO STEP 2:
- Participant filtering (>25% missing) - after variable aggregation
- Hybrid variable handling - during longitudinal aggregation

================================================================================
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import matplotlib.pyplot as plt

# Suppress all RuntimeWarnings globally (numpy operations on empty/invalid data)
warnings.filterwarnings('ignore', category=RuntimeWarning)

print("=" * 80)
print("STEP 1: VARIABLE CLASSIFICATION PIPELINE")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_month_variable(col_name):
    """Unified month detection used throughout classification."""
    col_lower = col_name.lower()
    month_patterns = [r'~m_', r'~m$', r'~m\b', r'_m_', r'_mo_', r'_m$', r'_mo$', r'\bmonth\b']
    return any(re.search(pattern, col_lower, re.IGNORECASE) for pattern in month_patterns)

# ============================================================================
# LOAD RAW DATA & STANDARDIZE MISSING VALUES
# ============================================================================
print("\n[1.1] Loading raw data")
SOURCE_FILE = "nlsy79_cleaned_4_new.parquet"
if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

df_raw = pd.read_parquet(SOURCE_FILE)
print(f"  Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ============================================================================
# PARSE SAS FILE (RENAME + CLASSIFY) - SINGLE READ
# ============================================================================
print("\n[1.2] Parsing SAS file for rename mappings and type classifications...")

def parse_sas_file_complete(sas_file_path, variables_of_interest):
    """
    Parse SAS file to extract variable type classifications based on format definitions.
    Returns dict mapping variable_name -> {'format_code': str, 'value_mappings': list, 'inferred_type': str}
    """
    rename_map = {}
    format_definitions = {}
    format_assignments = {}
    
    print("  Parsing line-by-line (optimized for large files)...")
    
    # Patterns for SAS syntax
    # A0003000 = 'VERSION_R30_2022'n  (rename statement with tick-n)
    rename_pattern = re.compile(r"([A-Z]\d+)\s*=\s*'([^']+)'n")
    # format A0003000 vx0f.;
    format_assign_pattern = re.compile(r'format\s+([A-Z]\d+)\s+(\w+)\.')
    # value vx0f (format definition with value mappings)
    value_start_pattern = re.compile(r'value\s+(\w+)', re.IGNORECASE)
    value_mapping_pattern = re.compile(r"(-?\d+(?:-\d+)?|\.)='([^']*)'")
    
    current_format = None
    current_mappings = []
    in_format_block = False
    
    with open(sas_file_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # RENAME mappings (tick-n format: A0003000 = 'VERSION_R30_2022'n)
            for match in rename_pattern.finditer(line):
                rename_map[match.group(1)] = match.group(2)
            
            # FORMAT assignments
            for match in format_assign_pattern.finditer(line):
                format_assignments[match.group(1)] = match.group(2)
            
            # PROC FORMAT blocks
            value_start = value_start_pattern.search(line)
            if value_start:
                # Save previous format if exists
                if current_format:
                    format_definitions[current_format] = current_mappings
                current_format = value_start.group(1)
                current_mappings = []
                in_format_block = True
            
            if in_format_block:
                # Extract value mappings from current line
                for match in value_mapping_pattern.finditer(line):
                    current_mappings.append((match.group(1), match.group(2)))
                
                # Check for end of format block
                if ';' in line and current_format:
                    format_definitions[current_format] = current_mappings
                    current_format = None
                    current_mappings = []
                    in_format_block = False
            
            # Progress indicator every 100k lines
            if line_num % 100000 == 0:
                print(f"    Processed {line_num:,} lines...")
    
    print(f"    Found {len(rename_map):,} rename mappings")
    print(f"    Found {len(format_definitions):,} format definitions")
    print(f"    Found {len(format_assignments):,} format assignments")
    
    print("  Building classifications for DataFrame variables...")
    # Combine all information
    classifications = {}
    for ref_num, var_name in rename_map.items():
        # Check if this REFERENCE NUMBER is in the DataFrame (before renaming)
        if ref_num not in variables_of_interest:
            continue
            
        format_code = format_assignments.get(ref_num)
        value_mappings = format_definitions.get(format_code, []) if format_code else []
        
        # Infer type from value mappings using the descriptive name
        inferred_type = infer_type_from_sas_format(var_name, value_mappings)
        
        # Store with original name as key (matches DataFrame columns after renaming)
        classifications[var_name] = {
            'reference_number': ref_num,
            'format_code': format_code,
            'value_mappings': value_mappings,
            'inferred_type': inferred_type
        }
    
    print(f"    Matched {len(classifications):,} SAS variables to DataFrame columns")
    return rename_map, classifications

def infer_type_from_sas_format(var_name, value_mappings):
    """
    Infer variable type from SAS format value mappings.
    Implements the classification logic from the R workflow documentation.
    """
    var_lower = var_name.lower()
    
    # Exclude month variables from year detection
    is_month_var = is_month_variable(var_name)
    
    # Year check (highest priority)
    year_patterns = [r'year', r'date', r'yr\d+', r'_y$', r'_yy$', r'cal_year', r'recip_fill', r'birth']
    if not is_month_var and any(re.search(pattern, var_lower) for pattern in year_patterns):
        # Check if values look like years
        if len(value_mappings) > 0:
            # Extract numeric values
            numeric_vals = []
            for val_spec, label in value_mappings:
                if val_spec != '.' and '-' not in val_spec:
                    try:
                        numeric_vals.append(int(val_spec))
                    except:
                        pass
            
            if len(numeric_vals) > 0:
                avg_val = sum(numeric_vals) / len(numeric_vals)
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                
                # 4-digit years in realistic range
                if 1957 <= avg_val <= 2025:
                    return 'year'
                
                # 2-digit years (0-99) - will be converted: 0-14->2000s, 15-99->1900s
                # Check if all values are 0-99 and variable name suggests year
                if 0 <= min_val <= 99 and 0 <= max_val <= 99:
                    if any(re.search(p, var_lower) for p in [r'year', r'_y', r'birth', r'date']):
                        return 'year'
    
    # No value mappings = continuous
    if len(value_mappings) == 0:
        return 'continuous'
    
    # Count distinct categories
    n_categories = len(value_mappings)
    
    # Binary check
    if n_categories == 2:
        labels = [label.lower() for _, label in value_mappings]
        all_labels = ' '.join(labels)
        
        binary_pairs = [
            ('yes', 'no'), ('true', 'false'), ('male', 'female'),
            ('0', '1'), ('positive', 'negative'), ('present', 'absent')
        ]
        
        for pair in binary_pairs:
            if pair[0] in all_labels and pair[1] in all_labels:
                return 'binary'
        
        # Check actual values (0/1 or 1/2 is often binary)
        vals = [val_spec for val_spec, _ in value_mappings if val_spec != '.']
        if set(vals) == {'0', '1'}:
            return 'binary'
        # 1/2 coding is binary if variable name suggests yes/no question
        if set(vals) == {'1', '2'}:
            # Q-prefixed variables with 2 categories are usually yes/no
            if var_lower.startswith('q') or 'item' in var_lower:
                return 'binary'
    
    # Continuous if >20 categories
    if n_categories > 20:
        # Exception: occupation codes are nominal even with >20 categories
        if 'occ' in var_lower or 'cpsocc' in var_lower or 'industry' in var_lower or 'soc' in var_lower:
            return 'categorical_nominal'
        return 'continuous'
    
    # Check for range patterns (indicates continuous)
    for val_spec, label in value_mappings:
        if '-' in val_spec and val_spec != '.':
            return 'continuous'
    
    # Between 2-20 categories: check for ordinal patterns
    if 2 <= n_categories <= 20:
        labels = [label.lower() for _, label in value_mappings]
        all_labels = ' '.join(labels)
        
        # Check for month names (nominal - months are CYCLICAL, not linear ordinal)
        # December (12) wraps to January (1), so standard ordinal stats are inappropriate
        month_keywords = ['january', 'february', 'march', 'april', 'may', 'june',
                         'july', 'august', 'september', 'october', 'november', 'december']
        if any(month in all_labels for month in month_keywords):
            return 'categorical_nominal'
        
        # Check for month numbers (1-12) or day numbers (1-31)
        if is_month_var:
            vals = []
            for val_spec, _ in value_mappings:
                if val_spec != '.' and '-' not in val_spec:
                    try:
                        vals.append(int(val_spec))
                    except:
                        pass
            # Months (1-12) are CYCLICAL/NOMINAL: Dec->Jan wraps around
            # Days (1-31) can be ordinal within a month context
            if len(vals) > 0:
                if min(vals) == 1 and max(vals) <= 12:
                    return 'categorical_nominal'  # Months are cyclical
                elif min(vals) == 1 and max(vals) <= 31:
                    return 'categorical_ordinal'  # Days have linear order
        
        # Check for occupation/industry codes (nominal)
        if 'occ' in var_lower or 'cpsocc' in var_lower or 'industry' in var_lower or 'soc' in var_lower:
            return 'categorical_nominal'
        
        # Q-prefixed variables with 3-7 categories are likely Likert scales (ordinal)
        if var_lower.startswith('q') and 3 <= n_categories <= 7:
            return 'categorical_ordinal'
        
        # 16 ordinal keyword patterns
        ordinal_patterns = [
            r'strongly (disagree|agree)', r'very (low|high)',
            r'never.*always', r'none.*all', r'poor.*excellent',
            r'first.*last', r'lowest.*highest', r'least.*most',
            r'(not at all|somewhat|very)',
            r'(never|rarely|sometimes|often|always)',
            r'(none|few|some|many|all)',
            r'(disagree|neutral|agree)', r'(low|medium|high)',
            r'(bad|fair|good)', r'(worse|same|better)',
            r'(less|equal|more)', r'grade|level|degree',
            r'health'  # health ratings are typically ordinal
        ]
        
        for pattern in ordinal_patterns:
            if re.search(pattern, all_labels):
                return 'categorical_ordinal'
        
        # No ordinal pattern = nominal
        return 'categorical_nominal'
    
    # Fallback
    return 'continuous'

# Parse SAS file and apply renaming
SAS_FILE = "nlsy79_all_1979-2022.sas"
sas_type_mapping = {}
rename_mappings = {}

if os.path.exists(SAS_FILE):
    try:
        rename_mappings, sas_results = parse_sas_file_complete(SAS_FILE, set(df_raw.columns))
        
        # Apply column renaming
        columns_to_rename = {ref: name for ref, name in rename_mappings.items() if ref in df_raw.columns}
        df_raw = df_raw.rename(columns=columns_to_rename)
        print(f"  Renamed {len(columns_to_rename):,} columns to descriptive names")
        
        # Extract type mappings
        sas_type_mapping = {var: info['inferred_type'] for var, info in sas_results.items()}
        print(f"  Successfully classified {len(sas_type_mapping):,} variables from SAS file")
        
        # Show distribution
        type_counts = pd.Series(sas_type_mapping).value_counts()
        print("  SAS-based classification distribution:")
        for vtype, count in type_counts.items():
            print(f"    {vtype}: {count}")
    except Exception as e:
        print(f"  Warning: Could not parse SAS file: {e}")
        print("  Will fall back to heuristic classification")
else:
    print(f"  SAS file not found: {SAS_FILE}")
    print("  Will use heuristic classification")

# ============================================================================
# DEFINE HELPER FUNCTIONS FOR MISSING VALUE DETECTION & CLASSIFICATION
# ============================================================================

def detect_embedded_missing_codes(series, var_name=""):
    """
    Detect values that are likely missing codes embedded in valid responses.
    Combines two approaches:
    1. Gap detection: if max value > second_largest * 3, it's likely a missing code
    2. Pattern matching: common codes (99, 999, 9999) with context-aware criteria
    """
    # Only process numeric columns with non-null values
    if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
        return []
    
    values = series.dropna()
    if len(values) < 10:  # Need sufficient data
        return []
    
    unique_vals = np.array(sorted(values.unique()))
    if len(unique_vals) < 2:  # Need at least 2 unique values
        return []
    
    missing_codes = []
    
    # Calculate data characteristics
    data_max = unique_vals.max()
    data_range = data_max - unique_vals.min()
    
    # Skip ID-like variables (very large numbers or wide range)
    if data_max > 100000 or data_range > 100000:
        return []
    
    # METHOD 1: Gap detection (3x rule)
    # If largest value > second_largest * 3, it's likely a missing code
    largest = unique_vals[-1]
    second_largest = unique_vals[-2] if len(unique_vals) >= 2 else 0
    
    if second_largest > 0 and largest > second_largest * 3:
        if largest not in missing_codes:
            missing_codes.append(largest)
    
    # METHOD 2: Pattern matching for common missing codes
    # Check for 95, 96, 97, 98, 99, 999, 9999 with principled gap criterion
    common_patterns = [95, 96, 97, 98, 99, 999, 9999]

    for candidate in common_patterns:
        if candidate not in unique_vals or candidate in missing_codes:
            continue

        # Get all other values (excluding already detected codes)
        other_vals = unique_vals[(unique_vals != candidate) & (~np.isin(unique_vals, missing_codes))]
        if len(other_vals) == 0:
            continue

        max_other = other_vals.max()
        gap_from_max_other = candidate - max_other

        # Principled gap: gap must be greater than half the candidate value
        if max_other < candidate and gap_from_max_other > (candidate / 2):
            missing_codes.append(candidate)

    return sorted(missing_codes)

# ============================================================================
# CLEAN MISSING VALUES (BEFORE CLASSIFICATION)
# ============================================================================
print("\n[1.3] Cleaning missing values (embedded detection)...")

# Embedded missing code detection
print("\n[1.3a] Detecting embedded missing codes (999, -9, etc.)...")

# Apply detection to all numeric columns
embedded_missing_report = []
total_values_converted = 0

numeric_cols = [col for col in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[col])]
print(f"  Scanning {len(numeric_cols)} numeric variables...")

skipped_variables = 0  

for idx, col in enumerate(numeric_cols):
    # Progress indicator every 1000 variables
    if (idx + 1) % 1000 == 0:
        print(f"    Processed {idx + 1}/{len(numeric_cols)} variables...")
    
    detected_codes = detect_embedded_missing_codes(df_raw[col], col)
    
    if detected_codes:
        # Count how many values will be converted
        mask = df_raw[col].isin(detected_codes)
        count = mask.sum()
        
        if count > 0:
            # Calculate percentage of non-null values that would be converted
            non_null_count = df_raw[col].notna().sum()
            if non_null_count > 0:
                conversion_pct = count / non_null_count
                
                # Skip conversion if >10% of non-null values would be converted
                if conversion_pct > 0.10:
                    skipped_variables += 1
                    continue
            
            # Convert to NA
            df_raw.loc[mask, col] = np.nan
            total_values_converted += count
            
            # Record for reporting
            embedded_missing_report.append({
                'variable': col,
                'detected_codes': detected_codes,
                'count_converted': count
            })

print(f"  Detected embedded missing codes in {len(embedded_missing_report):,} variables")
print(f"  Converted {total_values_converted:,} embedded missing code values to np.nan")
if skipped_variables > 0:
    print(f"  Skipped {skipped_variables:,} variables where >10% of values would be converted (likely valid data)")

# Save report
if embedded_missing_report:
    report_df = pd.DataFrame(embedded_missing_report)
    report_df.to_csv('step1_embedded_missing_codes.csv', index=False)
    print(f"  Saved: step1_embedded_missing_codes.csv")
    
    # Show examples
    if len(embedded_missing_report) > 0:
        print(f"  Example detections:")
        for i, item in enumerate(embedded_missing_report[:5]):
            print(f"    {item['variable']}: codes {item['detected_codes']} ({item['count_converted']} values)")

# ============================================================================
# CONSERVATIVE OUTLIER FILTER (3xIQR) FOR NUMERIC VARIABLES
# ============================================================================
print("\n[1.3b] Filtering outliers (3×IQR)...")

# Initialize skip trackers for outlier filtering
skipped_vars_2pct = []
skipped_outliers_closer = 0

outlier_report = []
skipped_status_vars = 0
for col in numeric_cols:
    # Skip STATUS_WK_NUM prefix variables
    if col.startswith('STATUS_WK_NUM'):
        skipped_status_vars += 1
        continue
    
    series = df_raw[col]
    non_null = series.dropna()
    if len(non_null) < 10:
        continue  # Skip columns with too few values
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue  # Skip columns with no spread
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outlier_mask = (series < lower) | (series > upper)
    outlier_values = series[outlier_mask & series.notna()]
    outlier_indices = series.index[outlier_mask & series.notna()].tolist()
    outlier_count = len(outlier_indices)
    if outlier_count == 0:
        continue

    # 1% identical value rule - selective outlier filtering
    value_counts = outlier_values.value_counts()
    mean_val = non_null.mean()
    
    # Find all values that meet the 1% rule (more than 1%)
    frequent_outlier_values = value_counts[value_counts / outlier_count > 0.01].index.tolist()
    
    if len(frequent_outlier_values) > 0:
        # Create mask for outliers to KEEP (not convert to NA):
        # Rule 1: Frequent outliers (>1% identical)
        # Rule 2: Outliers closer to mean than ANY skipped outlier
        outliers_to_keep = set()
        skip_due_to_closer = 0
        
        # First pass: identify all frequent outliers and their distances
        frequent_outlier_distances = set()
        for idx in outlier_indices:
            val = series.loc[idx]
            if val in frequent_outlier_values:
                outliers_to_keep.add(idx)
                frequent_outlier_distances.add(abs(val - mean_val))
        
        # Second pass: skip any outlier closer to mean than ANY skipped outlier
        if len(frequent_outlier_distances) > 0:
            # Find the closest skipped outlier distance
            min_skipped_dist = min(frequent_outlier_distances)
            
            for idx in outlier_indices:
                if idx not in outliers_to_keep:
                    val = series.loc[idx]
                    if abs(val - mean_val) < min_skipped_dist:
                        outliers_to_keep.add(idx)
                        skip_due_to_closer += 1
        
        # Convert only outliers that are NOT kept
        outliers_to_convert = [idx for idx in outlier_indices if idx not in outliers_to_keep]
        
        if len(outliers_to_convert) > 0:
            df_raw.loc[outliers_to_convert, col] = np.nan
            outlier_report.append({
                'variable': col,
                'outlier_count': len(outliers_to_convert),
                'outlier_indices': outliers_to_convert[:5]
            })
        
        # Track skipped outliers
        if len(frequent_outlier_values) > 0:
            skipped_vars_2pct.append(col)
            skipped_outliers_closer += skip_due_to_closer
    else:
        # No frequent outliers - convert all detected outliers to NA
        df_raw.loc[outlier_mask, col] = np.nan
    outlier_report.append({
        'variable': col,
        'outlier_count': outlier_count,
        'outlier_indices': outlier_indices[:5]  # Show up to 5 indices for preview
    })

# Print summary of skips
if 'skipped_vars_2pct' not in locals():
    skipped_vars_2pct = []
if 'skipped_outliers_closer' not in locals():
    skipped_outliers_closer = 0

if skipped_status_vars > 0:
    print(f"  Skipped {skipped_status_vars} STATUS_WK_NUM variables from outlier detection")

if outlier_report:
    total_outlier_values = sum(item['outlier_count'] for item in outlier_report)
    total_cells = df_raw.shape[0] * df_raw.shape[1]
    outlier_pct = (total_outlier_values / total_cells) * 100
    print(f"  Converted {total_outlier_values:,} outlier values to NA in {len(outlier_report)} variables (3×IQR rule)")
    print(f"  Percentage of total dataset: {outlier_pct:.4f}% ({total_outlier_values:,} / {total_cells:,} cells)")
    for item in outlier_report[:5]:
        print(f"    {item['variable']}: {item['outlier_count']} outliers (indices: {item['outlier_indices']})")
    # Save summary CSV
    outlier_summary_df = pd.DataFrame([
        {
            'variable': item['variable'],
            'outlier_count': item['outlier_count'],
            'example_indices': str(item['outlier_indices'])
        }
        for item in outlier_report
    ])
    outlier_summary_df.to_csv('step1_outlier_conversions.csv', index=False)
    print("  Saved: step1_outlier_conversions.csv")
if skipped_vars_2pct:
    print(f"  Skipped {len(skipped_vars_2pct)} variables due to 2% identical outlier rule.")
else:
    print("  No extreme outliers detected by 5×IQR rule.")

# ============================================================================
# CLASSIFY VARIABLES (SAS + HEURISTIC FALLBACK)
# ============================================================================
print("\n[1.4] Classifying variables by type...")

def classify_variable(col, series, df_ref):
    """
    Classify a variable using comprehensive heuristics derived from SAS format analysis.
    Also converts 2-digit years to 4-digit years immediately upon detection.
    """
    dtype = series.dtype
    col_lower = col.lower()
    
    # ========== STEP 1: YEAR DETECTION (HIGHEST PRIORITY) ==========
    # Exclude month variables from year detection
    is_month_var = is_month_variable(col)
    
    year_name_patterns = [r'year', r'date', r'yr\d+', r'_y$', r'_yy$', r'birth']
    name_suggests_year = any(re.search(pattern, col_lower) for pattern in year_name_patterns)
    
    if not is_month_var and name_suggests_year and dtype.kind in ['i', 'u', 'f']:
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_vals = non_null.unique()
            if len(unique_vals) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    min_val = non_null.min()
                    max_val = non_null.max()
                    avg_val = non_null.mean()
                
                # 4-digit years in realistic range
                if 1957 <= avg_val <= 2025:
                    if all(v >= 1900 and v <= 2100 for v in unique_vals[:100] if pd.notna(v)):
                        return 'year'
                
                # 2-digit years (0-99) - CONVERT IMMEDIATELY
                if 0 <= min_val <= 99 and 0 <= max_val <= 99:
                    if any(re.search(p, col_lower) for p in [r'year', r'_y', r'birth', r'date']):
                        # Convert 2-digit years: 0-14 -> 2000-2014, 15-99 -> 1915-1999 (vectorized)
                        mask_young = (df_ref[col] >= 0) & (df_ref[col] <= 14)
                        mask_old = (df_ref[col] >= 15) & (df_ref[col] <= 99)
                        df_ref.loc[mask_young, col] += 2000
                        df_ref.loc[mask_old, col] += 1900
                        return 'year'
    
    # ========== STEP 2: BINARY DETECTION ==========
    unique_values = series.dropna().unique()
    n_unique = len(unique_values)
    
    if n_unique == 2:
        vals_sorted = sorted([str(v).lower() for v in unique_values])
        binary_pairs = [
            ['0', '1'], ['0.0', '1.0'],
            ['false', 'true'], ['no', 'yes'],
            ['f', 't'], ['female', 'male'], ['m', 'f']
        ]
        if vals_sorted in binary_pairs or set(vals_sorted) in [set(p) for p in binary_pairs]:
            return 'binary'
        
        # Check numeric 0/1
        try:
            numeric_vals = sorted([float(v) for v in unique_values])
            if numeric_vals == [0.0, 1.0]:
                return 'binary'
        except:
            pass
    
    # ========== STEP 3: CONTINUOUS DETECTION ==========
    if dtype.kind in ['i', 'u', 'f']:
        if n_unique > 20:
            # Exception: occupation/industry codes are nominal
            if 'occ' in col_lower or 'cpsocc' in col_lower or 'industry' in col_lower or 'soc' in col_lower:
                return 'categorical_nominal'
            return 'continuous'
    
    # ========== STEP 4: CATEGORICAL (ORDINAL VS NOMINAL) ==========
    if 2 <= n_unique <= 20:
        # Check for month numbers (1-12) or day numbers (1-31)
        if is_month_var:
            try:
                numeric_vals = sorted([int(v) for v in unique_values])
                # Months (1-12) are CYCLICAL: treat as nominal
                # December (12) wraps to January (1) - not truly "greater than"
                if min(numeric_vals) == 1 and max(numeric_vals) <= 12:
                    return 'categorical_nominal'  # Cyclical data
                # Days (1-31) can be ordinal within a month
                elif min(numeric_vals) == 1 and max(numeric_vals) <= 31:
                    return 'categorical_ordinal'
            except:
                pass
        
        # Check for occupation/industry codes (nominal)
        if 'occ' in col_lower or 'cpsocc' in col_lower or 'industry' in col_lower or 'soc' in col_lower:
            return 'categorical_nominal'
        
        # Q-prefixed variables with 3-7 unique values are likely Likert scales
        if col_lower.startswith('q') and 3 <= n_unique <= 7:
            return 'categorical_ordinal'
        
        if dtype == 'object' or dtype.name.startswith('string'):
            str_vals = [str(v).lower() for v in unique_values if pd.notna(v)]
            all_text = ' '.join(str_vals)
            
            ordinal_keywords = [
                'strongly', 'very', 'somewhat', 'slightly',
                'never', 'rarely', 'sometimes', 'often', 'always',
                'none', 'few', 'some', 'many', 'all',
                'low', 'medium', 'high', 'lowest', 'highest',
                'poor', 'fair', 'good', 'excellent',
                'disagree', 'neutral', 'agree',
                'less', 'same', 'more', 'worse', 'better',
                'first', 'second', 'third', 'last',
                'grade', 'level', 'degree', 'health'
            ]
            
            if any(keyword in all_text for keyword in ordinal_keywords):
                return 'categorical_ordinal'
            
            return 'categorical_nominal'
        else:
            return 'categorical_ordinal'
    
    # ========== FALLBACK ==========
    return 'continuous'

def validate_sas_classification(col, series, sas_type):
    """
    Validate SAS classification against actual data patterns.
    Returns corrected type if SAS classification conflicts with data reality.
    """
    dtype = series.dtype
    col_lower = col.lower()
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return sas_type
    
    unique_vals = non_null.unique()
    n_unique = len(unique_vals)
    
    # VALIDATION 1: Check if "continuous" is actually binary (0/1 or 1/2)
    if sas_type == 'continuous' and n_unique == 2:
        try:
            numeric_vals = sorted([float(v) for v in unique_vals])
            # 0/1 is definitely binary
            if numeric_vals == [0.0, 1.0]:
                return 'binary'
            # 1/2 is binary for question/item variables
            if numeric_vals == [1.0, 2.0] and (col_lower.startswith('q') or 'item' in col_lower):
                return 'binary'
        except:
            pass
    
    # VALIDATION 2: Check if "continuous" is actually year (4-digit years in range)
    if sas_type == 'continuous' and dtype.kind in ['i', 'u', 'f']:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                numeric_vals = [float(v) for v in unique_vals]
                if len(numeric_vals) > 0:
                    avg_val = sum(numeric_vals) / len(numeric_vals)
                # If average is in year range and all values are 4-digit years
                if 1957 <= avg_val <= 2025:
                    if all(1900 <= v <= 2100 for v in numeric_vals[:50]):
                        # Additional check: if values look like survey years
                        if n_unique <= 30 and all(v >= 1979 for v in numeric_vals):
                            return 'year'
        except:
            pass
    
    # VALIDATION 3: Check if "continuous" is actually nominal (occupation/ID codes)
    if sas_type == 'continuous':
        # ID variables should not be continuous (regardless of unique count)
        if 'caseid' in col_lower or 'case_id' in col_lower or col_lower == 'id' or col_lower.endswith('_id'):
            return 'categorical_nominal'
        
        # For variables with >20 unique values
        if n_unique > 20:
            # Occupation codes
            if 'occ' in col_lower or 'cpsocc' in col_lower or 'industry' in col_lower or 'soc' in col_lower:
                return 'categorical_nominal'
    
    # VALIDATION 4: Check if "categorical_nominal" with 2 values should be binary
    if sas_type == 'categorical_nominal' and n_unique == 2:
        try:
            numeric_vals = sorted([float(v) for v in unique_vals])
            if numeric_vals == [0.0, 1.0] or numeric_vals == [1.0, 2.0]:
                return 'binary'
        except:
            pass
    
    # VALIDATION 5: Likert scales (3-7 categories) should be ordinal, not nominal
    if sas_type == 'categorical_nominal' and 3 <= n_unique <= 7:
        if col_lower.startswith('q'):  # Question variables
            return 'categorical_ordinal'
    
    # VALIDATION 6: Detect hybrid encoding (categorical codes + continuous IDs)
    # Common pattern: 0-10 are status codes, 100+ are job/round identifiers
    # NOTE: STATUS_WK_NUM is handled separately via recoding, so skip it here
    if sas_type == 'continuous' and dtype.kind in ['i', 'u', 'f'] and n_unique >= 10:
        if 'STATUS_WK_NUM' not in col_lower:  # Skip STATUS_WK_NUM - handled separately
            try:
                numeric_vals = [float(v) for v in unique_vals if pd.notna(v)]
                if len(numeric_vals) > 0:
                    # Check if values cluster in two distinct ranges
                    low_vals = [v for v in numeric_vals if v < 100]
                    high_vals = [v for v in numeric_vals if v >= 100]
                    
                    # Hybrid if: have both low (< 100) and high (>= 100) values
                    # AND low values look categorical (few unique, small integers)
                    if len(low_vals) > 0 and len(high_vals) > 0:
                        n_low_unique = len(set(low_vals))
                        # If low values are categorical-like (< 20 unique values, all < 100)
                        if n_low_unique < 20 and all(v < 100 for v in low_vals):
                            # This is a hybrid - keep as continuous but flag it
                            return 'continuous_hybrid'
            except:
                pass
    
    return sas_type

# Special case overrides (hard-coded classifications that bypass SAS/heuristics)
# rotter_score_2 is not in SAS file (only ROTTER_SCORE_YYYY versions exist)
# STATUS_WK_NUM recoded (100+ → 9) so now categorical_nominal
# CPSIND variables are industry codes (categorical_nominal despite hybrid encoding)
SPECIAL_CASES = {
    'rotter_score_2': 'continuous',
    # R_REL variables are hardcoded as categorical_nominal due to inconsistent
    # survey instrument changes across timepoints (1979: 0-300 range, later years: 1-10 scales)
    'R_REL-1_COL_1979': 'categorical_nominal',
    'R_REL-1_1979': 'categorical_nominal',
    'R_REL-2_COL_1979': 'categorical_nominal',
    'R_REL-2_1979': 'categorical_nominal',
    'R_REL-2_1982': 'categorical_nominal',
    'R_REL-1_2000': 'categorical_nominal',
    'R_REL-2-2000': 'categorical_nominal',
    'R_REL-2_2012': 'categorical_nominal',
}

print(f"  Classifying all {len(df_raw.columns)} variables...")
print("  (Year conversion happens automatically during classification)")

# Classify all variables with validation
variable_types = {}
sas_corrected = 0
converted_years = 0

correction_methods = {}

for col in df_raw.columns:
    col_before = df_raw[col].copy()
    
    # Check for special case overrides first
    if col in SPECIAL_CASES:
        variable_types[col] = SPECIAL_CASES[col]
    # STATUS_WK_NUM variables are categorical_nominal (after recoding)
    elif 'STATUS_WK_NUM' in col.upper():
        variable_types[col] = 'categorical_nominal'
    # CPSIND variables are industry codes (categorical_nominal)
    elif col.upper().startswith('CPSIND70') or col.upper().startswith('CPSIND80'):
        variable_types[col] = 'categorical_nominal'
    # Use SAS classification if available, but validate against actual data
    elif col in sas_type_mapping:
        sas_type = sas_type_mapping[col]
        validated_type = validate_sas_classification(col, df_raw[col], sas_type)
        variable_types[col] = validated_type
        if validated_type != sas_type:
            sas_corrected += 1
            # Track which validation method made the correction
            correction_key = f"{sas_type} → {validated_type}"
            correction_methods[correction_key] = correction_methods.get(correction_key, 0) + 1
        # Track if year was converted (check if values changed)
        if validated_type == 'year' and not col_before.equals(df_raw[col]):
            converted_years += 1
    else:
        variable_types[col] = classify_variable(col, df_raw[col], df_raw)
        # Track if year was converted (check if values changed)
        if variable_types[col] == 'year' and not col_before.equals(df_raw[col]):
            converted_years += 1

if sas_corrected > 0:
    print(f"  Corrected {sas_corrected} SAS classifications based on actual data patterns")
    print(f"\n  Correction breakdown by validation method:")
    for correction_type, count in sorted(correction_methods.items(), key=lambda x: x[1], reverse=True):
        print(f"    {correction_type}: {count}")

if converted_years > 0:
    print(f"\n  Converted {converted_years} 2-digit year variables: 0-14 -> 2000-2014, 15-99 -> 1915-1999")

# Show distribution (combine year + continuous)
type_counts_raw = pd.Series(variable_types).value_counts()

# Combine year and continuous into single continuous category
type_counts_display = type_counts_raw.copy()
if 'year' in type_counts_display.index:
    year_count = type_counts_display['year']
    continuous_count = type_counts_display.get('continuous', 0)
    type_counts_display['continuous'] = continuous_count + year_count
    type_counts_display = type_counts_display.drop('year')

print("\n  Final classification distribution:")
for vtype, count in type_counts_display.items():
    print(f"    {vtype}: {count}")

# Report hybrid variables
hybrid_vars = [col for col, vtype in variable_types.items() if vtype == 'continuous_hybrid']
if hybrid_vars:
    print(f"\n  ⚠️  Detected {len(hybrid_vars)} hybrid encoding variables:")
    print(f"     (mix categorical status codes with continuous IDs)")
    print(f"     These will be treated as continuous but may need special handling in ML.")
    print(f"     NOTE: These variables are NOT automatically handled in downstream steps.")
    print(f"           They remain flagged for awareness but could be manually assessed and split")
    print(f"           into separate categorical and continuous features if needed.")
    if len(hybrid_vars) <= 5:
        for var in hybrid_vars:
            print(f"       - {var}")
    else:
        print(f"       First 5: {', '.join(hybrid_vars[:5])}")
        print(f"       (See variable_classifications.csv and step1_hybrid_variables.csv for details)")
    
    # Generate detailed hybrid variable report
    print(f"     Generating detailed hybrid variable report...")
    hybrid_report = []
    for var in hybrid_vars:
        series = df_raw[var].dropna()
        if len(series) > 0:
            min_val = series.min()
            max_val = series.max()
            low_count = (series < 100).sum()
            high_count = (series >= 100).sum()
            total_count = len(series)
            pct_low = (low_count / total_count * 100) if total_count > 0 else 0
            unique_vals = sorted(series.unique())
            
            hybrid_report.append({
                'variable_name': var,
                'min_value': min_val,
                'max_value': max_val,
                'count_low_values': low_count,
                'count_high_values': high_count,
                'pct_low_values': pct_low,
                'n_unique_values': len(unique_vals),
                'unique_values': str(unique_vals)  # Store as string for CSV
            })
    
    if hybrid_report:
        hybrid_df = pd.DataFrame(hybrid_report)
        hybrid_df.to_csv('step1_hybrid_variables.csv', index=False)
        print(f"     Saved: step1_hybrid_variables.csv ({len(hybrid_report)} variables)")

# ============================================================================
# SAVE CLASSIFICATIONS
# ============================================================================
print("\n[1.5] Saving classifications and processed data...")

# Normalize hybrid types to continuous for downstream processing
# But keep a separate column marking them as hybrid
classifications_df = pd.DataFrame([
    {
        'variable_name': col, 
        'variable_type': variable_types[col].replace('continuous_hybrid', 'continuous'),
        'is_hybrid': variable_types[col] == 'continuous_hybrid'
    }
    for col in df_raw.columns
])
classifications_df.to_csv('variable_classifications.csv', index=False)
print(f"  Saved: variable_classifications.csv ({len(classifications_df)} variables)")

# Save cleaned and renamed data
df_raw.to_parquet('nlsy79_classified.parquet', index=True)
print(f"  Saved: nlsy79_classified.parquet ({df_raw.shape[0]} × {df_raw.shape[1]})")

# Final summary with combined year+continuous
final_type_counts = pd.Series(variable_types).value_counts()
if 'year' in final_type_counts.index:
    year_count = final_type_counts['year']
    continuous_count = final_type_counts.get('continuous', 0)
    final_type_counts['continuous'] = continuous_count + year_count
    final_type_counts = final_type_counts.drop('year')


# Print final classification distribution at the end
print("\nFinal classification distribution (after all corrections):")
for vtype, count in final_type_counts.items():
    print(f"  {vtype}: {count}")

print("\n" + "=" * 80)
print("CLASSIFICATION COMPLETE")
print("=" * 80)
print(f"\nTotal variables classified: {len(variable_types)}")
print(f"SAS-based: {len(sas_type_mapping)}")
print(f"Heuristic-based: {len(variable_types) - len(sas_type_mapping)}")
print("\nNext step: Run step2_trajectory_and_ml.py")

