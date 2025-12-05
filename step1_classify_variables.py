import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

print("=" * 80)
print("STEP 1: VARIABLE CLASSIFICATION PIPELINE")
print("=" * 80)

# ============================================================================
# LOAD RAW DATA & STANDARDIZE MISSING VALUES
# ============================================================================
print("\n[1.1] Loading raw data and standardizing missing values...")
SOURCE_FILE = "nlsy79_cleaned_4_new.parquet"
if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

df_raw = pd.read_parquet(SOURCE_FILE)
print(f"  Loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# Standardize missing value representations
missing_values = [
    '<NA>', 'NA', 'N/A', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'None', 'none',
    '', ' ', '  ', '\t', '\n',
    '.', '..', '...', 
    'missing', 'Missing', 'MISSING',
    'unknown', 'Unknown', 'UNKNOWN',
    'na', 'Na', 'n.a.', 'N.A.',
    '-999', '-99', '-9', '-1',
    '999', '99', '9999',
    'refused', 'Refused', 'REFUSED',
    "don't know", "Don't know", "DON'T KNOW",
    'not applicable', 'Not applicable', 'NOT APPLICABLE',
    'invalid', 'Invalid', 'INVALID',
    'skip', 'Skip', 'SKIP',
    'DK', 'dk', 'RF', 'rf', 'IAP', 'iap'
]
na_replacements = 0
for col in df_raw.columns:
    if df_raw[col].dtype == 'object' or df_raw[col].dtype.name.startswith('string'):
        mask = df_raw[col].isin(missing_values) | df_raw[col].isna()
        na_replacements += mask.sum() - df_raw[col].isna().sum()
        df_raw[col] = df_raw[col].where(~df_raw[col].isin(missing_values), np.nan)

print(f"  Standardized {na_replacements:,} non-standard missing values to np.nan")

# ============================================================================
# FILTER PARTICIPANTS WITH >25% MISSING DATA
# ============================================================================
print("\n[1.1b] Filtering participants with >25% missing data...")

# Calculate missingness per participant
initial_rows = len(df_raw)
initial_cols = df_raw.shape[1]
missingness_per_participant = df_raw.isnull().sum(axis=1) / initial_cols
participants_to_drop = missingness_per_participant > 0.25

# Save dropped participant IDs and missingness info BEFORE filtering
dropped_indices = df_raw.index[participants_to_drop].tolist()
dropped_info = pd.DataFrame({
    'participant_index': dropped_indices,
    'missing_percentage': (missingness_per_participant[participants_to_drop] * 100).round(2)
})

# Drop participants
df_raw = df_raw[~participants_to_drop]
dropped_count = participants_to_drop.sum()
print(f"  Removed {dropped_count:,} participants with >25% missing data")
print(f"  Retained {len(df_raw):,} participants ({len(df_raw)/initial_rows*100:.1f}%)")
dropped_info.to_csv('step1_dropped_participants_25pct.csv', index=False)
print(f"  Saved: step1_dropped_participants_25pct.csv ({len(dropped_info)} rows)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Participant Missingness Analysis (Step 1)', fontsize=16, fontweight='bold')

# Plot 1: Distribution of missingness (all participants)
ax1 = axes[0, 0]
ax1.hist(missingness_per_participant * 100, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(25, color='red', linestyle='--', linewidth=2, label='25% threshold')
ax1.set_xlabel('Missing Data (%)', fontsize=11)
ax1.set_ylabel('Number of Participants', fontsize=11)
ax1.set_title(f'Distribution of Missingness\n(Before Filtering: n={initial_rows:,})', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Retained vs Dropped
ax2 = axes[0, 1]
sizes = [len(df_raw), dropped_count]
labels = [f'Retained\n{len(df_raw):,}\n({len(df_raw)/initial_rows*100:.1f}%)', 
          f'Dropped\n{dropped_count:,}\n({dropped_count/initial_rows*100:.1f}%)']
colors = ['#2ecc71', '#e74c3c']
ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90, textprops={'fontsize': 11})
ax2.set_title('Participants Retained vs Dropped', fontsize=12)

# Plot 3: Distribution after filtering
ax3 = axes[1, 0]
retained_missingness = missingness_per_participant[~participants_to_drop]
ax3.hist(retained_missingness * 100, bins=50, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('Missing Data (%)', fontsize=11)
ax3.set_ylabel('Number of Participants', fontsize=11)
ax3.set_title(f'Missingness Distribution After Filtering\n(n={len(df_raw):,})', fontsize=12)
ax3.grid(alpha=0.3)

# Plot 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
MISSINGNESS SUMMARY

Before Filtering:
  • Total participants: {initial_rows:,}
  • Mean missingness: {missingness_per_participant.mean()*100:.2f}%
  • Median missingness: {missingness_per_participant.median()*100:.2f}%
  • Max missingness: {missingness_per_participant.max()*100:.2f}%

After Filtering:
  • Total participants: {len(df_raw):,}
  • Mean missingness: {retained_missingness.mean()*100:.2f}%
  • Median missingness: {retained_missingness.median()*100:.2f}%
  • Max missingness: {retained_missingness.max()*100:.2f}%

Dropped:
  • Count: {dropped_count:,}
  • Percentage: {dropped_count/initial_rows*100:.2f}%
"""
ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('step1_missingness_filtering.png', dpi=300, bbox_inches='tight')
print(f"  Saved: step1_missingness_filtering.png")
plt.close()

# ============================================================================
# RENAME COLUMNS FROM REFERENCE NUMBERS TO DESCRIPTIVE NAMES
# ============================================================================
print("\n[1.2] Renaming columns using SAS file mappings...")

def extract_rename_mappings(sas_file_path):
    """Extract reference_number -> descriptive_name mappings from SAS file."""
    rename_map = {}
    rename_pattern = re.compile(r"([A-Z]\d+)\s*=\s*'([^']+)'n")
    
    print("  Reading SAS file (line-by-line)...")
    with open(sas_file_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            for match in rename_pattern.finditer(line):
                ref_num = match.group(1)
                var_name = match.group(2)
                # Keep original name with special characters
                rename_map[ref_num] = var_name
            
            if line_num % 100000 == 0:
                print(f"    Processed {line_num:,} lines...")
    
    return rename_map

SAS_FILE = "nlsy79_all_1979-2022.sas"
if os.path.exists(SAS_FILE):
    rename_mappings = extract_rename_mappings(SAS_FILE)
    print(f"  Extracted {len(rename_mappings):,} variable name mappings")
    
    # Rename columns that exist in both DataFrame and SAS mappings
    columns_to_rename = {ref: name for ref, name in rename_mappings.items() if ref in df_raw.columns}
    df_raw = df_raw.rename(columns=columns_to_rename)
    print(f"  Renamed {len(columns_to_rename):,} columns to descriptive names")
    
    # Check for columns that weren't renamed (not in SAS file)
    unrenamed = [col for col in df_raw.columns if col in df_raw.columns and col not in rename_mappings.values()]
    if len(unrenamed) > 0:
        print(f"  Found {len(unrenamed)} variable(s) not in SAS file, will use original names")
else:
    print(f"  Warning: SAS file not found: {SAS_FILE}")
    print("  Proceeding with original column names")

# ============================================================================
# RECODE HYBRID VARIABLES (BEFORE CLASSIFICATION)
# ============================================================================
print("\n[1.2b] Recoding hybrid encoding variables...")

# STATUS_WK_NUM: Recode 100+ job IDs to category 9 (employed/working)
status_wk_cols = [col for col in df_raw.columns if 'STATUS_WK_NUM' in col.upper()]
if len(status_wk_cols) > 0:
    print(f"  Found {len(status_wk_cols)} STATUS_WK_NUM variables")
    recoded_count = 0
    for col in status_wk_cols:
        # Count values >= 100 before recoding
        high_vals = (df_raw[col] >= 100).sum()
        if high_vals > 0:
            df_raw.loc[df_raw[col] >= 100, col] = 9
            recoded_count += 1
    print(f"  Recoded {recoded_count} STATUS_WK_NUM variables: 100+ -> 9 (employed)")
else:
    print("  No STATUS_WK_NUM variables found")

# ============================================================================
# PARSE SAS FILE FOR AUTHORITATIVE TYPE CLASSIFICATIONS
# ============================================================================
print("\n[1.3] Parsing SAS file for variable type classifications...")

def parse_sas_file(sas_file_path, variables_of_interest):
    """
    Parse SAS file to extract variable type classifications based on format definitions.
    Returns dict mapping variable_name -> {'format_code': str, 'value_mappings': list, 'inferred_type': str}
    """
    rename_map = {}
    format_definitions = {}
    format_assignments = {}
    
    print("  Parsing line-by-line (optimized for large files)...")
    
    # Patterns
    rename_pattern = re.compile(r"([A-Z]\d+)\s*=\s*'([^']+)'n")
    format_assign_pattern = re.compile(r"format\s+([A-Z]\d+)\s+(\w+)\.")
    value_start_pattern = re.compile(r"value\s+(\w+)", re.IGNORECASE)
    value_mapping_pattern = re.compile(r"(-?\d+(?:-\d+)?|\.)='([^']*)'")
    
    current_format = None
    current_mappings = []
    in_format_block = False
    
    with open(sas_file_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # RENAME mappings
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
    
    print("  Combining information and inferring types...")
    # Combine all information
    results = {}
    for ref_num, var_name in rename_map.items():
        # Check if this variable name is in the DataFrame
        if var_name not in variables_of_interest:
            continue
            
        format_code = format_assignments.get(ref_num)
        value_mappings = format_definitions.get(format_code, []) if format_code else []
        
        # Infer type from value mappings using the descriptive name
        inferred_type = infer_type_from_sas_format(var_name, value_mappings)
        
        # Store with original name as key (matches DataFrame columns after renaming)
        results[var_name] = {
            'reference_number': ref_num,
            'format_code': format_code,
            'value_mappings': value_mappings,
            'inferred_type': inferred_type
        }
    
    print(f"    Matched {len(results):,} SAS variables to DataFrame columns")
    return results

def infer_type_from_sas_format(var_name, value_mappings):
    """
    Infer variable type from SAS format value mappings.
    Implements the classification logic from the R workflow documentation.
    """
    var_lower = var_name.lower()
    
    # Exclude month variables from year detection
    month_indicators = [r'~m_', r'~m$', r'~m\b', r'_m_', r'_mo_', r'_m$', r'_mo$', r'\bmonth\b']
    is_month_var = any(re.search(pattern, var_lower, re.IGNORECASE) for pattern in month_indicators)
    
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
                
                # 2-digit years (0-99) - will be converted: 0-14→2000s, 15-99→1900s
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
        # Month patterns: 'month', '_M_', '_MO_', 'stopdate', 'startdate'
        is_month_var = ('month' in var_lower or '_m_' in var_lower or '_mo_' in var_lower or 
                       'stopdate' in var_lower or 'startdate' in var_lower or 
                       var_lower.endswith('_m') or '_m' in var_lower)
        
        if is_month_var:
            vals = []
            for val_spec, _ in value_mappings:
                if val_spec != '.' and '-' not in val_spec:
                    try:
                        vals.append(int(val_spec))
                    except:
                        pass
            # Months (1-12) are CYCLICAL/NOMINAL: Dec→Jan wraps around
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

# Check if SAS file exists
sas_type_mapping = {}

if os.path.exists(SAS_FILE):
    try:
        sas_results = parse_sas_file(SAS_FILE, set(df_raw.columns))
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
# CLASSIFY VARIABLES (SAS + HEURISTIC FALLBACK)
# ============================================================================
print("\n[1.4] Classifying variables by type...")

def classify_variable(col, series):
    """
    Classify a variable using comprehensive heuristics derived from SAS format analysis.
    """
    dtype = series.dtype
    col_lower = col.lower()
    
    # ========== STEP 1: YEAR DETECTION (HIGHEST PRIORITY) ==========
    # Exclude month variables from year detection
    month_indicators = [r'~m_', r'~m$', r'~m\b', r'_m_', r'_mo_', r'_m$', r'_mo$', r'\bmonth\b']
    is_month_var = any(re.search(pattern, col_lower, re.IGNORECASE) for pattern in month_indicators)
    
    year_name_patterns = [r'year', r'date', r'yr\d+', r'_y$', r'_yy$', r'birth']
    name_suggests_year = any(re.search(pattern, col_lower) for pattern in year_name_patterns)
    
    if not is_month_var and name_suggests_year and dtype.kind in ['i', 'u', 'f']:
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_vals = non_null.unique()
            if len(unique_vals) > 0:
                min_val = non_null.min()
                max_val = non_null.max()
                avg_val = non_null.mean()
                
                # 4-digit years in realistic range
                if 1957 <= avg_val <= 2025:
                    if all(v >= 1900 and v <= 2100 for v in unique_vals[:100] if pd.notna(v)):
                        return 'year'
                
                # 2-digit years (0-99) - will be converted: 0-14→2000s, 15-99→1900s
                if 0 <= min_val <= 99 and 0 <= max_val <= 99:
                    if any(re.search(p, col_lower) for p in [r'year', r'_y', r'birth', r'date']):
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
        # Month patterns: 'month', '_M_', '_MO_', 'stopdate', 'startdate'
        is_month_var = ('month' in col_lower or '_m_' in col_lower or '_mo_' in col_lower or 
                       'stopdate' in col_lower or 'startdate' in col_lower or 
                       col_lower.endswith('_m') or '_m' in col_lower)
        
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
# CASEID is now handled by validation logic checking for 'caseid' in variable name
# STATUS_WK_NUM recoded (100+ → 9) so now categorical_nominal
# CPSIND variables are industry codes (categorical_nominal despite hybrid encoding)
SPECIAL_CASES = {
    'rotter_score_2': 'continuous',
}

# Classify all variables with validation
variable_types = {}
sas_corrected = 0

for col in df_raw.columns:
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
    else:
        variable_types[col] = classify_variable(col, df_raw[col])

if sas_corrected > 0:
    print(f"  Corrected {sas_corrected} SAS classifications based on actual data patterns")

# Show distribution
type_counts = pd.Series(variable_types).value_counts()
print("\n  Final classification distribution:")
for vtype, count in type_counts.items():
    print(f"    {vtype}: {count}")

# ============================================================================
# CONVERT 2-DIGIT YEARS TO 4-DIGIT YEARS
# ============================================================================
print("\n[1.4b] Converting 2-digit years to 4-digit years...")

year_vars = [col for col, vtype in variable_types.items() if vtype == 'year']
converted_count = 0
skipped_month_vars = 0

for col in year_vars:
    col_lower = col.lower()
    
    # Skip month variables - they should NOT be converted
    month_indicators = [r'~m_', r'~m$', r'~m\b', r'_m_', r'_mo_', r'_m$', r'_mo$', r'\bmonth\b']
    is_month_var = any(re.search(pattern, col_lower, re.IGNORECASE) for pattern in month_indicators)
    
    if is_month_var:
        skipped_month_vars += 1
        continue
    
    # Check if this variable has 2-digit year values (0-99)
    non_na_vals = df_raw[col].dropna()
    if len(non_na_vals) > 0:
        min_val = non_na_vals.min()
        max_val = non_na_vals.max()
        
        # If all values are 0-99, convert based on cutoff:
        # 0-14 → 2000-2014 (21st century)
        # 15-99 → 1915-1999 (20th century)
        if 0 <= min_val <= 99 and 0 <= max_val <= 99:
            def convert_2digit_year(x):
                if pd.notna(x) and 0 <= x <= 99:
                    if x <= 14:
                        return x + 2000  # 0-14 → 2000-2014
                    else:
                        return x + 1900  # 15-99 → 1915-1999
                return x
            
            df_raw[col] = df_raw[col].apply(convert_2digit_year)
            converted_count += 1

if converted_count > 0:
    print(f"  Converted {converted_count} year variables: 0-14 → 2000-2014, 15-99 → 1915-1999")
else:
    print("  No 2-digit year variables found")

if skipped_month_vars > 0:
    print(f"  Skipped {skipped_month_vars} month variables (not converted)")

# Report hybrid variables
hybrid_vars = [col for col, vtype in variable_types.items() if vtype == 'continuous_hybrid']

# Report hybrid variables
hybrid_vars = [col for col, vtype in variable_types.items() if vtype == 'continuous_hybrid']
if hybrid_vars:
    print(f"\n  ⚠️  Detected {len(hybrid_vars)} hybrid encoding variables:")
    print(f"     (mix categorical status codes with continuous IDs)")
    print(f"     These will be treated as continuous but may need special handling in ML.")
    if len(hybrid_vars) <= 5:
        for var in hybrid_vars:
            print(f"       - {var}")
    else:
        print(f"       First 5: {', '.join(hybrid_vars[:5])}")
        print(f"       (See variable_classifications.csv for full list)")

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

print("\n" + "=" * 80)
print("CLASSIFICATION COMPLETE")
print("=" * 80)
print(f"\nTotal variables classified: {len(variable_types)}")
print(f"SAS-based: {len(sas_type_mapping)}")
print(f"Heuristic-based: {len(variable_types) - len(sas_type_mapping)}")
print("\nNext step: Run step2_trajectory_and_ml.py")
