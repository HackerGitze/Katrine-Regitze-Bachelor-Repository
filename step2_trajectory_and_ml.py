import pandas as pd
import numpy as np
import os
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict

print("=" * 80)
print("STEP 2: AGGREGATION & ML PREPARATION")
print("=" * 80)

# ============================================================================
# LOAD CLASSIFIED DATA
# ============================================================================
print("\n[2.1] Loading classified data...")
SOURCE_FILE = "nlsy79_classified.parquet"
CLASSIFICATIONS_FILE = "variable_classifications.csv"
REVIEW_FILE = "inconsistent_variables_review.csv"

if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Classified data not found: {SOURCE_FILE}\nPlease run step1_classify_variables.py first")

if not os.path.exists(CLASSIFICATIONS_FILE):
    raise FileNotFoundError(f"Classifications not found: {CLASSIFICATIONS_FILE}\nPlease run step1_classify_variables.py first")

df = pd.read_parquet(SOURCE_FILE)
classifications_df = pd.read_csv(CLASSIFICATIONS_FILE)
variable_types = dict(zip(classifications_df['variable_name'], classifications_df['variable_type']))

# Load manual review corrections if available
manual_corrections = {}
subgroup_assignments = {}  # For YES1/YES2 split groups
if os.path.exists(REVIEW_FILE):
    print(f"  Loading manual corrections from {REVIEW_FILE}...")
    review_df = pd.read_csv(REVIEW_FILE, sep=';')
    
    # Apply corrected types (if corrected_type is not empty)
    corrections_applied = 0
    for _, row in review_df.iterrows():
        var_name = row['variable_name']
        corrected_type = row.get('corrected_type', '')
        aggregate_flag = row.get('aggregate_with_group', 'YES')
        
        # Apply type correction
        if pd.notna(corrected_type) and corrected_type.strip() != '':
            manual_corrections[var_name] = corrected_type.strip()
            variable_types[var_name] = corrected_type.strip()
            corrections_applied += 1
        
        # Handle subgroup assignments (YES1, YES2, etc.)
        if pd.notna(aggregate_flag) and aggregate_flag not in ['YES', 'NO', '']:
            base_name = row['base_name']
            # Convert YES1 -> g1, YES2 -> g2, etc.
            group_suffix = aggregate_flag.replace('YES', 'g')
            subgroup_assignments[var_name] = f"{base_name}_{group_suffix}"
    
    print(f"    Applied {corrections_applied} type corrections")
    print(f"    Found {len(subgroup_assignments)} subgroup assignments (YES1/YES2)")

print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Classifications: {len(variable_types)} variables")

# ============================================================================
# GROUP VARIABLES BY BASE NAME
# ============================================================================
print("\n[2.2] Grouping longitudinal variables by base name...")

def extract_base_name(var_name):
    """Extract base name by removing common suffixes."""
    # Hardcoded mappings for specific prefixes that should be aggregated together
    if var_name.startswith('STATUS'):
        # All STATUS* variables aggregate together
        base = 'STATUS'
        # Check if it has temporal suffix
        if re.search(r'(_19\d{2}|_20\d{2}|\.\d{1,2}|_XRND|_TNEW_XRND)$', var_name):
            return base
        else:
            return None
    
    if var_name.startswith('HRS_WO'):
        # All HRS_WO* variables aggregate together
        base = 'HRS_WO'
        # Check if it has temporal suffix
        if re.search(r'(_19\d{2}|_20\d{2}|\.\d{1,2}|_XRND|_TNEW_XRND)$', var_name):
            return base
        else:
            return None
    
    # Standard pattern-based extraction
    patterns = [
        r'_19\d{2}$', r'_20\d{2}$',  # Year suffixes
        r'\.\d{1,2}$',  # Numeric identifiers
        r'_XRND$', r'_TNEW_XRND$'  # Cross-round indicators
    ]
    
    base = var_name
    for pattern in patterns:
        base = re.sub(pattern, '', base)
    
    return base if base != var_name else None

# Group variables
variable_groups = defaultdict(list)
for var in df.columns:
    # Check if variable has subgroup assignment (YES1/YES2)
    if var in subgroup_assignments:
        base = subgroup_assignments[var]
    else:
        base = extract_base_name(var)
    
    if base:
        variable_groups[base].append(var)

# Check for NO aggregation flags and remove those variables from groups
if os.path.exists(REVIEW_FILE):
    review_df = pd.read_csv(REVIEW_FILE, sep=';')
    no_aggregate_vars = review_df[review_df['aggregate_with_group'] == 'NO']['variable_name'].tolist()
    
    if no_aggregate_vars:
        print(f"  Excluding {len(no_aggregate_vars)} variables marked with aggregate_with_group=NO")
        # Remove from groups
        for base_name in list(variable_groups.keys()):
            variable_groups[base_name] = [v for v in variable_groups[base_name] if v not in no_aggregate_vars]
            # Remove empty groups
            if len(variable_groups[base_name]) == 0:
                del variable_groups[base_name]

# Filter to groups with >=3 timepoints
variable_groups_filtered = {base: vars_list for base, vars_list in variable_groups.items() if len(vars_list) >= 3}

print(f"  Found {len(variable_groups)} variable groups")
print(f"  Retained {len(variable_groups_filtered)} groups with ≥3 timepoints")

# ============================================================================
# TYPE-SPECIFIC AGGREGATION FUNCTIONS
# ============================================================================
print("\n[2.3] Defining aggregation functions...")

def aggregate_binary(df, cols, base_name):
    """Aggregate binary variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    # Extract years
    years = []
    for col in cols:
        year_match = re.search(r'(19|20)\d{2}', col)
        if year_match:
            years.append(int(year_match.group()))
    
    if len(years) >= 2:
        span = max(years) - min(years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    # Compute metrics
    prop_positive = subset.mean(axis=1)
    shifts = (subset.diff(axis=1) != 0).sum(axis=1)
    prop_shifts = shifts / (n_vars - 1)
    
    return pd.DataFrame({
        f"{base_name.lower()}_prop_positive_n{n_vars:04d}_{span_suffix}": prop_positive,
        f"{base_name.lower()}_prop_shifts_n{n_vars:04d}_{span_suffix}": prop_shifts
    })

def aggregate_ordinal(df, cols, base_name):
    """Aggregate ordinal variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    years = []
    for col in cols:
        year_match = re.search(r'(19|20)\d{2}', col)
        if year_match:
            years.append(int(year_match.group()))
    
    if len(years) >= 2:
        span = max(years) - min(years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    mean_vals = subset.mean(axis=1)
    std_vals = subset.std(axis=1)
    
    # Compute time-weighted trend
    def compute_trend(row):
        vals = row.dropna()
        if len(vals) < 2:
            return 0
        x = np.arange(len(vals))
        try:
            slope, _ = np.polyfit(x, vals.values, 1)
            return slope
        except:
            return 0
    
    harmonized = subset.apply(lambda row: row.dropna(), axis=1)
    trends = harmonized.apply(compute_trend, axis=1, raw=False)
    
    return pd.DataFrame({
        f"{base_name.lower()}_mean_n{n_vars:04d}_{span_suffix}": mean_vals,
        f"{base_name.lower()}_std_n{n_vars:04d}_{span_suffix}": std_vals,
        f"{base_name.lower()}_trend_n{n_vars:04d}_{span_suffix}": trends
    })

def aggregate_continuous(df, cols, base_name):
    """Aggregate continuous variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    years = []
    for col in cols:
        year_match = re.search(r'(19|20)\d{2}', col)
        if year_match:
            years.append(int(year_match.group()))
    
    if len(years) >= 2:
        span = max(years) - min(years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    mean_vals = subset.mean(axis=1)
    std_vals = subset.std(axis=1)
    
    def compute_trend(row):
        vals = row.dropna()
        if len(vals) < 2:
            return 0
        x = np.arange(len(vals))
        try:
            slope, _ = np.polyfit(x, vals.values, 1)
            return slope
        except:
            return 0
    
    harmonized = subset.apply(lambda row: row.dropna(), axis=1)
    trends = harmonized.apply(compute_trend, axis=1, raw=False)
    
    return pd.DataFrame({
        f"{base_name.lower()}_mean_n{n_vars:04d}_{span_suffix}": mean_vals,
        f"{base_name.lower()}_std_n{n_vars:04d}_{span_suffix}": std_vals,
        f"{base_name.lower()}_trend_n{n_vars:04d}_{span_suffix}": trends
    })

def aggregate_nominal(df, cols, base_name):
    """Aggregate nominal variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    years = []
    for col in cols:
        year_match = re.search(r'(19|20)\d{2}', col)
        if year_match:
            years.append(int(year_match.group()))
    
    if len(years) >= 2:
        span = max(years) - min(years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    # Mode
    mode_vals = subset.mode(axis=1).iloc[:, 0] if not subset.mode(axis=1).empty else pd.Series(index=subset.index)
    
    # Shannon entropy
    def compute_entropy(row):
        vals = row.dropna()
        if len(vals) == 0:
            return 0
        value_counts = vals.value_counts()
        probs = value_counts / len(vals)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    entropy_vals = subset.apply(compute_entropy, axis=1)
    
    return pd.DataFrame({
        f"{base_name.lower()}_mode_n{n_vars:04d}_{span_suffix}": mode_vals,
        f"{base_name.lower()}_entropy_n{n_vars:04d}_{span_suffix}": entropy_vals
    })

def aggregate_year(df, cols, base_name):
    """Aggregate year variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    earliest = subset.min(axis=1)
    latest = subset.max(axis=1)
    range_vals = latest - earliest
    
    return pd.DataFrame({
        f"{base_name.lower()}_earliest_n{n_vars:04d}_s00": earliest,
        f"{base_name.lower()}_latest_n{n_vars:04d}_s00": latest,
        f"{base_name.lower()}_range_n{n_vars:04d}_s00": range_vals
    })

# ============================================================================
# AGGREGATE VARIABLES
# ============================================================================
print("\n[2.4] Aggregating longitudinal variables...")

aggregated_dfs = []
inconsistent_vars = []
total_groups = len(variable_groups_filtered)
processed = 0

for base_name, vars_list in variable_groups_filtered.items():
    processed += 1
    if processed % 100 == 0 or processed == total_groups:
        print(f"  Progress: {processed}/{total_groups} groups ({processed/total_groups*100:.1f}%)")
    # Determine group type (most common type in group)
    types_in_group = [variable_types.get(v, 'continuous') for v in vars_list]
    group_type = max(set(types_in_group), key=types_in_group.count)
    
    # Check for inconsistency
    if len(set(types_in_group)) > 1:
        inconsistent_vars.append({
            'base_name': base_name,
            'variables': vars_list,
            'types': types_in_group,
            'assigned_type': group_type
        })
    
    # Aggregate based on type
    try:
        if group_type == 'binary':
            agg_df = aggregate_binary(df, vars_list, base_name)
        elif group_type == 'categorical_ordinal':
            agg_df = aggregate_ordinal(df, vars_list, base_name)
        elif group_type == 'continuous':
            agg_df = aggregate_continuous(df, vars_list, base_name)
        elif group_type == 'categorical_nominal':
            agg_df = aggregate_nominal(df, vars_list, base_name)
        elif group_type == 'year':
            agg_df = aggregate_year(df, vars_list, base_name)
        else:
            continue
        
        aggregated_dfs.append(agg_df)
    except Exception as e:
        print(f"  Warning: Failed to aggregate {base_name}: {e}")

# Combine all aggregated variables
df_aggregated = pd.concat(aggregated_dfs, axis=1)
print(f"  Created {len(df_aggregated.columns)} aggregated features from {len(variable_groups_filtered)} variable groups")

# Identify all variables that were aggregated (to be dropped)
aggregated_vars = []
for base, vars_list in variable_groups_filtered.items():
    aggregated_vars.extend(vars_list)

# Get non-aggregated variables (static/non-temporal features)
non_aggregated_vars = [col for col in df.columns if col not in aggregated_vars]
df_non_aggregated = df[non_aggregated_vars]

print(f"  Aggregated variables (to be dropped): {len(aggregated_vars)}")
print(f"  Non-aggregated variables (to keep): {len(non_aggregated_vars)}")

# Combine aggregated features with non-aggregated variables
df_combined = pd.concat([df_non_aggregated, df_aggregated], axis=1)
print(f"  Combined dataset: {df_combined.shape[0]} × {df_combined.shape[1]} (non-aggregated + aggregated)")

# Save inconsistency report
if len(inconsistent_vars) > 0:
    with open('inconsistent_variables_report.txt', 'w') as f:
        f.write(f"INCONSISTENT VARIABLE GROUPS REPORT\n")
        f.write("=" * 80 + "\n\n")
        for item in inconsistent_vars:
            f.write(f"Base Name: {item['base_name']}\n")
            f.write(f"Assigned Type: {item['assigned_type']}\n")
            f.write(f"Variables ({len(item['variables'])}):\n")
            for var, vtype in zip(item['variables'], item['types']):
                f.write(f"  - {var}: {vtype}\n")
            f.write("\n")
    print(f"  Found {len(inconsistent_vars)} groups with inconsistent types (saved to inconsistent_variables_report.txt)")

# ============================================================================
# POST-AGGREGATION RECLASSIFICATION
# ============================================================================
print("\n[2.5] Reclassifying all variables...")
print(f"  Processing {len(df_combined.columns)} variables...")

def classify_aggregated_variable(col, series):
    """Simple classification for aggregated variables."""
    dtype = series.dtype
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)
    
    if n_unique == 2:
        return 'binary'
    elif n_unique <= 20 and dtype == 'object':
        return 'categorical_nominal'
    elif dtype.kind in ['i', 'u', 'f']:
        if n_unique <= 10:
            return 'categorical_ordinal'
        else:
            return 'continuous'
    return 'continuous'

# Classify aggregated variables
aggregated_types = {col: classify_aggregated_variable(col, df_aggregated[col]) for col in df_aggregated.columns}

# Preserve types for non-aggregated variables from original classifications
combined_types = {}
for col in df_combined.columns:
    if col in aggregated_types:
        combined_types[col] = aggregated_types[col]
    elif col in variable_types:
        combined_types[col] = variable_types[col]
    else:
        # Fallback classification for any unclassified variables
        combined_types[col] = classify_aggregated_variable(col, df_combined[col])

print(f"  Classified {len(combined_types)} total variables ({len(non_aggregated_vars)} non-aggregated + {len(aggregated_types)} aggregated)")

# ============================================================================
# MISSINGNESS FILTERING (TWO-STAGE) - PARTICIPANT FILTERING NOW IN STEP1
# ============================================================================
print("\n[2.6] Filtering by missingness...")

# Stage 1: Remove variables with >25% missing
missing_pct_cols = df_combined.isnull().sum() / len(df_combined) * 100
cols_to_keep = missing_pct_cols[missing_pct_cols <= 25].index
df_filtered_vars = df_combined[cols_to_keep]

print(f"  Removed {len(df_combined.columns) - len(cols_to_keep)} variables with >25% missingness")
print(f"  Remaining variables: {len(df_filtered_vars.columns)}")

# Stage 2: SKIPPED - Participants already filtered in step1 for Rotter and >25% missingness
print(f"  Participant filtering skipped (already done in step1)")
df_filtered = df_filtered_vars

# # Stage 2: Remove participants with >25% missing (DISABLED - NOW IN STEP1)
# missing_pct_rows_filtered = df_filtered_vars.isnull().sum(axis=1) / len(df_filtered_vars.columns) * 100
# rows_to_keep = missing_pct_rows_filtered <= 25
# df_filtered = df_filtered_vars[rows_to_keep]
# 
# # Save dropped participant IDs
# dropped_ids = df_filtered_vars.index[~rows_to_keep].tolist()
# if len(dropped_ids) > 0:
#     with open('dropped_participant_ids.txt', 'w') as f:
#         f.write("Participant IDs Removed Due to >25% Missingness\n")
#         f.write("="*60 + "\n\n")
#         for pid in dropped_ids:
#             f.write(f"{pid}\n")
#     print(f"  Saved dropped IDs to: dropped_participant_ids.txt")
# 
# print(f"  Removed {(~rows_to_keep).sum()} participants with >25% missingness")

print(f"  Remaining participants: {len(df_filtered)}")

# ============================================================================
# TYPE-SPECIFIC IMPUTATION
# ============================================================================
print("\n[2.7] Imputing remaining missing values...")

# Separate by type (use combined_types instead of aggregated_types)
binary_cols = [col for col in df_filtered.columns if combined_types[col] == 'binary']
ordinal_cols = [col for col in df_filtered.columns if combined_types[col] == 'categorical_ordinal']
nominal_cols = [col for col in df_filtered.columns if combined_types[col] == 'categorical_nominal']
continuous_cols = [col for col in df_filtered.columns if combined_types[col] == 'continuous']
year_cols = [col for col in df_filtered.columns if combined_types[col] == 'year']

print(f"  Found: {len(binary_cols)} binary, {len(ordinal_cols)} ordinal, {len(nominal_cols)} nominal, {len(continuous_cols)} continuous, {len(year_cols)} year variables")

df_imputed = df_filtered.copy()

# Binary & Ordinal: Mode imputation
if len(binary_cols) > 0:
    print(f"  Imputing {len(binary_cols)} binary variables (mode)...")
    imputer_binary = SimpleImputer(strategy='most_frequent')
    df_imputed[binary_cols] = imputer_binary.fit_transform(df_filtered[binary_cols])

if len(ordinal_cols) > 0:
    print("  Imputing ordinal variables (mode)...")
    imputer_ordinal = SimpleImputer(strategy='most_frequent')
    df_imputed[ordinal_cols] = imputer_ordinal.fit_transform(df_filtered[ordinal_cols])

# Nominal: Mode imputation
if len(nominal_cols) > 0:
    print("  Imputing nominal variables (mode)...")
    imputer_nominal = SimpleImputer(strategy='most_frequent')
    df_imputed[nominal_cols] = imputer_nominal.fit_transform(df_filtered[nominal_cols])

# Continuous & Year: Median imputation
if len(continuous_cols) > 0:
    print("  Imputing continuous variables (median)...")
    imputer_continuous = SimpleImputer(strategy='median')
    df_imputed[continuous_cols] = imputer_continuous.fit_transform(df_filtered[continuous_cols])

if len(year_cols) > 0:
    print("  Imputing year variables (median)...")
    imputer_year = SimpleImputer(strategy='median')
    df_imputed[year_cols] = imputer_year.fit_transform(df_filtered[year_cols])

print(f"  Imputation complete. Missing values remaining: {df_imputed.isnull().sum().sum()}")

# ============================================================================
# LOW-VARIANCE FILTERING
# ============================================================================
print("\n[2.8] Filtering low-variance variables...")

variance_data = []
for col in df_imputed.columns:
    vals = df_imputed[col].value_counts()
    if len(vals) > 0:
        most_common_pct = vals.iloc[0] / len(df_imputed) * 100
        variance_data.append({
            'variable': col,
            'unique_values': len(vals),
            'most_common_pct': most_common_pct
        })

variance_df = pd.DataFrame(variance_data)
low_variance_cols = variance_df[variance_df['most_common_pct'] >= 95]['variable'].tolist()

df_variance_filtered = df_imputed.drop(columns=low_variance_cols)
low_var_count = len(low_variance_cols)

print(f"  Removed {low_var_count} variables with ≥95% identical values")
print(f"  Remaining variables: {len(df_variance_filtered.columns)}")

# Save variance report
variance_df.to_csv('variance_report.csv', index=False)
print("  Saved: variance_report.csv")

# ============================================================================
# STANDARDIZATION (SCALING) - COMMENTED OUT FOR NOW
# ============================================================================
# print("\n[2.9] Standardizing continuous/ordinal/year variables...")

# # Identify columns to scale (use combined_types instead of aggregated_types)
# cols_to_scale = [col for col in df_variance_filtered.columns 
#                  if combined_types.get(col) in ['continuous', 'categorical_ordinal', 'year']]

# df_scaled = df_variance_filtered.copy()

# if len(cols_to_scale) > 0:
#     scaler = StandardScaler()
#     df_scaled[cols_to_scale] = scaler.fit_transform(df_variance_filtered[cols_to_scale])
    
#     # Save scaler parameters
#     scaler_params = pd.DataFrame({
#         'variable': cols_to_scale,
#         'mean': scaler.mean_,
#         'std': scaler.scale_
#     })
#     scaler_params.to_csv('scaler_parameters.csv', index=False)
#     print(f"  Scaled {len(cols_to_scale)} variables")
#     print("  Saved: scaler_parameters.csv")
# else:
#     print("  No variables to scale")

# Use unscaled data for now
df_scaled = df_variance_filtered.copy()

# ============================================================================
# SAVE FINAL OUTPUTS
# ============================================================================
print("\n[2.10] Saving final ML-ready outputs...")

# Save ML-ready dataset
df_scaled.to_parquet('ml_ready_data.parquet', index=True)
print(f"  Saved: ml_ready_data.parquet ({df_scaled.shape[0]} × {df_scaled.shape[1]})")

# Save classifications
classifications_df = pd.DataFrame([
    {'variable_name': col, 'variable_type': combined_types[col]}
    for col in df_scaled.columns
])
classifications_df.to_csv('ml_ready_classifications.csv', index=False)
print(f"  Saved: ml_ready_classifications.csv ({len(classifications_df)} variables)")

# Save summary
summary = {
    'classified_data_rows': len(df),
    'classified_data_cols': len(df.columns),
    'aggregated_vars_count': len(aggregated_vars),
    'non_aggregated_vars_count': len(non_aggregated_vars),
    'aggregated_cols': len(df_aggregated.columns),
    'combined_cols': len(df_combined.columns),
    'after_missingness_filter_rows': len(df_filtered),
    'after_missingness_filter_cols': len(df_filtered.columns),
    'variance_threshold': 0.05,
    'low_variance_removed': low_var_count,
    'final_rows': len(df_scaled),
    'final_cols': len(df_scaled.columns),
    'dimensionality_reduction_pct': (1 - len(df_scaled.columns)/len(df.columns)) * 100
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv('pipeline_summary.csv', index=False)
print("  Saved: pipeline_summary.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("AGGREGATION & ML PREPARATION COMPLETE")
print("=" * 80)
print(f"\nClassified dataset: {df.shape[0]} × {df.shape[1]}")
print(f"  - Aggregated variables (dropped): {len(aggregated_vars)}")
print(f"  - Non-aggregated variables (kept): {len(non_aggregated_vars)}")
print(f"After aggregation: {df_aggregated.shape[0]} × {df_aggregated.shape[1]}")
print(f"Combined (non-aggregated + aggregated): {df_combined.shape[0]} × {df_combined.shape[1]}")
print(f"After filtering: {df_filtered.shape[0]} × {df_filtered.shape[1]}")
print(f"ML-ready dataset: {df_scaled.shape[0]} × {df_scaled.shape[1]}")
print(f"\nDimensionality reduction: {summary['dimensionality_reduction_pct']:.1f}%")
print(f"Variables removed (low variance): {low_var_count}")
print(f"Participants retained: {len(df_scaled)} / {len(df)} ({len(df_scaled)/len(df)*100:.1f}%)")
print("\n✅ Data is ready for machine learning!")
