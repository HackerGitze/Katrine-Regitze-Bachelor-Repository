import pandas as pd
import numpy as np
import os
import re
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict

# Set random seed for reproducibility (probabilistic imputation uses random sampling)
np.random.seed(42)

print("=" * 80)
print("STEP 2: AGGREGATION & ML PREPARATION")
print("=" * 80)

# ============================================================================
# LOAD CLASSIFIED DATA & FILTER HIGH-MISSINGNESS VARIABLES
# ============================================================================
print("\n[2.1] Loading classified data...")
SOURCE_FILE = "nlsy79_classified.parquet"
CLASSIFICATIONS_FILE = "variable_classifications.csv"
REVIEW_FILE = "inconsistent_variables_review.csv"

if not os.path.exists(SOURCE_FILE):
    raise FileNotFoundError(f"Classified data not found: {SOURCE_FILE}\nPlease run step_1_classification.py first")

if not os.path.exists(CLASSIFICATIONS_FILE):
    raise FileNotFoundError(f"Classifications not found: {CLASSIFICATIONS_FILE}\nPlease run step_1_classification.py first")

df = pd.read_parquet(SOURCE_FILE)
classifications_df = pd.read_csv(CLASSIFICATIONS_FILE)
variable_types = dict(zip(classifications_df['variable_name'], classifications_df['variable_type']))

# Remove participants with >25% missing data (now in Step 2)
print("\n[2.1a] Filtering participants with >25% missing data...")
missing_pct_rows = df.isnull().sum(axis=1) / len(df.columns) * 100
rows_to_keep = missing_pct_rows <= 25
participants_removed = (~rows_to_keep).sum()
df = df[rows_to_keep]
print(f"  Removed {participants_removed} participants with >25% missingness")
print(f"  Remaining: {df.shape[0]} rows × {df.shape[1]} columns")

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
    # First, check if variable has dot notation (e.g., .01, .02) - keep it in base name
    # This prevents mixing cross-sectional IDs (household members) with temporal aggregation
    base = var_name
    
    # Extract patterns in order: remove year suffixes but preserve dot notation
    year_patterns = [
        r'_19\d{2}$', r'_20\d{2}$',  # Year suffixes
        r'_XRND$', r'_TNEW_XRND$'  # Cross-round indicators
    ]
    
    for pattern in year_patterns:
        base = re.sub(pattern, '', base)
    
    # After removing year, if there's still a trailing dot notation, it becomes part of the base
    # This means HHI_FINAL_GENCODE.01_1979 -> HHI_FINAL_GENCODE.01
    #            HHI_FINAL_GENCODE.02_1979 -> HHI_FINAL_GENCODE.02
    # These will be separate groups for each household member
    
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

# Create two filtered groupings for different aggregation strategies
variable_groups_highly = {base: vars_list for base, vars_list in variable_groups.items() if len(vars_list) >= 3}
variable_groups_mildly = {base: vars_list for base, vars_list in variable_groups.items() if len(vars_list) >= 30}

print(f"  Found {len(variable_groups)} variable groups")
print(f"  Highly aggregated branch: {len(variable_groups_highly)} groups (≥3 timepoints)")
print(f"  Mildly aggregated branch: {len(variable_groups_mildly)} groups (≥30 timepoints)")

# ============================================================================
# CLASSIFICATION DISTRIBUTION SUMMARY
# ============================================================================
print("\n[2.2a] Variable classification distribution before branching:")

# Count classifications across all variables
type_counts = pd.Series(variable_types).value_counts()
print(f"\n  Total variables: {len(variable_types)}")
print(f"  Classification breakdown:")
for vtype, count in type_counts.items():
    pct = count / len(variable_types) * 100
    print(f"    {vtype}: {count} ({pct:.1f}%)")

# Count variables that will be aggregated vs. kept static
all_aggregated_vars = []
for base, vars_list in variable_groups.items():
    all_aggregated_vars.extend(vars_list)
all_aggregated_vars = set(all_aggregated_vars)
static_vars = [v for v in df.columns if v not in all_aggregated_vars]

print(f"\n  Variables to aggregate (in any branch): {len(all_aggregated_vars)}")
print(f"  Variables to keep static: {len(static_vars)}")

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
        else:
            years.append(None)
    
    has_years = len([y for y in years if y is not None]) >= 2
    
    if has_years:
        valid_years = [y for y in years if y is not None]
        span = max(valid_years) - min(valid_years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    mean_vals = subset.mean(axis=1)
    std_vals = subset.std(axis=1)
    
    # Compute time-weighted trend using actual years
    def compute_trend_with_years(row):
        # Pair values with their years, then drop NAs
        pairs = [(year, val) for year, val in zip(years, row) if pd.notna(val) and year is not None]
        if len(pairs) < 2:
            return 0
        years_used, vals_used = zip(*pairs)
        try:
            slope, _ = np.polyfit(years_used, vals_used, 1)
            return slope
        except:
            return 0
    
    trends = subset.apply(compute_trend_with_years, axis=1)
    
    return pd.DataFrame({
        f"{base_name.lower()}_mean_n{n_vars:04d}_{span_suffix}": mean_vals,
        f"{base_name.lower()}_std_n{n_vars:04d}_{span_suffix}": std_vals,
        f"{base_name.lower()}_trend_n{n_vars:04d}_{span_suffix}": trends
    }), has_years

def aggregate_continuous(df, cols, base_name):
    """Aggregate continuous variables."""
    subset = df[cols].copy()
    n_vars = len(cols)
    
    years = []
    for col in cols:
        year_match = re.search(r'(19|20)\d{2}', col)
        if year_match:
            years.append(int(year_match.group()))
        else:
            years.append(None)
    
    has_years = len([y for y in years if y is not None]) >= 2
    
    if has_years:
        valid_years = [y for y in years if y is not None]
        span = max(valid_years) - min(valid_years)
        span_suffix = f"s{span:02d}"
    else:
        span_suffix = "s00"
    
    mean_vals = subset.mean(axis=1)
    std_vals = subset.std(axis=1)
    
    # Compute time-weighted trend using actual years
    def compute_trend_with_years(row):
        # Pair values with their years, then drop NAs
        pairs = [(year, val) for year, val in zip(years, row) if pd.notna(val) and year is not None]
        if len(pairs) < 2:
            return 0
        years_used, vals_used = zip(*pairs)
        try:
            slope, _ = np.polyfit(years_used, vals_used, 1)
            return slope
        except:
            return 0
    
    trends = subset.apply(compute_trend_with_years, axis=1)
    
    return pd.DataFrame({
        f"{base_name.lower()}_mean_n{n_vars:04d}_{span_suffix}": mean_vals,
        f"{base_name.lower()}_std_n{n_vars:04d}_{span_suffix}": std_vals,
        f"{base_name.lower()}_trend_n{n_vars:04d}_{span_suffix}": trends
    }), has_years

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
# AGGREGATION BRANCH FUNCTION
# ============================================================================
def run_aggregation_branch(df_input, variable_groups_branch, variable_types, branch_name, threshold):
    """
    Run complete aggregation pipeline for a specific branch.
    
    Parameters:
    - df_input: Input DataFrame
    - variable_groups_branch: Dictionary of variable groups to aggregate
    - variable_types: Dictionary mapping variable names to types
    - branch_name: Name for output files (e.g., 'highly', 'mildly')
    - threshold: Minimum number of timepoints for this branch
    """
    print("\n" + "=" * 80)
    print(f"BRANCH: {branch_name.upper()} AGGREGATED (≥{threshold} timepoints)")
    print("=" * 80)
    
    # Save list of groups being aggregated
    groups_log_data = []
    for base_name, vars_list in variable_groups_branch.items():
        types_in_group = [variable_types.get(v, 'continuous') for v in vars_list]
        group_type = max(set(types_in_group), key=types_in_group.count)
        groups_log_data.append({
            'base_name': base_name,
            'n_timepoints': len(vars_list),
            'assigned_type': group_type,
            'variables': ', '.join(vars_list)
        })
    
    groups_log_df = pd.DataFrame(groups_log_data)
    groups_log_file = f'aggregated_groups_{branch_name}.csv'
    groups_log_df.to_csv(groups_log_file, index=False)
    print(f"  Saved list of aggregated groups to: {groups_log_file}")
    
    # ============================================================================
    # AGGREGATE VARIABLES
    # ============================================================================
    print(f"\n[2.4-{branch_name}] Aggregating longitudinal variables...")
    
    aggregated_dfs = []
    inconsistent_vars = []
    total_groups = len(variable_groups_branch)
    processed = 0
    
    # Track groups without year information
    ordinal_no_years = 0
    continuous_no_years = 0
    ordinal_total = 0
    continuous_total = 0
    ordinal_no_years_list = []
    continuous_no_years_list = []
    
    for base_name, vars_list in variable_groups_branch.items():
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
        agg_df = None
        has_years = True
        try:
            if group_type == 'binary':
                agg_df = aggregate_binary(df_input, vars_list, base_name)
            elif group_type == 'categorical_ordinal':
                ordinal_total += 1
                agg_df, has_years = aggregate_ordinal(df_input, vars_list, base_name)
                if not has_years:
                    ordinal_no_years += 1
                    ordinal_no_years_list.append(base_name)
            elif group_type == 'continuous' or group_type == 'year':
                continuous_total += 1
                # Treat year variables as continuous (mean, std, trend more informative than earliest/latest/range)
                agg_df, has_years = aggregate_continuous(df_input, vars_list, base_name)
                if not has_years:
                    continuous_no_years += 1
                    continuous_no_years_list.append(base_name)
            elif group_type == 'categorical_nominal':
                agg_df = aggregate_nominal(df_input, vars_list, base_name)
            
            if agg_df is not None:
                aggregated_dfs.append(agg_df)
        except Exception as e:
            print(f"  Warning: Failed to aggregate {base_name}: {e}")
    
    # Report year information statistics
    print(f"\n  Year information statistics:")
    if ordinal_total > 0:
        print(f"    Categorical Ordinal: {ordinal_no_years}/{ordinal_total} groups ({ordinal_no_years/ordinal_total*100:.1f}%) without sufficient year info")
        if ordinal_no_years_list:
            print(f"      Variables: {', '.join(ordinal_no_years_list)}")
    if continuous_total > 0:
        print(f"    Continuous/Year: {continuous_no_years}/{continuous_total} groups ({continuous_no_years/continuous_total*100:.1f}%) without sufficient year info")
        if continuous_no_years_list:
            print(f"      Variables: {', '.join(continuous_no_years_list)}")

    # Combine all aggregated variables
    df_aggregated = pd.concat(aggregated_dfs, axis=1) if aggregated_dfs else pd.DataFrame(index=df_input.index)
    print(f"  Created {len(df_aggregated.columns)} aggregated features from {len(variable_groups_branch)} variable groups")
    
    # Identify all variables that were aggregated (to be dropped)
    aggregated_vars = []
    for base, vars_list in variable_groups_branch.items():
        aggregated_vars.extend(vars_list)
    
    # Get non-aggregated variables (static/non-temporal features)
    non_aggregated_vars = [col for col in df_input.columns if col not in aggregated_vars]
    df_non_aggregated = df_input[non_aggregated_vars]
    
    print(f"  Aggregated variables (to be dropped): {len(aggregated_vars)}")
    print(f"  Non-aggregated variables (to keep): {len(non_aggregated_vars)}")
    
    # Combine aggregated features with non-aggregated variables
    df_combined = pd.concat([df_non_aggregated, df_aggregated], axis=1)
    print(f"  Combined dataset: {df_combined.shape[0]} × {df_combined.shape[1]} (non-aggregated + aggregated)")
    
    # Save inconsistency report
    if len(inconsistent_vars) > 0:
        with open(f'inconsistent_variables_report_{branch_name}.txt', 'w') as f:
            f.write(f"INCONSISTENT VARIABLE GROUPS REPORT - {branch_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            for item in inconsistent_vars:
                f.write(f"Base Name: {item['base_name']}\n")
                f.write(f"Assigned Type: {item['assigned_type']}\n")
                f.write(f"Variables ({len(item['variables'])}):\n")
                for var, vtype in zip(item['variables'], item['types']):
                    f.write(f"  - {var}: {vtype}\n")
                f.write("\n")
        print(f"  Found {len(inconsistent_vars)} groups with inconsistent types (saved to inconsistent_variables_report_{branch_name}.txt)")

    # ============================================================================
    # POST-AGGREGATION RECLASSIFICATION
    # ============================================================================
    print(f"\n[2.5-{branch_name}] Reclassifying all variables...")
    print(f"  Processing {len(df_combined.columns)} variables...")
    
    def classify_aggregated_variable(col):
        """
        Hard-coded classifications for aggregated features.
        
        Each aggregation function creates specific features with known types:
        - Binary: prop_positive, prop_shifts → continuous
        - Ordinal: mean, std, trend → continuous
        - Continuous: mean, std, trend → continuous
        - Nominal: mode → categorical_nominal, entropy → continuous
        """
        # Nominal aggregation: mode stays categorical, entropy is continuous
        if '_mode_' in col:
            return 'categorical_nominal'
        if '_entropy_' in col:
            return 'continuous'
        
        # Binary aggregation: both proportions are continuous
        if '_prop_positive_' in col or '_prop_shifts_' in col:
            return 'continuous'
        
        # Ordinal/Continuous aggregation: all are continuous
        if '_mean_' in col or '_std_' in col or '_trend_' in col:
            return 'continuous'
        
        # This should never happen, but default to continuous
        return 'continuous'
    
    # Classify aggregated variables
    aggregated_types = {col: classify_aggregated_variable(col) for col in df_aggregated.columns}
    
    # Preserve types for non-aggregated variables from original classifications
    combined_types = {}
    for col in df_combined.columns:
        if col in aggregated_types:
            combined_types[col] = aggregated_types[col]
        elif col in variable_types:
            combined_types[col] = variable_types[col]
        else:
            # Fallback classification for any unclassified variables
            combined_types[col] = classify_aggregated_variable(col)
    
    print(f"  Classified {len(combined_types)} total variables ({len(non_aggregated_vars)} non-aggregated + {len(aggregated_types)} aggregated)")

    # Note: Variable missingness filtering already done in Step 2.1a
    # Participant filtering already done in Step 1
    
    # ============================================================================
    # MISSING VALUE ANALYSIS
    # ============================================================================
    print(f"\n[2.7a-{branch_name}] Missing value analysis before imputation...")
    
    total_missing = df_combined.isnull().sum().sum()
    total_values = df_combined.shape[0] * df_combined.shape[1]
    missing_pct = (total_missing / total_values) * 100
    
    print(f"  Total missing values: {total_missing:,} / {total_values:,} ({missing_pct:.2f}%)")
    print(f"  Columns with missing values: {df_combined.isnull().any().sum()} / {len(df_combined.columns)}")
    print(f"  Rows with missing values: {df_combined.isnull().any(axis=1).sum()} / {len(df_combined)}")
    
    # Save missing value report
    missing_report = pd.DataFrame({
        'variable': df_combined.columns,
        'missing_count': df_combined.isnull().sum().values,
        'missing_pct': (df_combined.isnull().sum().values / len(df_combined) * 100)
    }).sort_values('missing_count', ascending=False)
    missing_report.to_csv(f'missing_values_report_{branch_name}.csv', index=False)
    print(f"  Saved: missing_values_report_{branch_name}.csv")
    
    # ============================================================================
    # TYPE-SPECIFIC IMPUTATION
    # ============================================================================
    print(f"\n[2.7b-{branch_name}] Imputing remaining missing values...")
    
    # Separate by type
    binary_cols = [col for col in df_combined.columns if combined_types[col] == 'binary']
    ordinal_cols = [col for col in df_combined.columns if combined_types[col] == 'categorical_ordinal']
    nominal_cols = [col for col in df_combined.columns if combined_types[col] == 'categorical_nominal']
    continuous_cols = [col for col in df_combined.columns if combined_types[col] == 'continuous']
    year_cols = [col for col in df_combined.columns if combined_types[col] == 'year']
    
    print(f"  Found: {len(binary_cols)} binary, {len(ordinal_cols)} ordinal, {len(nominal_cols)} nominal, {len(continuous_cols)} continuous, {len(year_cols)} year variables")
    
    df_imputed = df_combined.copy()
    
    # Helper function for probabilistic imputation (preserves distribution)
    def probabilistic_impute(series):
        """
        Impute missing values by sampling from the observed distribution.
        Prevents inflation of the mode by preserving value frequencies.
        """
        if series.isnull().sum() == 0:
            return series
        
        # Get distribution of non-missing values
        value_counts = series.dropna().value_counts(normalize=True)
        
        if len(value_counts) == 0:
            return series
        
        # Sample from distribution for missing values
        missing_mask = series.isnull()
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            # Random sampling based on observed frequencies
            imputed_values = np.random.choice(
                value_counts.index, 
                size=n_missing, 
                p=value_counts.values
            )
            series = series.copy()
            series.loc[missing_mask] = imputed_values
        
        return series
    
    # Binary: Probabilistic imputation (preserves 0/1 distribution)
    if len(binary_cols) > 0:
        print(f"  Imputing {len(binary_cols)} binary variables (probabilistic sampling)...")
        for col in binary_cols:
            df_imputed[col] = probabilistic_impute(df_combined[col])
    
    # Ordinal: Probabilistic imputation (preserves category distribution)
    if len(ordinal_cols) > 0:
        print(f"  Imputing {len(ordinal_cols)} ordinal variables (probabilistic sampling)...")
        for col in ordinal_cols:
            df_imputed[col] = probabilistic_impute(df_combined[col])
    
    # Nominal: Probabilistic imputation (preserves category distribution)
    if len(nominal_cols) > 0:
        print(f"  Imputing {len(nominal_cols)} nominal variables (probabilistic sampling)...")
        for col in nominal_cols:
            df_imputed[col] = probabilistic_impute(df_combined[col])
    
    # Continuous & Year: Median imputation (numeric features, robust to outliers)
    if len(continuous_cols) > 0:
        print(f"  Imputing {len(continuous_cols)} continuous variables (median)...")
        imputer_continuous = SimpleImputer(strategy='median')
        df_imputed[continuous_cols] = imputer_continuous.fit_transform(df_combined[continuous_cols])
    
    if len(year_cols) > 0:
        print("  Imputing year variables (median)...")
        imputer_year = SimpleImputer(strategy='median')
        df_imputed[year_cols] = imputer_year.fit_transform(df_combined[year_cols])
    
    print(f"  Imputation complete. Missing values remaining: {df_imputed.isnull().sum().sum()}")

    # ============================================================================
    # LOW-VARIANCE FILTERING (APPLIED TO BOTH IMPUTED AND NON-IMPUTED)
    # ============================================================================
    print(f"\n[2.8-{branch_name}] Filtering low-variance variables...")
    
    # Use imputed data for variance calculation (more reliable with complete data)
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
    
    # Apply variance filter to both imputed and non-imputed versions
    df_imputed_filtered = df_imputed.drop(columns=low_variance_cols)
    df_notimputed_filtered = df_combined.drop(columns=low_variance_cols)
    low_var_count = len(low_variance_cols)
    
    print(f"  Removed {low_var_count} variables with ≥95% identical values")
    print(f"  Remaining variables: {len(df_imputed_filtered.columns)}")
    
    # Save variance report
    variance_df.to_csv(f'variance_report_{branch_name}.csv', index=False)
    print(f"  Saved: variance_report_{branch_name}.csv")

    # Prepare both versions
    df_final_imputed = df_imputed_filtered.copy()
    df_final_notimputed = df_notimputed_filtered.copy()
    
    # ============================================================================
    # SAVE FINAL OUTPUTS (BOTH IMPUTED AND NON-IMPUTED)
    # ============================================================================
    print(f"\n[2.10-{branch_name}] Saving final ML-ready outputs...")
    
    # Save imputed version
    parquet_name_imputed = f'step_2_{branch_name}_aggregated.parquet'
    df_final_imputed.to_parquet(parquet_name_imputed, index=True)
    print(f"  Saved (imputed): {parquet_name_imputed} ({df_final_imputed.shape[0]} × {df_final_imputed.shape[1]})")
    
    # Save non-imputed version
    parquet_name_notimputed = f'step_2_{branch_name}_aggregated_notimputed.parquet'
    df_final_notimputed.to_parquet(parquet_name_notimputed, index=True)
    print(f"  Saved (not imputed): {parquet_name_notimputed} ({df_final_notimputed.shape[0]} × {df_final_notimputed.shape[1]})")
    print(f"    Missing values in non-imputed: {df_final_notimputed.isnull().sum().sum():,}")
    
    # Save classifications (same for both versions)
    classifications_output = pd.DataFrame([
        {'variable_name': col, 'variable_type': combined_types[col]}
        for col in df_final_imputed.columns
    ])
    class_name = f'step_2_{branch_name}_classifications.csv'
    classifications_output.to_csv(class_name, index=False)
    print(f"  Saved: {class_name} ({len(classifications_output)} variables)")
    
    # Save summary
    summary = {
        'branch': branch_name,
        'threshold': threshold,
        'input_rows': len(df_input),
        'input_cols': len(df_input.columns),
        'aggregated_vars_count': len(aggregated_vars),
        'non_aggregated_vars_count': len(non_aggregated_vars),
        'aggregated_cols': len(df_aggregated.columns),
        'combined_cols': len(df_combined.columns),
        'missing_before_imputation': total_missing,
        'missing_pct_before_imputation': missing_pct,
        'low_variance_removed': low_var_count,
        'final_rows_imputed': len(df_final_imputed),
        'final_cols_imputed': len(df_final_imputed.columns),
        'final_rows_notimputed': len(df_final_notimputed),
        'final_cols_notimputed': len(df_final_notimputed.columns),
        'missing_in_notimputed': df_final_notimputed.isnull().sum().sum(),
        'dimensionality_reduction_pct': (1 - len(df_final_imputed.columns)/len(df_input.columns)) * 100
    }
    summary_df = pd.DataFrame([summary])
    summary_name = f'pipeline_summary_{branch_name}.csv'
    summary_df.to_csv(summary_name, index=False)
    print(f"  Saved: {summary_name}")

    # ============================================================================
    # BRANCH SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"BRANCH COMPLETE: {branch_name.upper()} AGGREGATED")
    print("=" * 80)
    print(f"\nInput dataset: {df_input.shape[0]} × {df_input.shape[1]}")
    print(f"  - Aggregated variables (dropped): {len(aggregated_vars)}")
    print(f"  - Non-aggregated variables (kept): {len(non_aggregated_vars)}")
    print(f"After aggregation: {df_aggregated.shape[0]} × {df_aggregated.shape[1]}")
    print(f"Combined (non-aggregated + aggregated): {df_combined.shape[0]} × {df_combined.shape[1]}")
    print(f"Missing values before imputation: {total_missing:,} ({missing_pct:.2f}%)")
    print(f"After imputation: {df_imputed.shape[0]} × {df_imputed.shape[1]}")
    print(f"ML-ready dataset (imputed): {df_final_imputed.shape[0]} × {df_final_imputed.shape[1]}")
    print(f"ML-ready dataset (not imputed): {df_final_notimputed.shape[0]} × {df_final_notimputed.shape[1]}")
    print(f"\nDimensionality reduction: {summary['dimensionality_reduction_pct']:.1f}%")
    print(f"Variables removed (low variance): {low_var_count}")
    print(f"Participants retained: {len(df_final_imputed)} / {len(df_input)} ({len(df_final_imputed)/len(df_input)*100:.1f}%)")
    print("\n✅ Data is ready for machine learning!")
    
    return df_final_imputed, df_final_notimputed, classifications_output, summary

# ============================================================================
# RUN BOTH AGGREGATION BRANCHES
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING DUAL-BRANCH AGGREGATION PIPELINE")
print("=" * 80)

# Run highly aggregated branch (≥3 timepoints)
df_highly_imp, df_highly_notimp, class_highly, summary_highly = run_aggregation_branch(
    df, variable_groups_highly, variable_types, 'highly', 3
)

# Run mildly aggregated branch (≥30 timepoints)
df_mildly_imp, df_mildly_notimp, class_mildly, summary_mildly = run_aggregation_branch(
    df, variable_groups_mildly, variable_types, 'mildly', 30
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DUAL-BRANCH AGGREGATION COMPLETE")
print("=" * 80)
print(f"\nHIGHLY AGGREGATED (≥3 timepoints):")
print(f"  Groups aggregated: {len(variable_groups_highly)}")
print(f"  Final dimensions (imputed): {df_highly_imp.shape[0]} × {df_highly_imp.shape[1]}")
print(f"  Final dimensions (not imputed): {df_highly_notimp.shape[0]} × {df_highly_notimp.shape[1]}")
print(f"  Missing values (not imputed): {df_highly_notimp.isnull().sum().sum():,}")
print(f"  Dimensionality reduction: {summary_highly['dimensionality_reduction_pct']:.1f}%")
print(f"  Output files:")
print(f"    - step_2_highly_aggregated.parquet (imputed)")
print(f"    - step_2_highly_aggregated_notimputed.parquet")
print(f"    - step_2_highly_classifications.csv ({len(class_highly)} variables)")

print(f"\nMILDLY AGGREGATED (≥30 timepoints):")
print(f"  Groups aggregated: {len(variable_groups_mildly)}")
print(f"  Final dimensions (imputed): {df_mildly_imp.shape[0]} × {df_mildly_imp.shape[1]}")
print(f"  Final dimensions (not imputed): {df_mildly_notimp.shape[0]} × {df_mildly_notimp.shape[1]}")
print(f"  Missing values (not imputed): {df_mildly_notimp.isnull().sum().sum():,}")
print(f"  Dimensionality reduction: {summary_mildly['dimensionality_reduction_pct']:.1f}%")
print(f"  Output files:")
print(f"    - step_2_mildly_aggregated.parquet (imputed)")
print(f"    - step_2_mildly_aggregated_notimputed.parquet")
print(f"    - step_2_mildly_classifications.csv ({len(class_mildly)} variables)")

print("\n✅ Both aggregation branches complete! Ready for machine learning.")
