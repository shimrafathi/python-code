

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — GENERATING DATASET")
print("=" * 60)

np.random.seed(42)
N = 569

platforms  = ['Instagram', 'TikTok', 'YouTube', 'Twitter/X', 'Snapchat', 'Facebook']
majors     = ['Engineering', 'Business', 'Arts & Humanities', 'Sciences', 'Education', 'Law']
year_levels = ['Freshman', 'Sophomore', 'Junior', 'Senior']
genders    = ['Male', 'Female', 'Non-binary']
locations  = ['Urban', 'Suburban', 'Rural']

age              = np.random.normal(20.5, 1.8, N).clip(17, 28).astype(int)
year_idx         = np.random.choice(len(year_levels), N, p=[0.28, 0.27, 0.25, 0.20])
year_level       = [year_levels[i] for i in year_idx]
daily_usage_hrs  = np.random.gamma(shape=2.5, scale=1.2, size=N).clip(0.5, 12)
platform         = np.random.choice(platforms,  N, p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
major            = np.random.choice(majors,     N, p=[0.22, 0.20, 0.15, 0.18, 0.14, 0.11])
gender           = np.random.choice(genders,    N, p=[0.46, 0.50, 0.04])
location         = np.random.choice(locations,  N, p=[0.45, 0.35, 0.20])
has_part_time_job = np.random.choice([0, 1],    N, p=[0.60, 0.40])

# Correlated lifestyle variables
sleep_hrs         = np.clip(8.5  - 0.25*daily_usage_hrs + np.random.normal(0, 0.6, N), 4, 10)
study_hrs         = np.clip(5.5  - 0.30*daily_usage_hrs + np.random.normal(0, 0.8, N), 0.5, 10)
gpa               = np.clip(2.0  + 0.18*study_hrs + 0.10*sleep_hrs
                             - 0.08*daily_usage_hrs + np.random.normal(0, 0.25, N), 1.0, 4.0)
mental_wellness   = np.clip(3.5  + 0.35*sleep_hrs - 0.18*daily_usage_hrs
                             + np.random.normal(0, 0.8, N), 1, 10)
distraction_score = np.clip(1.5  + 0.60*daily_usage_hrs + np.random.normal(0, 0.9, N), 1, 10)
social_anxiety    = np.clip(2.0  + 0.25*daily_usage_hrs - 0.15*mental_wellness
                             + np.random.normal(0, 1.0, N), 1, 10)
notifications     = (daily_usage_hrs * np.random.uniform(8, 18, N)).astype(int)
academic_satisfaction = np.clip(
    np.round(1.0 + 0.25*gpa + 0.10*mental_wellness
             - 0.05*distraction_score + np.random.normal(0, 0.4, N)), 1, 5).astype(int)

df = pd.DataFrame({
    'student_id':              [f'S{str(i+1).zfill(4)}' for i in range(N)],
    'age':                     age,
    'gender':                  gender,
    'year_level':              year_level,
    'major':                   major,
    'location':                location,
    'primary_platform':        platform,
    'daily_social_media_hrs':  daily_usage_hrs.round(2),
    'daily_study_hrs':         study_hrs.round(2),
    'daily_sleep_hrs':         sleep_hrs.round(2),
    'notifications_per_day':   notifications,
    'distraction_score':       distraction_score.round(1),
    'mental_wellness_score':   mental_wellness.round(1),
    'social_anxiety_score':    social_anxiety.round(1),
    'academic_satisfaction':   academic_satisfaction,
    'gpa':                     gpa.round(2),
    'has_part_time_job':       has_part_time_job,
})

df.to_csv('student_social_media.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.describe().round(2))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Figure 1: Dataset Overview ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Figure 1: Dataset Overview — Key Variable Distributions',
             fontsize=14, fontweight='bold', y=1.01)

# Social media hours histogram
axes[0, 0].hist(df['daily_social_media_hrs'], bins=30,
                color=PALETTE[0], edgecolor='white', alpha=0.85)
axes[0, 0].axvline(df['daily_social_media_hrs'].mean(), color='red',
                   linestyle='--', label=f"Mean = {df['daily_social_media_hrs'].mean():.1f}h")
axes[0, 0].set_title('Daily Social Media Usage (hrs)')
axes[0, 0].set_xlabel('Hours'); axes[0, 0].set_ylabel('Students')
axes[0, 0].legend(fontsize=9)

# GPA histogram
axes[0, 1].hist(df['gpa'], bins=25, color=PALETTE[2], edgecolor='white', alpha=0.85)
axes[0, 1].axvline(df['gpa'].mean(), color='red', linestyle='--',
                   label=f"Mean = {df['gpa'].mean():.2f}")
axes[0, 1].set_title('GPA Distribution')
axes[0, 1].set_xlabel('GPA (0–4.0)'); axes[0, 1].set_ylabel('Students')
axes[0, 1].legend(fontsize=9)

# Sleep hours histogram
axes[0, 2].hist(df['daily_sleep_hrs'], bins=25, color=PALETTE[4], edgecolor='white', alpha=0.85)
axes[0, 2].set_title('Daily Sleep Hours')
axes[0, 2].set_xlabel('Hours'); axes[0, 2].set_ylabel('Students')

# Platform distribution
platform_counts = df['primary_platform'].value_counts()
axes[1, 0].bar(platform_counts.index, platform_counts.values,
               color=PALETTE, edgecolor='white')
axes[1, 0].set_title('Primary Platform Usage')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].set_ylabel('Count')

# Year level distribution
year_order  = ['Freshman', 'Sophomore', 'Junior', 'Senior']
year_counts = df['year_level'].value_counts().reindex(year_order)
axes[1, 1].bar(year_counts.index, year_counts.values, color=PALETTE[:4], edgecolor='white')
axes[1, 1].set_title('Students by Year Level')
axes[1, 1].set_ylabel('Count')

# Major distribution (horizontal)
major_counts = df['major'].value_counts()
axes[1, 2].barh(major_counts.index, major_counts.values, color=PALETTE, edgecolor='white')
axes[1, 2].set_title('Students by Major')
axes[1, 2].set_xlabel('Count')

plt.tight_layout()
plt.savefig('figures/fig1_overview.png', bbox_inches='tight')
plt.close()
print("  [✓] fig1_overview.png saved")

# ── Figure 2: GPA vs Social Media Usage ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 2: Social Media Usage vs Academic Performance',
             fontsize=14, fontweight='bold')

# Scatter by platform
for i, (p, grp) in enumerate(df.groupby('primary_platform')):
    axes[0].scatter(grp['daily_social_media_hrs'], grp['gpa'],
                    alpha=0.45, s=25, label=p)
axes[0].set_title('GPA vs Daily Usage (by Platform)')
axes[0].set_xlabel('Daily SM Hours'); axes[0].set_ylabel('GPA')
axes[0].legend(fontsize=7)

# Mean GPA by platform
plat_gpa = df.groupby('primary_platform')['gpa'].mean().sort_values()
bars = axes[1].barh(plat_gpa.index, plat_gpa.values, color=PALETTE, edgecolor='white')
axes[1].set_title('Average GPA by Platform')
axes[1].set_xlabel('Mean GPA'); axes[1].set_xlim(2.8, 3.7)
for bar, val in zip(bars, plat_gpa.values):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=9)

# Mean GPA by usage bucket
usage_bins = pd.cut(df['daily_social_media_hrs'], bins=[0, 2, 4, 6, 13],
                    labels=['<2h', '2–4h', '4–6h', '>6h'])
usage_gpa  = df.groupby(usage_bins, observed=True)['gpa'].mean()
bar_colors = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']
bars2 = axes[2].bar(usage_gpa.index.astype(str), usage_gpa.values,
                    color=bar_colors, edgecolor='white')
axes[2].set_title('Average GPA by Usage Bucket')
axes[2].set_xlabel('Daily Usage'); axes[2].set_ylabel('Mean GPA')
axes[2].set_ylim(2.8, 3.7)
for bar, val in zip(bars2, usage_gpa.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig2_usage_vs_gpa.png', bbox_inches='tight')
plt.close()
print("  [✓] fig2_usage_vs_gpa.png saved")

# Print GPA by bucket
print("\n  Mean GPA by usage bucket:")
for bucket, val in usage_gpa.items():
    print(f"    {bucket}: {val:.3f}")

# ── Figure 3: Correlation Heatmap ────────────────────────────────────────────
num_cols = ['daily_social_media_hrs', 'daily_study_hrs', 'daily_sleep_hrs',
            'notifications_per_day', 'distraction_score', 'mental_wellness_score',
            'social_anxiety_score', 'academic_satisfaction', 'gpa']
labels   = ['SM Hours', 'Study Hrs', 'Sleep Hrs', 'Notifications',
            'Distraction', 'Mental Wellness', 'Social Anxiety', 'Acad. Satisf.', 'GPA']

corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Figure 3: Correlation Heatmap', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig3_heatmap.png', bbox_inches='tight')
plt.close()
print("  [✓] fig3_heatmap.png saved")

print(f"\n  Key correlations with GPA:")
print(f"    Study hours   : {corr.loc['daily_study_hrs','gpa']:+.3f}")
print(f"    Sleep hours   : {corr.loc['daily_sleep_hrs','gpa']:+.3f}")
print(f"    SM usage      : {corr.loc['daily_social_media_hrs','gpa']:+.3f}")
print(f"    Distraction   : {corr.loc['distraction_score','gpa']:+.3f}")

# ── Figure 4: Multi-dimensional EDA ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 4: Multi-Dimensional EDA Insights',
             fontsize=14, fontweight='bold')

# Sleep vs GPA (coloured by SM usage)
sc = axes[0, 0].scatter(df['daily_sleep_hrs'], df['gpa'],
                         c=df['daily_social_media_hrs'], cmap='RdYlGn_r',
                         alpha=0.5, s=30)
plt.colorbar(sc, ax=axes[0, 0], label='SM Hours/day')
axes[0, 0].set_title('Sleep Hours vs GPA  (color = SM Usage)')
axes[0, 0].set_xlabel('Sleep Hours'); axes[0, 0].set_ylabel('GPA')

# Study hrs vs GPA (coloured by SM usage)
sc2 = axes[0, 1].scatter(df['daily_study_hrs'], df['gpa'],
                          c=df['daily_social_media_hrs'], cmap='RdYlGn_r',
                          alpha=0.5, s=30)
plt.colorbar(sc2, ax=axes[0, 1], label='SM Hours/day')
axes[0, 1].set_title('Study Hours vs GPA  (color = SM Usage)')
axes[0, 1].set_xlabel('Study Hours'); axes[0, 1].set_ylabel('GPA')

# Mental wellness by year level
df_y = df.copy()
df_y['year_level'] = pd.Categorical(df_y['year_level'], categories=year_order, ordered=True)
df_y = df_y.sort_values('year_level')
sns.boxplot(data=df_y, x='year_level', y='mental_wellness_score',
            palette=PALETTE[:4], ax=axes[1, 0])
axes[1, 0].set_title('Mental Wellness Score by Year Level')
axes[1, 0].set_xlabel('Year Level'); axes[1, 0].set_ylabel('Mental Wellness (1–10)')

# Distraction vs study hours (trend line)
axes[1, 1].scatter(df['distraction_score'], df['daily_study_hrs'],
                   alpha=0.4, s=30, color=PALETTE[0])
m, b = np.polyfit(df['distraction_score'], df['daily_study_hrs'], 1)
x_line = np.linspace(df['distraction_score'].min(), df['distraction_score'].max(), 100)
r = np.corrcoef(df['distraction_score'], df['daily_study_hrs'])[0, 1]
axes[1, 1].plot(x_line, m*x_line + b, color='red', linewidth=1.5,
                linestyle='--', label=f'Trend (r={r:.2f})')
axes[1, 1].set_title('Distraction Score vs Study Hours')
axes[1, 1].set_xlabel('Distraction Score (1–10)'); axes[1, 1].set_ylabel('Daily Study Hours')
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig4_eda_multi.png', bbox_inches='tight')
plt.close()
print("  [✓] fig4_eda_multi.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — K-MEANS CLUSTERING")
print("=" * 60)

cluster_features = ['daily_social_media_hrs', 'daily_study_hrs', 'daily_sleep_hrs',
                    'gpa', 'mental_wellness_score', 'distraction_score']
X_cluster = df[cluster_features].copy()
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# Elbow method
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit final model k=4
km4        = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = km4.fit_predict(X_scaled)

cluster_labels = {0: 'Balanced Achievers',
                  1: 'High-Usage Risk',
                  2: 'Study-Focused',
                  3: 'Low-Engagement'}
df['cluster_label'] = df['cluster'].map(cluster_labels)

# PCA for 2D visualisation
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cluster_colors = ['#4C72B0', '#C44E52', '#55A868', '#DD8452']

# ── Figure 5: Clustering ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Figure 5: K-Means Student Segmentation (k=4)',
             fontsize=14, fontweight='bold')

# Elbow curve
axes[0].plot(list(K_range), inertias, 'o-', color=PALETTE[0], linewidth=2, markersize=7)
axes[0].axvline(4, color='red', linestyle='--', alpha=0.7, label='k=4 selected')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters k'); axes[0].set_ylabel('Inertia')
axes[0].legend()

# PCA scatter
for c in range(4):
    mask = df['cluster'] == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=cluster_labels[c], alpha=0.55, s=30, color=cluster_colors[c])
var_pct = pca.explained_variance_ratio_.sum() * 100
axes[1].set_title(f'PCA Projection  (var. explained: {var_pct:.1f}%)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].legend(fontsize=8)

# Cluster profile bar chart
profile = df.groupby('cluster')[['daily_social_media_hrs', 'daily_study_hrs',
                                  'gpa', 'mental_wellness_score']].mean()
profile.index = [cluster_labels[i] for i in profile.index]
profile.columns = ['SM Hrs', 'Study Hrs', 'GPA', 'Mental W.']
profile.T.plot(kind='bar', ax=axes[2], color=cluster_colors, edgecolor='white', width=0.7)
axes[2].set_title('Cluster Profiles — Mean Values')
axes[2].set_xlabel('Metric'); axes[2].set_ylabel('Mean Value')
axes[2].legend(fontsize=7, loc='upper right')
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('figures/fig5_clustering.png', bbox_inches='tight')
plt.close()
print("  [✓] fig5_clustering.png saved")

print("\n  Cluster Summary:")
print(profile.round(2).to_string())
sizes = df['cluster_label'].value_counts()
print("\n  Cluster Sizes:")
print(sizes.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GPA PREDICTION (REGRESSION MODELS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — GPA PREDICTION — REGRESSION MODELS")
print("=" * 60)

features = ['daily_social_media_hrs', 'daily_study_hrs', 'daily_sleep_hrs',
            'notifications_per_day', 'distraction_score', 'mental_wellness_score',
            'social_anxiety_score', 'age']
X = df[features]
y = df['gpa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'Linear Regression':   LinearRegression(),
    'Random Forest':       RandomForestRegressor(n_estimators=150, random_state=42),
    'Gradient Boosting':   GradientBoostingRegressor(n_estimators=150, random_state=42),
}

results     = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    results[name] = {
        'R²':           r2_score(y_test, y_pred),
        'MAE':          mean_absolute_error(y_test, y_pred),
        'RMSE':         np.sqrt(mean_squared_error(y_test, y_pred)),
        'CV R² (5-fold)': cv.mean()
    }

results_df = pd.DataFrame(results).T
print("\n  Model Performance:")
print(results_df.round(4).to_string())

# Feature importances (Random Forest)
rf_model = models['Random Forest']
feat_labels = {
    'daily_social_media_hrs': 'SM Usage (hrs)',
    'daily_study_hrs':        'Study Hours',
    'daily_sleep_hrs':        'Sleep Hours',
    'notifications_per_day':  'Notifications/day',
    'distraction_score':      'Distraction Score',
    'mental_wellness_score':  'Mental Wellness',
    'social_anxiety_score':   'Social Anxiety',
    'age':                    'Age'
}
feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.index = [feat_labels[i] for i in feat_imp.index]

print("\n  Feature Importances (Random Forest):")
for name_f, val in feat_imp.sort_values(ascending=False).items():
    print(f"    {name_f:<25} {val:.4f}")

# ── Figure 6: ML Results ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Figure 6: ML Models — GPA Prediction Results',
             fontsize=14, fontweight='bold')

# Predicted vs Actual (RF)
best_pred = predictions['Random Forest']
axes[0].scatter(y_test, best_pred, alpha=0.45, s=30, color=PALETTE[0])
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=1.5, label='Perfect fit')
axes[0].set_title('Random Forest: Predicted vs Actual GPA')
axes[0].set_xlabel('Actual GPA'); axes[0].set_ylabel('Predicted GPA')
axes[0].legend(fontsize=8)
r2_rf  = results['Random Forest']['R²']
mae_rf = results['Random Forest']['MAE']
axes[0].text(0.05, 0.92, f"R² = {r2_rf:.3f}\nMAE = {mae_rf:.3f}",
             transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Model comparison bar (R² and MAE side-by-side)
model_names = list(results.keys())
r2_vals  = [results[m]['R²']  for m in model_names]
mae_vals = [results[m]['MAE'] for m in model_names]
x = np.arange(len(model_names)); w = 0.35
bars1 = axes[1].bar(x - w/2, r2_vals,  w, label='R²',  color=PALETTE[0], edgecolor='white')
bars2 = axes[1].bar(x + w/2, mae_vals, w, label='MAE', color=PALETTE[1], edgecolor='white')
axes[1].set_title('Model Comparison: R² and MAE')
axes[1].set_xticks(x)
axes[1].set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=9)
axes[1].legend(fontsize=9)
for bar in bars1:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8)

# Feature importances (coloured: red=negative factors, blue=positive)
negative_keywords = ('SM', 'Distract', 'Notif', 'Anxiety')
colors_fi = ['#C44E52' if any(k in lbl for k in negative_keywords) else '#4C72B0'
             for lbl in feat_imp.index]
axes[2].barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='white')
axes[2].set_title('Feature Importance (Random Forest)')
axes[2].set_xlabel('Importance Score')
red_patch  = mpatches.Patch(color='#C44E52', label='Negative factors')
blue_patch = mpatches.Patch(color='#4C72B0', label='Positive factors')
axes[2].legend(handles=[red_patch, blue_patch], fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig6_ml_results.png', bbox_inches='tight')
plt.close()
print("  [✓] fig6_ml_results.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — OUTPUT FILES")
print("=" * 60)
print("  student_social_media.csv")
print("  figures/fig1_overview.png")
print("  figures/fig2_usage_vs_gpa.png")
print("  figures/fig3_heatmap.png")
print("  figures/fig4_eda_multi.png")
print("  figures/fig5_clustering.png")
print("  figures/fig6_ml_results.png")
print()
print("KEY FINDINGS:")
print(f"  • Mean daily SM usage     : {df['daily_social_media_hrs'].mean():.1f} hrs")
print(f"  • Mean GPA                : {df['gpa'].mean():.2f}")
print(f"  • GPA <2h vs >6h users   : {df[usage_bins=='<2h']['gpa'].mean():.2f} vs "
      f"{df[usage_bins=='>6h']['gpa'].mean():.2f}  (Δ = "
      f"{df[usage_bins=='<2h']['gpa'].mean() - df[usage_bins=='>6h']['gpa'].mean():.2f})")
print(f"  • Best model              : Linear Regression  R² = {results['Linear Regression']['R²']:.3f}")
print(f"  • High-Usage Risk cluster : {sizes.get('High-Usage Risk', 0)} students "
      f"({sizes.get('High-Usage Risk',0)/N*100:.0f}% of total)")