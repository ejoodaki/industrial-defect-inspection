from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/final_test_error_audit/all_test_errors_quantitative.csv"
)

OUTPUT_DIR = Path("outputs/final_test_error_audit")

SUMMARY_CSV = OUTPUT_DIR / "error_distribution_summary.csv"
DENSITY_CSV = OUTPUT_DIR / "error_distribution_density_bins.csv"
COMPONENT_CSV = OUTPUT_DIR / "error_component_ratio_bins.csv"
DIAGNOSIS_CSV = OUTPUT_DIR / "error_diagnosis_summary.csv"
REPORT_TXT = OUTPUT_DIR / "error_distribution_report.txt"


PREDICTION_FEATURES = [
    "rf_probability",
    "max_probability",
    "prediction_pixels",
    "prediction_density",
    "prediction_component_count",
    "prediction_largest_component_area",
    "prediction_largest_component_width",
    "prediction_largest_component_height",
    "prediction_bbox_width",
    "prediction_bbox_height",
    "prediction_bbox_area",
    "prediction_bbox_aspect_ratio",
]

SEGMENTATION_QUALITY_FEATURES = [
    "pixel_precision",
    "pixel_recall",
    "pixel_iou",
    "pixel_dice",
]

GT_FEATURES = [
    "gt_pixels",
    "gt_density",
    "gt_component_count",
    "gt_largest_component_area",
    "gt_bbox_width",
    "gt_bbox_height",
    "gt_bbox_area",
    "gt_bbox_aspect_ratio",
]


def validate_schema(df):
    required = (
        ["image_id", "category", "diagnosis"]
        + PREDICTION_FEATURES
        + SEGMENTATION_QUALITY_FEATURES
        + GT_FEATURES
    )

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def summarize_group(df, group_name, features):
    rows = []

    for feature in features:
        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "group": group_name,
                "feature": feature,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }
        )

    return rows


def make_density_bins(df):
    bins = [0.0, 0.10, 0.25, 0.50, 0.75, 1.000001]

    labels = [
        "0-10%",
        "10-25%",
        "25-50%",
        "50-75%",
        "75-100%",
    ]

    rows = []

    for category in ["FP", "FN"]:
        subset = df[df["category"] == category].copy()

        subset["density_bin"] = pd.cut(
            subset["prediction_density"],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True,
        )

        counts = (
            subset["density_bin"]
            .value_counts(sort=False)
            .rename_axis("density_bin")
            .reset_index(name="count")
        )

        total = len(subset)

        counts["category"] = category
        counts["fraction"] = (
            counts["count"] / total if total else 0
        )

        rows.append(counts)

    return pd.concat(rows, ignore_index=True)


def make_component_ratio_bins(df):
    work = df.copy()

    work["largest_component_ratio"] = np.where(
        work["prediction_pixels"] > 0,
        work["prediction_largest_component_area"]
        / work["prediction_pixels"],
        0.0,
    )

    bins = [0.0, 0.25, 0.50, 0.75, 0.90, 1.000001]

    labels = [
        "0-25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "90-100%",
    ]

    work["component_ratio_bin"] = pd.cut(
        work["largest_component_ratio"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    grouped = (
        work.groupby(
            ["category", "component_ratio_bin"],
            observed=False,
        )
        .size()
        .reset_index(name="count")
    )

    totals = (
        work.groupby("category")
        .size()
        .rename("total")
        .reset_index()
    )

    grouped = grouped.merge(
        totals,
        on="category",
        how="left",
    )

    grouped["fraction"] = (
        grouped["count"] / grouped["total"]
    )

    return grouped


def make_diagnosis_summary(df):
    result = (
        df.groupby(
            ["category", "diagnosis"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.groupby("category")
        .size()
        .rename("total")
        .reset_index()
    )

    result = result.merge(
        totals,
        on="category",
        how="left",
    )

    result["fraction"] = (
        result["count"] / result["total"]
    )

    return result


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    validate_schema(df)

    fp = df[df["category"] == "FP"].copy()
    fn = df[df["category"] == "FN"].copy()

    print("=" * 70)
    print("FINAL TEST ERROR DISTRIBUTION AUDIT")
    print("=" * 70)

    print(f"Input: {INPUT_CSV}")
    print(f"Total errors: {len(df)}")
    print(f"FP: {len(fp)}")
    print(f"FN: {len(fn)}")

    # ---------------------------------------------------------
    # 1. Feature summaries
    # ---------------------------------------------------------

    summary_rows = []

    for feature_group in [
        PREDICTION_FEATURES,
        SEGMENTATION_QUALITY_FEATURES,
        GT_FEATURES,
    ]:
        summary_rows.extend(
            summarize_group(
                fp,
                "FP",
                feature_group,
            )
        )

        summary_rows.extend(
            summarize_group(
                fn,
                "FN",
                feature_group,
            )
        )

        summary_rows.extend(
            summarize_group(
                df,
                "ALL_ERRORS",
                feature_group,
            )
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Prediction-density bins
    # ---------------------------------------------------------

    density = make_density_bins(df)

    density.to_csv(
        DENSITY_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Largest-component concentration
    # ---------------------------------------------------------

    component = make_component_ratio_bins(df)

    component.to_csv(
        COMPONENT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Diagnosis summary
    # ---------------------------------------------------------

    diagnosis = make_diagnosis_summary(df)

    diagnosis.to_csv(
        DIAGNOSIS_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Text report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 70)
    lines.append("FINAL TEST ERROR DISTRIBUTION AUDIT")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Total errors: {len(df)}")
    lines.append(f"False Positives: {len(fp)}")
    lines.append(f"False Negatives: {len(fn)}")
    lines.append("")

    # ---------------------------------------------------------
    # FP vs FN
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("FP vs FN — MEDIAN COMPARISON")
    lines.append("-" * 70)

    for feature in PREDICTION_FEATURES:
        fp_median = fp[feature].median()
        fn_median = fn[feature].median()

        lines.append(
            f"{feature:40s} "
            f"FP={fp_median:12.4f} "
            f"FN={fn_median:12.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Segmentation quality
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("SEGMENTATION QUALITY")
    lines.append("-" * 70)

    for feature in SEGMENTATION_QUALITY_FEATURES:
        fp_median = fp[feature].median()
        fn_median = fn[feature].median()

        lines.append(
            f"{feature:25s} "
            f"FP={fp_median:12.4f} "
            f"FN={fn_median:12.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Density distribution
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("PREDICTION DENSITY DISTRIBUTION")
    lines.append("-" * 70)

    for category in ["FP", "FN"]:
        lines.append("")
        lines.append(category)

        subset = density[
            density["category"] == category
        ]

        for _, row in subset.iterrows():
            lines.append(
                f"  {str(row['density_bin']):>8} : "
                f"{int(row['count']):3d} "
                f"({pct(row['fraction'])})"
            )

    lines.append("")

    # ---------------------------------------------------------
    # Component concentration
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("LARGEST COMPONENT / PREDICTION RATIO")
    lines.append("-" * 70)

    for category in ["FP", "FN"]:
        lines.append("")
        lines.append(category)

        subset = component[
            component["category"] == category
        ]

        for _, row in subset.iterrows():
            lines.append(
                f"  {str(row['component_ratio_bin']):>8} : "
                f"{int(row['count']):3d} "
                f"({pct(row['fraction'])})"
            )

    lines.append("")

    # ---------------------------------------------------------
    # Diagnosis
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("DIAGNOSIS SUMMARY")
    lines.append("-" * 70)

    for _, row in diagnosis.iterrows():
        lines.append(
            f"{row['category']:>3} | "
            f"{int(row['count']):3d} | "
            f"{pct(row['fraction']):>8} | "
            f"{row['diagnosis']}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # High-density errors
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("HIGHEST PREDICTION DENSITY ERRORS")
    lines.append("-" * 70)

    high_density = (
        df.sort_values(
            "prediction_density",
            ascending=False,
        )
        .head(20)
    )

    for _, row in high_density.iterrows():
        lines.append(
            f"{row['image_id']} "
            f"{row['category']} "
            f"density={row['prediction_density']:.4f} "
            f"pixels={int(row['prediction_pixels'])} "
            f"rf={row['rf_probability']:.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Highest-confidence FP
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("HIGHEST RF-SCORE FP CASES")
    lines.append("-" * 70)

    high_fp = (
        fp.sort_values(
            "rf_probability",
            ascending=False,
        )
        .head(20)
    )

    for _, row in high_fp.iterrows():
        lines.append(
            f"{row['image_id']} "
            f"rf={row['rf_probability']:.4f} "
            f"density={row['prediction_density']:.4f} "
            f"pixels={int(row['prediction_pixels'])} "
            f"dice={row['pixel_dice']:.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Lowest-confidence FN
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("LOWEST RF-SCORE FN CASES")
    lines.append("-" * 70)

    low_fn = (
        fn.sort_values(
            "rf_probability",
            ascending=True,
        )
        .head(20)
    )

    for _, row in low_fn.iterrows():
        lines.append(
            f"{row['image_id']} "
            f"rf={row['rf_probability']:.4f} "
            f"density={row['prediction_density']:.4f} "
            f"pixels={int(row['prediction_pixels'])} "
            f"recall={row['pixel_recall']:.4f} "
            f"dice={row['pixel_dice']:.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Interpretation guardrails
    # ---------------------------------------------------------

    lines.append("-" * 70)
    lines.append("INTERPRETATION GUARDRAILS")
    lines.append("-" * 70)

    lines.append(
        "This is a descriptive audit of the frozen final-test errors."
    )

    lines.append(
        "No model was retrained and no test threshold was tuned."
    )

    lines.append(
        "Distribution differences are hypotheses, not validated fixes."
    )

    lines.append(
        "Any intervention must be evaluated using train/validation "
        "data before another final-test evaluation."
    )

    REPORT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"Summary:   {SUMMARY_CSV}")
    print(f"Density:   {DENSITY_CSV}")
    print(f"Component: {COMPONENT_CSV}")
    print(f"Diagnosis: {DIAGNOSIS_CSV}")
    print(f"Report:    {REPORT_TXT}")


if __name__ == "__main__":
    main()