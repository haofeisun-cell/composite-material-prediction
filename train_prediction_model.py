import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "保温温度1",
    "保温时间1",
    "保温温度2",
    "保温时间2",
    "保温温度3",
    "保温时间3",
    "烧结温度6",
    "烧结时间6",
    "加压压力2",
    "加压温度2",
    "金刚石粒径",
]

TARGET_COLUMNS = ["致密度", "热导率"]
THERMAL_CONDUCTIVITY_TOLERANCE = 20.0


def make_selector(random_state: int) -> SelectFromModel:
    return SelectFromModel(
        estimator=ExtraTreesRegressor(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
        ),
        threshold="median",
    )


def make_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("只支持 .csv、.xlsx、.xls 数据表")


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"数据表缺少列: {missing}")


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            continue

        text = cleaned[col].astype(str).str.strip()
        text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "无": np.nan, "-": np.nan, "—": np.nan})
        text = text.str.replace(",", "", regex=False)
        text = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        cleaned[col] = pd.to_numeric(text, errors="coerce")
    return cleaned


def validate_sample_count(x: pd.DataFrame, y: pd.DataFrame, raw_rows: int) -> None:
    valid_targets = ~y.isna().any(axis=1)
    valid_rows = int(valid_targets.sum())
    if valid_rows == 0:
        target_missing = y.isna().sum().to_dict()
        raise ValueError(
            "清洗后没有可用于训练的样本。请检查 `致密度` 和 `热导率` 两列是否有数值。"
            f" 原始行数: {raw_rows}; 目标列无法识别/缺失数量: {target_missing}"
        )
    if valid_rows < 10:
        raise ValueError(
            "有效样本少于 10 行，无法稳定按 8:1:1 划分训练集、验证集和测试集。"
            f" 当前有效样本数: {valid_rows}"
        )


def build_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "extra_trees": Pipeline(
            steps=[
                ("imputer", make_imputer()),
                ("selector", make_selector(random_state)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=800,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", make_imputer()),
                ("selector", make_selector(random_state)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=800,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("imputer", make_imputer()),
                ("selector", make_selector(random_state)),
                (
                    "model",
                    MultiOutputRegressor(
                        XGBRegressor(
                            n_estimators=2000,
                            learning_rate=0.03,
                            max_depth=3,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
    }


def get_target_tolerance(target: str, y_true: pd.Series) -> float:
    if target == "致密度":
        return 0.02 if y_true.max() <= 1.5 else 2.0
    if target == "热导率":
        return THERMAL_CONDUCTIVITY_TOLERANCE
    raise ValueError(f"未知目标列: {target}")


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.DataFrame) -> dict:
    pred = model.predict(x)
    metrics = {
        "overall": {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "pass_rate": 0.0,
        },
        "by_target": {},
    }
    pass_rates = []
    for idx, target in enumerate(TARGET_COLUMNS):
        error = np.abs(y.iloc[:, idx].to_numpy() - pred[:, idx])
        tolerance = get_target_tolerance(target, y.iloc[:, idx])
        pass_rate = float(np.mean(error <= tolerance))
        pass_rates.append(pass_rate)
        metrics["by_target"][target] = {
            "mae": float(mean_absolute_error(y.iloc[:, idx], pred[:, idx])),
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[:, idx], pred[:, idx]))),
            "tolerance": tolerance,
            "pass_rate": pass_rate,
        }
    metrics["overall"]["pass_rate"] = float(np.mean(pass_rates))
    return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    for target in TARGET_COLUMNS:
        target_metrics = metrics["by_target"][target]
        print(
            f"{target}达标率: {target_metrics['pass_rate'] * 100:.2f}% | "
            f"达标误差阈值: {target_metrics['tolerance']:.4f} | "
            f"MAE: {target_metrics['mae']:.4f} | "
            f"RMSE: {target_metrics['rmse']:.4f}"
        )
    print(f"整体达标率: {metrics['overall']['pass_rate'] * 100:.2f}%")


def get_selected_features(model: Pipeline) -> list[str]:
    imputer = model.named_steps["imputer"]
    selector = model.named_steps["selector"]

    feature_names = list(FEATURE_COLUMNS)
    if getattr(imputer, "indicator_", None) is not None:
        missing_indices = imputer.indicator_.features_
        feature_names.extend([f"{FEATURE_COLUMNS[i]}_是否缺失" for i in missing_indices])

    selected_mask = selector.get_support()
    return [name for name, selected in zip(feature_names, selected_mask) if selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="训练致密度和热导率预测模型")
    parser.add_argument("--data", required=True, help="数据表路径，支持 csv/xlsx/xls")
    parser.add_argument("--output-dir", default="model_output", help="模型和报告输出目录")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(data_path)
    df.columns = df.columns.astype(str).str.strip()
    validate_columns(df)

    # 目标值缺失的样本无法监督训练；特征缺失用 0 填补，并保留缺失指示列。
    raw_rows = len(df)
    x = clean_numeric_frame(df[FEATURE_COLUMNS])
    y = clean_numeric_frame(df[TARGET_COLUMNS])
    validate_sample_count(x, y, raw_rows)
    keep = ~y.isna().any(axis=1)
    x = x.loc[keep]
    y = y.loc[keep]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=args.random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=1 / 9,
        random_state=args.random_state,
    )

    candidates = build_candidates(args.random_state)
    best_name = None
    best_score = -np.inf
    best_model = None
    best_val_metrics = None
    candidate_scores = []

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        val_metrics = evaluate(model, x_val, y_val)
        score = val_metrics["overall"]["pass_rate"]
        candidate_scores.append((name, score))
        if score > best_score:
            best_name = name
            best_score = score
            best_model = model
            best_val_metrics = val_metrics

    assert best_name is not None and best_model is not None and best_val_metrics is not None

    test_metrics = evaluate(best_model, x_test, y_test)
    selected_features = get_selected_features(best_model)

    joblib.dump(best_model, output_dir / "best_model.joblib")

    print(f"最佳模型: {best_name}")
    print(f"数据划分: 训练集 {len(x_train)} 行, 验证集 {len(x_val)} 行, 测试集 {len(x_test)} 行")
    print(f"自动选择的特征: {', '.join(selected_features)}")
    print("\n候选模型验证集达标率:")
    for name, score in sorted(candidate_scores, key=lambda item: item[1], reverse=True):
        print(f"{name}: {score * 100:.2f}%")
    print_metrics("验证集达标率", best_val_metrics)
    print_metrics("测试集达标率", test_metrics)
    print(f"\n模型已保存到: {output_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
