import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm
from xgboost import XGBRegressor


TITLE_COLUMN = "文献标题"
TARGET_COLUMNS = ["致密度", "热导率"]
DENSITY_PERCENT_TOLERANCE = 1.0
DENSITY_DECIMAL_TOLERANCE = 0.01
THERMAL_CONDUCTIVITY_TOLERANCE = 30.0

PARAM_GRIDS: dict[str, dict] = {
    "extra_trees": {
        "model__n_estimators": [500, 1000],
        "model__max_depth": [20, 50, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5],
    },
    "random_forest": {
        "model__n_estimators": [500, 800],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5],
    },
    "xgboost": {
        "model__n_estimators": [500, 1000, 1500],
        "model__learning_rate": [0.01, 0.03, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9],
    },
}


def make_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", make_imputer(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=True,
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("只支持 .csv、.xlsx、.xls 数据表")


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    return [col for col in df.columns if col not in {TITLE_COLUMN, target}]


def get_input_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    feature_columns = get_feature_columns(df, target)
    return feature_columns, []


def validate_columns(df: pd.DataFrame, target: str) -> None:
    if TITLE_COLUMN not in df.columns:
        raise ValueError(f"数据表缺少列: [{TITLE_COLUMN}]")
    if target not in df.columns:
        raise ValueError(f"数据表缺少列: [{target}]")
    if not get_feature_columns(df, target):
        raise ValueError("除文献标题和目标列外，未找到可用特征列")


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            continue

        text = cleaned[col].astype("string").str.strip()
        no_feature_mask = text.eq("无").fillna(False)
        missing_mask = text.isin(["", "nan", "None", "-", "—", "缺失"])
        text = text.mask(missing_mask)
        text = text.str.replace(",", "", regex=False)
        text = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        cleaned[col] = pd.to_numeric(text, errors="coerce")
        cleaned.loc[no_feature_mask, col] = 0
    return cleaned


def clean_categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        text = cleaned[col].astype("string").str.strip()
        missing_mask = text.isin(["", "nan", "None", "-", "—", "缺失"])
        text = text.mask(missing_mask, "缺失")
        cleaned[col] = text.fillna("缺失").astype(str)
    return cleaned


def build_feature_frame(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    numeric_df = clean_numeric_frame(df[numeric_columns]) if numeric_columns else pd.DataFrame(index=df.index)
    categorical_df = (
        clean_categorical_frame(df[categorical_columns])
        if categorical_columns
        else pd.DataFrame(index=df.index)
    )
    return pd.concat([numeric_df, categorical_df], axis=1)


def validate_sample_count(y: pd.Series, raw_rows: int, target: str) -> None:
    valid_targets = ~y.isna()
    valid_rows = int(valid_targets.sum())
    if valid_rows == 0:
        raise ValueError(
            f"清洗后没有可用于训练 `{target}` 的样本。请检查该列是否有数值。"
            f" 原始行数: {raw_rows}; `{target}` 无法识别/缺失数量: {int(y.isna().sum())}"
        )
    if valid_rows < 10:
        raise ValueError(
            "有效样本少于 10 行，无法稳定按 6:2:2 划分训练集、验证集和测试集。"
            f" 当前有效样本数: {valid_rows}"
        )


def build_base_pipeline(
    name: str,
    random_state: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    if name == "extra_trees":
        model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    elif name == "random_forest":
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif name == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"未知模型: {name}")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_target_tolerance(target: str, y_true: pd.Series) -> float:
    if target == "致密度":
        return DENSITY_DECIMAL_TOLERANCE if y_true.max() <= 1.5 else DENSITY_PERCENT_TOLERANCE
    if target == "热导率":
        return THERMAL_CONDUCTIVITY_TOLERANCE
    raise ValueError(f"未知目标列: {target}")


def make_pass_rate_scorer(target: str):
    def pass_rate(y_true, y_pred) -> float:
        tolerance = get_target_tolerance(target, pd.Series(y_true))
        return float(np.mean(np.abs(y_true - y_pred) <= tolerance))

    return make_scorer(pass_rate, greater_is_better=True)


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series, target: str) -> dict:
    pred = model.predict(x)
    error = np.abs(y.to_numpy() - pred)
    tolerance = get_target_tolerance(target, y)
    pass_rate = float(np.mean(error <= tolerance))
    metrics = {
        "target": target,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "tolerance": tolerance,
        "pass_rate": pass_rate,
    }
    return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(
        f"{metrics['target']}达标率: {metrics['pass_rate'] * 100:.2f}% | "
        f"达标误差阈值: {metrics['tolerance']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f}"
    )


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = []

    for name in preprocessor.get_feature_names_out():
        if name.startswith("numeric__missingindicator_"):
            feature_names.append(f"{name.replace('numeric__missingindicator_', '')}_是否缺失")
        elif name.startswith("numeric__"):
            feature_names.append(name.replace("numeric__", ""))
        elif name.startswith("categorical__"):
            feature_names.append(name.replace("categorical__", ""))
        else:
            feature_names.append(name)

    return feature_names


def tune_candidate(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    random_state: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[Pipeline, dict, float]:
    pipeline = build_base_pipeline(name, random_state, numeric_columns, categorical_columns)
    cv_folds = min(5, len(x_train))
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS[name],
        scoring=make_pass_rate_scorer(target),
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练单目标预测模型")
    parser.add_argument("--data", required=True, help="数据表路径，支持 csv/xlsx/xls")
    parser.add_argument("--target", required=True, choices=TARGET_COLUMNS, help="预测目标：致密度 或 热导率")
    parser.add_argument("--output-dir", default="model_output", help="模型和报告输出目录")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(data_path)
    df.columns = df.columns.astype(str).str.strip()
    validate_columns(df, args.target)
    numeric_columns, categorical_columns = get_input_columns(df, args.target)

    raw_rows = len(df)
    x = build_feature_frame(df, numeric_columns, categorical_columns)
    y = clean_numeric_frame(df[[args.target]])[args.target]
    validate_sample_count(y, raw_rows, args.target)

    # 仅丢弃目标列缺失的行；特征列缺失由填补器 + 缺失指示器处理
    keep = ~y.isna()
    x = x.loc[keep]
    y = y.loc[keep]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=args.random_state,
    )

    best_name = None
    best_score = -np.inf
    best_model = None
    best_val_metrics = None
    best_params = None
    best_cv_score = None
    candidate_scores = []

    candidate_names = list(PARAM_GRIDS.keys())
    with tqdm(total=len(candidate_names), desc="网格搜索与验证", unit="model") as progress:
        for name in candidate_names:
            progress.set_postfix_str(f"调参 {name}")
            tuned_model, params, cv_score = tune_candidate(
                name,
                x_train,
                y_train,
                args.target,
                args.random_state,
                numeric_columns,
                categorical_columns,
            )

            progress.set_postfix_str(f"验证 {name}")
            val_metrics = evaluate(tuned_model, x_val, y_val, args.target)
            score = val_metrics["pass_rate"]
            candidate_scores.append((name, score, cv_score, params))
            if score > best_score:
                best_name = name
                best_score = score
                best_model = tuned_model
                best_val_metrics = val_metrics
                best_params = params
                best_cv_score = cv_score
            progress.update(1)

    assert best_name is not None and best_model is not None and best_val_metrics is not None

    with tqdm(total=2, desc="测试与保存", unit="step") as progress:
        progress.set_postfix_str("评估测试集")
        test_metrics = evaluate(best_model, x_test, y_test, args.target)
        progress.update(1)

        progress.set_postfix_str("保存模型")
        feature_names = get_transformed_feature_names(best_model)
        model_path = output_dir / f"{args.target}_best_model.joblib"
        joblib.dump(best_model, model_path)
        progress.update(1)

    print(f"预测目标: {args.target}")
    print(f"特征列 ({len(numeric_columns)}): {', '.join(numeric_columns)}")
    print(f"最佳模型: {best_name}")
    print(f"最佳网格搜索参数: {best_params}")
    print(f"数据划分: 训练集 {len(x_train)} 行, 验证集 {len(x_val)} 行, 测试集 {len(x_test)} 行")
    print(f"预处理后特征数: {len(feature_names)}（含缺失指示列）")
    print("\n候选模型验证集达标率:")
    for name, score, cv_score, params in sorted(candidate_scores, key=lambda item: item[1], reverse=True):
        print(f"{name}: 验证达标率 {score * 100:.2f}% | 交叉验证达标率 {cv_score * 100:.2f}% | 最优参数 {params}")
    print_metrics("验证集达标率", best_val_metrics)
    print_metrics("测试集达标率", test_metrics)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
