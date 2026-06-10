import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



TARGET_COLUMNS = ["致密度", "热导率"]
DENSITY_PERCENT_TOLERANCE = 1.0
DENSITY_DECIMAL_TOLERANCE = 0.01
THERMAL_CONDUCTIVITY_TOLERANCE = 5.0

DEFAULT_EXCLUDED_COLUMNS = {
    "文件名",
    "文献名",
    "额外备注",
    "热膨胀系数",
    "抗弯强度",
    "温循后热导率",
}

MISSING_TOKENS = {
    "缺失",
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("只支持 .csv、.xlsx、.xls 数据表")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype(str).str.strip()
    text = text.replace({token: np.nan for token in MISSING_TOKENS})
    text = text.str.replace(",", "", regex=False)
    text = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(text, errors="coerce")


def clean_categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.astype(str).apply(lambda col: col.str.strip())
    cleaned = cleaned.replace({token: "缺失" for token in MISSING_TOKENS})
    return cleaned.fillna("缺失")


def get_target_tolerance(target: str, y_true: pd.Series) -> float:
    if target == "致密度":
        return DENSITY_DECIMAL_TOLERANCE if y_true.max() <= 1.5 else DENSITY_PERCENT_TOLERANCE
    if target == "热导率":
        return THERMAL_CONDUCTIVITY_TOLERANCE
    raise ValueError(f"未知目标列: {target}")


def build_feature_columns(df: pd.DataFrame, target: str, include_text_id: bool) -> list[str]:
    excluded = set(TARGET_COLUMNS)
    if not include_text_id:
        excluded.update(DEFAULT_EXCLUDED_COLUMNS)
    excluded.add(target)
    return [col for col in df.columns if col not in excluded]


def split_feature_types(
    df: pd.DataFrame,
    feature_columns: list[str],
    min_numeric_valid_rate: float,
    max_categories: int,
) -> tuple[list[str], list[str]]:
    numeric_columns = []
    categorical_columns = []

    for column in feature_columns:
        numeric = clean_numeric_series(df[column])
        numeric_valid_rate = float(numeric.notna().mean())
        unique_count = int(df[column].astype(str).str.strip().nunique(dropna=True))

        if numeric_valid_rate >= min_numeric_valid_rate:
            numeric_columns.append(column)
        elif unique_count <= max_categories:
            categorical_columns.append(column)

    if not numeric_columns and not categorical_columns:
        raise ValueError("没有可用特征。请降低 --min-numeric-valid-rate 或增大 --max-categories。")

    return numeric_columns, categorical_columns


def fit_preprocessor(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict:
    numeric_scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)

    numeric_values = build_numeric_matrix(df, numeric_columns)
    numeric_scaler.fit(numeric_values)

    if categorical_columns:
        categorical_values = clean_categorical_frame(df[categorical_columns])
        encoder.fit(categorical_values)
    else:
        encoder = None

    return {
        "numeric_scaler": numeric_scaler,
        "encoder": encoder,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def build_numeric_matrix(df: pd.DataFrame, numeric_columns: list[str]) -> np.ndarray:
    if not numeric_columns:
        return np.empty((len(df), 0), dtype=np.float32)

    numeric_df = pd.DataFrame({col: clean_numeric_series(df[col]) for col in numeric_columns}, index=df.index)
    numeric_missing = numeric_df.isna().astype(np.float32).to_numpy()
    numeric_values = numeric_df.fillna(0.0).astype(np.float32).to_numpy()
    return np.hstack([numeric_values, numeric_missing])


def transform_features(df: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    numeric_columns = preprocessor["numeric_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    numeric_scaler = preprocessor["numeric_scaler"]
    encoder = preprocessor["encoder"]

    numeric_values = build_numeric_matrix(df, numeric_columns)
    numeric_values = numeric_scaler.transform(numeric_values) if numeric_values.shape[1] else numeric_values

    if categorical_columns:
        categorical_values = clean_categorical_frame(df[categorical_columns])
        categorical_values = encoder.transform(categorical_values).astype(np.float32)
    else:
        categorical_values = np.empty((len(df), 0), dtype=np.float32)

    return np.hstack([numeric_values.astype(np.float32), categorical_values.astype(np.float32)])


def get_feature_names(preprocessor: dict) -> list[str]:
    numeric_columns = preprocessor["numeric_columns"]
    categorical_columns = preprocessor["categorical_columns"]
    encoder = preprocessor["encoder"]

    names = []
    for column in numeric_columns:
        names.append(column)
    for column in numeric_columns:
        names.append(f"{column}_是否缺失")
    if categorical_columns and encoder is not None:
        names.extend(encoder.get_feature_names_out(categorical_columns).tolist())
    return names


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, target: str) -> dict:
    error = np.abs(y_true.to_numpy() - y_pred)
    tolerance = get_target_tolerance(target, y_true)
    return {
        "target": target,
        "pass_rate": float(np.mean(error <= tolerance)),
        "tolerance": tolerance,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(
        f"{metrics['target']}达标率: {metrics['pass_rate'] * 100:.2f}% | "
        f"达标误差阈值: {metrics['tolerance']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f}"
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, x: np.ndarray, y_mean: float, y_std: float, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x).float().to(device)
        pred_scaled = model(tensor).cpu().numpy()
    return pred_scaled * y_std + y_mean


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    random_state: int,
) -> tuple[TabularMLP, dict]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std == 0:
        y_std = 1.0

    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_scaled = ((y_val - y_mean) / y_std).astype(np.float32)

    model = TabularMLP(x_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    train_loader = make_loader(x_train, y_train_scaled, batch_size, shuffle=True)
    val_x = torch.from_numpy(x_val).float().to(device)
    val_y = torch.from_numpy(y_val_scaled).float().to(device)

    best_state = None
    best_val_mae = np.inf
    wait = 0

    epoch_progress = tqdm(range(1, epochs + 1), desc="深度学习训练", unit="epoch")
    for epoch in epoch_progress:
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_mae_scaled = torch.mean(torch.abs(val_pred - val_y)).item()

        if val_mae_scaled < best_val_mae:
            best_val_mae = val_mae_scaled
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        epoch_progress.set_postfix(
            {
                "val_mae_scaled": f"{val_mae_scaled:.4f}",
                "best": f"{best_val_mae:.4f}",
                "patience": f"{wait}/{patience}",
            }
        )

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "device": str(device),
        "best_val_mae_scaled": float(best_val_mae),
        "epochs_ran": epoch,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, train_info


def build_top_error_rows(
    raw_df: pd.DataFrame,
    target: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    tolerance: float,
    limit: int,
) -> pd.DataFrame:
    error = np.abs(y_true.to_numpy() - y_pred)
    result = raw_df.copy()
    result.insert(0, "是否达标", error <= tolerance)
    result.insert(0, "绝对误差", error)
    result.insert(0, "预测值", y_pred)
    result.insert(0, "真实值", y_true.to_numpy())
    return result.sort_values("绝对误差", ascending=False).head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 PyTorch MLP 训练单目标表格预测模型")
    parser.add_argument("--data", required=True, help="数据表路径，支持 csv/xlsx/xls")
    parser.add_argument("--target", required=True, choices=TARGET_COLUMNS, help="预测目标：致密度 或 热导率")
    parser.add_argument("--output-dir", default="model_output", help="模型和结果输出目录")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64], help="MLP 隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--top-errors", type=int, default=10)
    parser.add_argument("--min-numeric-valid-rate", type=float, default=0.2)
    parser.add_argument("--max-categories", type=int, default=60)
    parser.add_argument("--include-text-id", action="store_true", help="是否纳入文件名、文献名、备注等高风险文本列")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.data))
    df.columns = df.columns.astype(str).str.strip()
    if args.target not in df.columns:
        raise ValueError(f"数据表缺少目标列: {args.target}")

    y_all = clean_numeric_series(df[args.target])
    breakpoint()
    keep = ~y_all.isna()
    if int(keep.sum()) < 20:
        raise ValueError(f"`{args.target}` 有效样本少于 20 行，不建议训练深度学习模型。")

    df_model = df.loc[keep].copy()
    y = y_all.loc[keep]
    feature_columns = build_feature_columns(df_model, args.target, args.include_text_id)
    numeric_columns, categorical_columns = split_feature_types(
        df_model,
        feature_columns,
        args.min_numeric_valid_rate,
        args.max_categories,
    )

    train_val_df, test_df, y_train_val, y_test = train_test_split(
        df_model,
        y,
        test_size=0.1,
        random_state=args.random_state,
    )
    train_df, val_df, y_train, y_val = train_test_split(
        train_val_df,
        y_train_val,
        test_size=1 / 9,
        random_state=args.random_state,
    )

    preprocessor = fit_preprocessor(train_df, numeric_columns, categorical_columns)
    x_train = transform_features(train_df, preprocessor)
    x_val = transform_features(val_df, preprocessor)
    x_train_val = transform_features(train_val_df, preprocessor)
    x_test = transform_features(test_df, preprocessor)

    model, train_info = train_model(
        x_train,
        y_train.to_numpy(dtype=np.float32),
        x_val,
        y_val.to_numpy(dtype=np.float32),
        args.hidden_dims,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.epochs,
        args.patience,
        args.random_state,
    )

    device = torch.device(train_info["device"])
    with tqdm(total=4, desc="深度学习测试", unit="step") as progress:
        progress.set_postfix_str("预测训练/验证合并集")
        train_val_pred = predict(model, x_train_val, train_info["y_mean"], train_info["y_std"], device)
        progress.update(1)

        progress.set_postfix_str("预测测试集")
        test_pred = predict(model, x_test, train_info["y_mean"], train_info["y_std"], device)
        progress.update(1)

        progress.set_postfix_str("评估训练/验证合并集")
        train_val_metrics = evaluate_predictions(y_train_val, train_val_pred, args.target)
        progress.update(1)

        progress.set_postfix_str("评估测试集")
        test_metrics = evaluate_predictions(y_test, test_pred, args.target)
        progress.update(1)

    top_error_rows = build_top_error_rows(
        test_df,
        args.target,
        y_test,
        test_pred,
        test_metrics["tolerance"],
        args.top_errors,
    )

    model_path = output_dir / f"{args.target}_mlp_model.pt"
    preprocessor_path = output_dir / f"{args.target}_mlp_preprocessor.joblib"
    top_errors_path = output_dir / f"{args.target}_mlp_test_top_errors.csv"

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": x_train.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "target": args.target,
            "train_info": train_info,
        },
        model_path,
    )
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_names": get_feature_names(preprocessor),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        preprocessor_path,
    )
    top_error_rows.to_csv(top_errors_path, encoding="utf-8-sig")

    print(f"预测目标: {args.target}")
    print(f"有效样本数: {len(df_model)}")
    print(f"训练集: {len(train_df)} 行, 验证集: {len(val_df)} 行, 测试集: {len(test_df)} 行")
    print(f"数值特征列: {numeric_columns}")
    print(f"类别特征列: {categorical_columns}")
    print(f"模型输入维度: {x_train.shape[1]}")
    print(f"训练设备: {train_info['device']}, 实际训练轮数: {train_info['epochs_ran']}")
    print_metrics("训练/验证合并集达标率", train_val_metrics)
    print_metrics("测试集达标率", test_metrics)
    print(f"\n模型已保存到: {model_path}")
    print(f"预处理器已保存到: {preprocessor_path}")
    print(f"测试集误差最大样本已保存到: {top_errors_path}")


if __name__ == "__main__":
    main()
