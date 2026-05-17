"""Data loading and preprocessing methods."""

from __future__ import annotations

import gzip
import importlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


_ZVUK_INTERACTIONS_FILE = "zvuk-interactions.parquet"
_ZVUK_EMBEDDINGS_FILE = "zvuk-track_artist_embedding.parquet"


def _zvuk_interactions_path(data_path: str) -> str:
    return os.path.join(data_path, _ZVUK_INTERACTIONS_FILE)


def _zvuk_embeddings_path(data_path: str) -> str:
    return os.path.join(data_path, _ZVUK_EMBEDDINGS_FILE)


def _has_zvuk_interactions(data_path: str) -> bool:
    return os.path.exists(_zvuk_interactions_path(data_path))


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        return df.drop(columns=unnamed)
    return df


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return _drop_unnamed_columns(pd.read_csv(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Неподдерживаемый формат файла split: {path}")


def _normalize_official_split_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if 'track_id' in df.columns and 'item_id' not in df.columns:
        df = df.rename(columns={'track_id': 'item_id'})

    required = {'user_id', 'item_id'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"В split `{split_name}` отсутствуют колонки: {sorted(missing)}")

    out = df.copy()
    out['user_id'] = out['user_id']
    out['item_id'] = pd.to_numeric(out['item_id'], errors='coerce')
    out = out[out['item_id'].notna()].copy()
    out['item_id'] = out['item_id'].astype(int)

    if 'timestamp' not in out.columns:
        out['timestamp'] = out.groupby('user_id').cumcount()
    else:
        out['timestamp'] = pd.to_numeric(out['timestamp'], errors='coerce')
        if out['timestamp'].isna().any():
            out['timestamp'] = out.groupby('user_id').cumcount()
        out['timestamp'] = out['timestamp'].astype(np.int64)

    return out


def _resolve_letitgo_dirs(cfg) -> tuple[Path, Path]:
    repo_path = Path(str(getattr(cfg, "letitgo_repo_path", Path.home() / "let-it-go"))).expanduser()
    dataset_key = str(getattr(cfg, "letitgo_dataset_key", "zvuk")).strip().lower().replace("-", "_")

    processed_dir_cfg = str(getattr(cfg, "letitgo_processed_dir", "")).strip()
    if processed_dir_cfg:
        processed_dir = Path(processed_dir_cfg).expanduser()
    else:
        processed_dir = repo_path / "data" / dataset_key / "processed"

    item_embeddings_dir_cfg = str(getattr(cfg, "letitgo_item_embeddings_dir", "")).strip()
    if item_embeddings_dir_cfg:
        item_embeddings_dir = Path(item_embeddings_dir_cfg).expanduser()
    else:
        item_embeddings_dir = processed_dir.parent / "item_embeddings"

    return processed_dir, item_embeddings_dir


def _resolve_letitgo_split_paths(cfg) -> tuple[str, dict[str, Path]]:
    dataset_key = str(getattr(cfg, "letitgo_dataset_key", "zvuk")).strip().lower().replace("-", "_")
    processed_dir, _ = _resolve_letitgo_dirs(cfg)

    if dataset_key == "zvuk":
        paths = {
            "train": processed_dir / "train_interactions.parquet",
            "val": processed_dir / "val_interactions.parquet",
            "test_inputs": processed_dir / "test_interactions.parquet",
            "ground_truth": processed_dir / "ground_truth.parquet",
        }
    elif dataset_key in {"amazon_m2", "amazonm2"}:
        dataset_key = "amazon_m2"
        paths = {
            "train": processed_dir / "train_data.csv",
            "val": processed_dir / "val_data.csv",
            "test_inputs": processed_dir / "test_inputs.csv",
            "ground_truth": processed_dir / "test_target.csv",
        }
    elif dataset_key == "yambda":
        paths = {
            "train": processed_dir / "train_interactions.parquet",
            "val": processed_dir / "val_interactions.parquet",
            "test_inputs": processed_dir / "test_interactions.parquet",
            "ground_truth": processed_dir / "ground_truth.parquet",
        }
    else:
        raise ValueError(
            f"Неподдерживаемый let-it-go dataset_key={dataset_key}. "
            "Ожидалось: zvuk, amazon_m2 или yambda."
        )

    return dataset_key, paths


def _load_warm_cold_items_from_pickles(processed_dir: Path) -> tuple[list[int], list[int]]:
    warm_pkl = processed_dir / "item2index_warm.pkl"
    cold_pkl = processed_dir / "item2index_cold.pkl"
    if not warm_pkl.exists() or not cold_pkl.exists():
        return [], []

    with warm_pkl.open("rb") as f:
        warm_map = pickle.load(f)
    with cold_pkl.open("rb") as f:
        cold_map = pickle.load(f)

    warm_items = sorted({int(v) for v in warm_map.values() if int(v) > 0})
    cold_items = sorted({int(v) for v in cold_map.values() if int(v) > 0})
    return warm_items, cold_items


def _load_letitgo_item_embeddings(
    item_embeddings_dir: Path,
    warm_items: list[int],
    cold_items: list[int],
) -> dict[int, np.ndarray]:
    warm_path = item_embeddings_dir / "embeddings_warm.npy"
    cold_path = item_embeddings_dir / "embeddings_cold.npy"
    if not warm_path.exists():
        return {}

    warm_arr = np.load(warm_path)
    cold_arr = np.load(cold_path) if cold_path.exists() else None

    if not warm_items:
        warm_items = list(range(1, int(warm_arr.shape[0]) + 1))
    if cold_arr is not None and not cold_items:
        start_idx = (max(warm_items) + 1) if warm_items else 1
        cold_items = list(range(start_idx, start_idx + int(cold_arr.shape[0])))

    item_embeddings: dict[int, np.ndarray] = {}

    warm_count = min(len(warm_items), int(warm_arr.shape[0]))
    for i in range(warm_count):
        item_embeddings[int(warm_items[i])] = np.asarray(warm_arr[i], dtype=np.float32)

    if cold_arr is not None:
        cold_count = min(len(cold_items), int(cold_arr.shape[0]))
        for i in range(cold_count):
            item_embeddings[int(cold_items[i])] = np.asarray(cold_arr[i], dtype=np.float32)

    return item_embeddings


def load_letitgo_official_splits(cfg):
    dataset_key, split_paths = _resolve_letitgo_split_paths(cfg)
    processed_dir, item_embeddings_dir = _resolve_letitgo_dirs(cfg)

    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены официальные split-файлы let-it-go:\n"
            + "\n".join(missing)
            + "\nСкачайте/подготовьте processed сплиты из репозитория let-it-go."
        )

    train_df = _normalize_official_split_df(_read_table(split_paths["train"]), "train")
    val_df = _normalize_official_split_df(_read_table(split_paths["val"]), "val")
    test_inputs_df = _normalize_official_split_df(_read_table(split_paths["test_inputs"]), "test_inputs")
    ground_truth_df = _normalize_official_split_df(_read_table(split_paths["ground_truth"]), "ground_truth")

    warm_items, cold_items = _load_warm_cold_items_from_pickles(processed_dir)
    if not warm_items:
        warm_items = sorted({int(v) for v in train_df['item_id'].unique() if int(v) > 0})
    if not cold_items:
        cold_from_cols = []
        if 'is_cold' in test_inputs_df.columns:
            cold_from_cols.extend(test_inputs_df.loc[test_inputs_df['is_cold'].astype(bool), 'item_id'].astype(int).tolist())
        if 'is_cold' in ground_truth_df.columns:
            cold_from_cols.extend(ground_truth_df.loc[ground_truth_df['is_cold'].astype(bool), 'item_id'].astype(int).tolist())
        if cold_from_cols:
            cold_items = sorted({int(v) for v in cold_from_cols if int(v) > 0})
        else:
            test_item_ids = set(test_inputs_df['item_id'].astype(int).tolist()) | set(
                ground_truth_df['item_id'].astype(int).tolist()
            )
            cold_items = sorted({int(v) for v in test_item_ids if int(v) not in set(warm_items) and int(v) > 0})

    all_interactions = pd.concat(
        [
            train_df[['user_id', 'item_id', 'timestamp']],
            val_df[['user_id', 'item_id', 'timestamp']],
            test_inputs_df[['user_id', 'item_id', 'timestamp']],
            ground_truth_df[['user_id', 'item_id', 'timestamp']],
        ],
        ignore_index=True,
    )

    item_embeddings = _load_letitgo_item_embeddings(item_embeddings_dir, warm_items, cold_items)

    return {
        "dataset_key": dataset_key,
        "processed_dir": str(processed_dir),
        "item_embeddings_dir": str(item_embeddings_dir),
        "train_df": train_df,
        "val_df": val_df,
        "test_inputs_df": test_inputs_df,
        "ground_truth_df": ground_truth_df,
        "full_interactions": all_interactions,
        "warm_items": warm_items,
        "cold_items": cold_items,
        "item_embeddings": item_embeddings,
    }


def encode_preencoded_interactions(data: pd.DataFrame):
    out = data.copy()
    out = out[out['user_id'].notna() & out['item_id'].notna()].copy()
    out['item_id'] = pd.to_numeric(out['item_id'], errors='coerce')
    out = out[out['item_id'].notna()].copy()
    out['item_id'] = out['item_id'].astype(int)

    if 'timestamp' not in out.columns:
        out['timestamp'] = out.groupby('user_id').cumcount()
    else:
        out['timestamp'] = pd.to_numeric(out['timestamp'], errors='coerce')
        if out['timestamp'].isna().any():
            out['timestamp'] = out.groupby('user_id').cumcount()
        out['timestamp'] = out['timestamp'].astype(np.int64)

    unique_items = sorted({int(v) for v in out['item_id'].unique() if int(v) > 0})
    unique_users = sorted(out['user_id'].unique().tolist())

    item_to_idx = {item_id: item_id for item_id in unique_items}
    idx_to_item = {item_id: item_id for item_id in unique_items}

    # Keep 1-based user indexing for compatibility with sparse interaction features.
    user_to_idx = {user_id: idx + 1 for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx + 1: user_id for idx, user_id in enumerate(unique_users)}

    out['user_id_encoded'] = out['user_id'].map(user_to_idx).astype(int)
    out['item_id_encoded'] = out['item_id'].astype(int)

    num_items = max(unique_items) if unique_items else 0
    num_users = len(unique_users)
    return out, item_to_idx, idx_to_item, user_to_idx, idx_to_user, num_items, num_users


def apply_encoded_columns_to_split(df: pd.DataFrame, user_to_idx: dict) -> pd.DataFrame:
    out = df.copy()
    out['user_id_encoded'] = out['user_id'].map(user_to_idx)
    out = out[out['user_id_encoded'].notna()].copy()
    out['user_id_encoded'] = out['user_id_encoded'].astype(int)
    out['item_id_encoded'] = pd.to_numeric(out['item_id'], errors='coerce').astype(int)
    if 'timestamp' not in out.columns:
        out['timestamp'] = out.groupby('user_id').cumcount()
    return out


def build_test_dataframe_from_inputs_and_ground_truth(
    test_inputs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    test_inputs = test_inputs_df.copy()
    ground_truth = ground_truth_df.copy()

    if 'timestamp' not in ground_truth.columns:
        last_timestamp = test_inputs.groupby('user_id')['timestamp'].max()
        ground_truth['timestamp'] = ground_truth['user_id'].map(last_timestamp).fillna(-1).astype(np.int64) + 1
    else:
        ground_truth['timestamp'] = pd.to_numeric(ground_truth['timestamp'], errors='coerce')
        if ground_truth['timestamp'].isna().any():
            last_timestamp = test_inputs.groupby('user_id')['timestamp'].max()
            ground_truth['timestamp'] = ground_truth['user_id'].map(last_timestamp).fillna(-1).astype(np.int64) + 1
        ground_truth['timestamp'] = ground_truth['timestamp'].astype(np.int64)

    cols = ['user_id', 'user_id_encoded', 'item_id', 'item_id_encoded', 'timestamp']
    merged = pd.concat(
        [
            test_inputs[cols],
            ground_truth[cols],
        ],
        ignore_index=True,
    )
    return merged.sort_values(['user_id', 'timestamp']).reset_index(drop=True)


def _download_zvuk_dataset(dataset_id: str) -> str | None:
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError:
        print("⚠ kagglehub не установлен. Установите: pip install kagglehub")
        return None

    try:
        download_path = kagglehub.dataset_download(dataset_id)
    except Exception as exc:
        print(f"⚠ Не удалось скачать {dataset_id}: {exc}")
        return None

    print(f"✓ Zvuk датасет скачан: {download_path}")
    return download_path


def _ensure_zvuk_dataset_available(cfg) -> str | None:
    if _has_zvuk_interactions(cfg.zvuk_data_path):
        return cfg.zvuk_data_path

    if not getattr(cfg, "auto_download_zvuk_dataset", False):
        return None

    print("=" * 80)
    print("Zvuk датасет не найден локально. Запускаем скачивание через kagglehub...")
    print("=" * 80)
    downloaded_path = _download_zvuk_dataset(getattr(cfg, "zvuk_dataset_id", "alexxl/zvuk-dataset"))
    if downloaded_path is None:
        return None
    if not _has_zvuk_interactions(downloaded_path):
        print(f"⚠ После скачивания не найден {_ZVUK_INTERACTIONS_FILE} в {downloaded_path}")
        return None
    return downloaded_path


def load_amazon_m2_dataset(data_path: str, locale: str = "DE", min_sessions: int = 5):
    """Load Kaggle Amazon M2 interactions from sessions/product files."""
    sessions_file = os.path.join(data_path, "sessions_train.csv")
    products_file = os.path.join(data_path, "products_train.csv")

    if not os.path.exists(sessions_file):
        print(f"⚠ Файл {sessions_file} не найден")
        return None, None

    print("Загрузка Kaggle Amazon M2 датасета...")
    sessions = pd.read_csv(sessions_file)

    if locale:
        sessions = sessions[sessions['locale'] == locale]
        print(f"Отфильтровано по locale={locale}: {len(sessions)} сессий")

    import re

    rows = []
    for idx, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Обработка сессий"):
        try:
            prev_items = re.findall(r"'([^']+)'", str(row['prev_items']))
            if not prev_items:
                try:
                    prev_items_array = eval(str(row['prev_items']))
                    if isinstance(prev_items_array, (list, tuple, np.ndarray)):
                        prev_items = [str(item) for item in prev_items_array]
                    else:
                        prev_items = [str(prev_items_array)]
                except Exception:
                    continue

            sequence = prev_items + [str(row['next_item'])]
            for i in range(len(sequence) - 1):
                rows.append(
                    {
                        'session_id': idx,
                        'item_id': sequence[i],
                        'next_item_id': sequence[i + 1],
                        'position': i,
                        'locale': row['locale'],
                    }
                )
        except Exception:
            continue

    data = pd.DataFrame(rows)
    if data.empty:
        return None, None

    data['user_id'] = data['session_id']
    data['item_id'] = data['next_item_id']
    data['timestamp'] = data.groupby('session_id').cumcount()

    user_counts = data['user_id'].value_counts()
    data = data[data['user_id'].isin(user_counts[user_counts >= min_sessions].index)]

    products = None
    if os.path.exists(products_file):
        products = pd.read_csv(products_file)
        if locale:
            products = products[products['locale'] == locale]

    data = data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    print(f"✓ Загружено {len(data)} взаимодействий")
    print(f"✓ Уникальных пользователей: {data['user_id'].nunique()}")
    print(f"✓ Уникальных товаров: {data['item_id'].nunique()}")

    return data, products


def load_zvuk_dataset(data_path: str, sample_fraction: float = 0.01, min_user_interactions: int = 5,
                    min_item_interactions: int = 5):
    """Load the Zvuk dataset and its precomputed text embeddings."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("⚠ pyarrow не установлен. Установите: pip install pyarrow")
        return None, None

    interactions_file = _zvuk_interactions_path(data_path)
    embeddings_file = _zvuk_embeddings_path(data_path)

    if not os.path.exists(interactions_file):
        print(f"⚠ Файл {interactions_file} не найден")
        return None, None

    print(f"Загрузка Zvuk датасета (sample_fraction={sample_fraction})...")
    df_inter = pd.read_parquet(interactions_file, engine='pyarrow')

    if sample_fraction < 1.0:
        df_inter = df_inter.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        print(f"Загружено {len(df_inter):,} взаимодействий")

    df_inter = df_inter.rename(columns={'track_id': 'item_id'})

    if 'datetime' in df_inter.columns:
        df_inter['timestamp'] = pd.to_datetime(df_inter['datetime']).astype('int64') // 10**9
    else:
        df_inter['timestamp'] = df_inter.groupby('user_id').cumcount()

    user_counts = df_inter['user_id'].value_counts()
    item_counts = df_inter['item_id'].value_counts()
    df_inter = df_inter[df_inter['user_id'].isin(user_counts[user_counts >= min_user_interactions].index)]
    df_inter = df_inter[df_inter['item_id'].isin(item_counts[item_counts >= min_item_interactions].index)]

    user_counts = df_inter['user_id'].value_counts()
    item_counts = df_inter['item_id'].value_counts()
    df_inter = df_inter[df_inter['user_id'].isin(user_counts[user_counts >= min_user_interactions].index)]
    df_inter = df_inter[df_inter['item_id'].isin(item_counts[item_counts >= min_item_interactions].index)]

    df_inter = df_inter.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    if not os.path.exists(embeddings_file):
        print("⚠ Файл с эмбеддингами не найден")
        return df_inter, {}

    df_emb = pd.read_parquet(embeddings_file, engine='pyarrow')
    df_emb_unique = df_emb.drop_duplicates(subset=['track_id'], keep='first').rename(columns={'track_id': 'item_id'})

    unique_items = set(df_inter['item_id'].unique())
    item_embeddings = {}

    for _, row in tqdm(df_emb_unique.iterrows(), total=len(df_emb_unique), desc="Загрузка эмбеддингов"):
        item_id = row['item_id']
        if item_id not in unique_items:
            continue
        vec = row['vector']
        if isinstance(vec, list):
            vec = np.array(vec, dtype=np.float32)
        elif isinstance(vec, np.ndarray):
            vec = vec.astype(np.float32)
        else:
            continue
        item_embeddings[item_id] = vec

    print(f"✓ Загружено {len(df_inter):,} взаимодействий")
    print(f"✓ Уникальных пользователей: {df_inter['user_id'].nunique():,}")
    print(f"✓ Уникальных товаров: {df_inter['item_id'].nunique():,}")
    print(f"✓ Эмбеддингов загружено: {len(item_embeddings):,}")

    return df_inter, item_embeddings


def load_amazon_dataset(dataset_name: str = 'Beauty', data_dir: str = './data', min_interactions: int = 5):
    """Load standard Amazon JSON.gz dataset if available."""
    reviews_file = os.path.join(data_dir, f'{dataset_name}.json.gz')
    if not os.path.exists(reviews_file):
        print(f"⚠ Файл {reviews_file} не найден.")
        return None, None

    data_list = []
    with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                data_list.append(
                    {
                        'user_id': review.get('reviewerID', ''),
                        'item_id': review.get('asin', ''),
                        'rating': review.get('overall', 0),
                        'timestamp': review.get('unixReviewTime', 0),
                        'review_text': review.get('reviewText', ''),
                        'summary': review.get('summary', ''),
                    }
                )
            except Exception:
                continue

    data = pd.DataFrame(data_list)
    if data.empty:
        return None, None

    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()

    valid_users = user_counts[user_counts >= min_interactions].index
    valid_items = item_counts[item_counts >= min_interactions].index

    data = data[data['user_id'].isin(valid_users)]
    data = data[data['item_id'].isin(valid_items)]

    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()
    data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
    data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]

    data = data.sort_values('timestamp').reset_index(drop=True)
    return data, None


def generate_embeddings_from_text(texts, embedding_dim: int = 128, method: str = 'random'):
    if method == 'random':
        out = {}
        if isinstance(texts, dict):
            items = list(texts.keys())
            for item_id in items:
                out[item_id] = np.random.randn(embedding_dim).astype(np.float32)
        else:
            for item_id in range(len(list(texts))):
                out[item_id] = np.random.randn(embedding_dim).astype(np.float32)
        return out

    raise NotImplementedError("Только random mode реализован в базовой версии")


def create_item_embeddings_if_missing(item_embeddings, items, embedding_dim: int = 128):
    if item_embeddings is None:
        item_embeddings = {}

    if len(item_embeddings) == 0:
        np.random.seed(42)
        for item in items:
            item_seed = hash(str(item)) % (2**32)
            rng = np.random.RandomState(item_seed)
            item_embeddings[item] = rng.randn(embedding_dim).astype(np.float32)
        return item_embeddings

    missing = set(items) - set(item_embeddings.keys())
    if missing:
        print(f"⚠ Найдено {len(missing)} товаров без эмбеддингов. Добавляем случайные.")
    for item in missing:
        item_seed = hash(str(item)) % (2**32)
        rng = np.random.RandomState(item_seed)
        any_vec = next(iter(item_embeddings.values()))
        item_embeddings[item] = rng.randn(len(any_vec)).astype(np.float32)

    return item_embeddings


def add_time_idx(df: pd.DataFrame, user_col='user_id', timestamp_col='timestamp', sort: bool = True) -> pd.DataFrame:
    if sort:
        df = df.sort_values([user_col, timestamp_col])
    out = df.copy()
    out['time_idx'] = out.groupby(user_col).cumcount()
    out['time_idx_reversed'] = out.groupby(user_col).cumcount(ascending=False)
    return out


def filter_items(df: pd.DataFrame, item_min_count: int, item_col='item_id', user_col='user_id') -> pd.DataFrame:
    item_count = df.groupby(item_col)[user_col].nunique()
    item_ids = item_count[item_count >= item_min_count].index
    return df[df[item_col].isin(item_ids)]


def filter_users(df: pd.DataFrame, user_min_count: int, user_col='user_id', item_col='item_id') -> pd.DataFrame:
    user_count = df.groupby(user_col)[item_col].nunique()
    user_ids = user_count[user_count >= user_min_count].index
    return df[df[user_col].isin(user_ids)]


def create_cold_items(data: pd.DataFrame, cold_threshold: int = 5, cold_fraction: float = 0.2):
    item_counts = data['item_id'].value_counts()
    total_items = len(item_counts)
    num_cold = int(total_items * cold_fraction)
    items_to_cool = item_counts.head(num_cold).index.tolist()

    data_modified = data.copy()
    for item_id in tqdm(items_to_cool, desc="Охлаждение товаров"):
        item_interactions = data_modified[data_modified['item_id'] == item_id].sort_values('timestamp')
        if len(item_interactions) > cold_threshold:
            keep_indices = item_interactions.head(cold_threshold).index
            data_modified = data_modified.drop(item_interactions.index.difference(keep_indices))

    item_counts_after = data_modified['item_id'].value_counts()
    cold_items = item_counts_after[item_counts_after <= cold_threshold].index.tolist()
    return data_modified, cold_items


def save_embeddings(path: str, item_embeddings) -> None:
    with open(path, 'wb') as f:
        pickle.dump(item_embeddings, f)


def load_data_bundle(config):
    cfg = config.data
    os.makedirs(cfg.data_dir, exist_ok=True)

    data = None
    metadata = None
    item_embeddings = None

    if getattr(cfg, "use_letitgo_official_splits", False):
        print("=" * 80)
        print(
            "Используются официальные сплиты let-it-go "
            f"({getattr(cfg, 'letitgo_dataset_key', 'zvuk')})"
        )
        print("=" * 80)
        letitgo_bundle = load_letitgo_official_splits(cfg)
        data = letitgo_bundle["full_interactions"].copy()
        item_embeddings = letitgo_bundle.get("item_embeddings", {})
        metadata = {"letitgo_bundle": letitgo_bundle}

    if data is None and cfg.use_zvuk_dataset:
        zvuk_data_path = _ensure_zvuk_dataset_available(cfg)
        if zvuk_data_path is not None:
            print("=" * 80)
            print("Используется Zvuk датасет")
            print("=" * 80)
            data, item_embeddings = load_zvuk_dataset(
                zvuk_data_path,
                sample_fraction=cfg.zvuk_sample_fraction,
                min_user_interactions=cfg.zvuk_min_user_interactions,
                min_item_interactions=cfg.zvuk_min_item_interactions,
            )
        else:
            print(f"⚠ Zvuk датасет недоступен по пути: {cfg.zvuk_data_path}")

    if data is None and cfg.use_kaggle_dataset and os.path.exists(cfg.kaggle_data_path):
        print("=" * 80)
        print("Используется Kaggle Amazon M2 датасет")
        print("=" * 80)
        data, metadata = load_amazon_m2_dataset(cfg.kaggle_data_path, locale=cfg.locale, min_sessions=cfg.min_interactions)
    elif data is None and os.path.exists(os.path.join(cfg.data_dir, f'{cfg.dataset_name}.json.gz')):
        print("=" * 80)
        print("Используется стандартный Amazon датасет")
        print("=" * 80)
        data, metadata = load_amazon_dataset(cfg.dataset_name, cfg.data_dir, min_interactions=cfg.min_interactions)
    elif data is None:
        print("=" * 80)
        print("Реальные данные не найдены")
        print("=" * 80)

    if data is None:
        print("\nИспользуются синтетические данные для демонстрации...")
        rng = np.random.RandomState(42)
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        data = pd.DataFrame(
            {
                'user_id': rng.randint(0, n_users, n_interactions),
                'item_id': rng.randint(0, n_items, n_interactions),
                'timestamp': np.sort(rng.randint(0, 1_000_000, n_interactions)),
                'rating': rng.randint(1, 6, n_interactions),
            }
        )
        data = data.drop_duplicates(subset=['user_id', 'item_id'])

    if data is None:
        raise RuntimeError('Не удалось загрузить или синтезировать данные')

    cold_items_present = data['item_id'].value_counts().loc[lambda s: s <= cfg.cold_threshold] if not data.empty else pd.Series(dtype=int)

    if (
        len(cold_items_present) == 0
        and cfg.create_cold_items_flag
        and not getattr(cfg, "use_letitgo_official_splits", False)
    ):
        print("\n" + "=" * 80)
        print("Холодных товаров не найдено. Создаем искусственно...")
        print("=" * 80)
        data, synthetic_cold = create_cold_items(
            data,
            cold_threshold=cfg.cold_threshold,
            cold_fraction=cfg.cold_fraction,
        )
        print(f"✓ Данные обновлены: {len(data):,} взаимодействий")

    if item_embeddings is None:
        if metadata is not None and len(metadata) > 0 and isinstance(metadata, pd.DataFrame) and 'id' in metadata.columns:
            item_texts = {item_id: str(item_id) for item_id in data['item_id'].unique()}
        else:
            item_texts = {item_id: f"Item {item_id}" for item_id in data['item_id'].unique()}
        item_embeddings = generate_embeddings_from_text(item_texts, embedding_dim=128, method='random')

    item_embeddings = create_item_embeddings_if_missing(item_embeddings, set(data['item_id'].unique()), embedding_dim=128)

    return data.reset_index(drop=True), metadata, item_embeddings


def encode_interactions(data: pd.DataFrame):
    data = data.copy()
    data = data[data['user_id'].notna() & data['item_id'].notna()]

    if data['user_id'].dtype == object:
        data = data[(data['user_id'] != '') & (data['item_id'] != '')]

    unique_items = sorted(data['item_id'].unique())
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
    idx_to_item = {idx + 1: item for idx, item in enumerate(unique_items)}

    unique_users = sorted(data['user_id'].unique())
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user = {idx: user for idx, user in enumerate(unique_users)}

    data['user_id_encoded'] = data['user_id'].map(user_to_idx)
    data['item_id_encoded'] = data['item_id'].map(item_to_idx)
    data = data[data['user_id_encoded'].notna() & data['item_id_encoded'].notna()]

    return (
        data,
        item_to_idx,
        idx_to_item,
        user_to_idx,
        idx_to_user,
        len(unique_items),
        len(unique_users),
    )


def split_train_test_by_time(data_encoded: pd.DataFrame, ratio: float = 0.8):
    train_parts = []
    test_parts = []

    for user_id in data_encoded['user_id'].unique():
        user_data = data_encoded[data_encoded['user_id'] == user_id].sort_values('timestamp')
        if len(user_data) < 2:
            continue
        split_idx = int(len(user_data) * ratio)
        if split_idx == 0:
            split_idx = 1
        train_parts.append(user_data.iloc[:split_idx])
        test_parts.append(user_data.iloc[split_idx:])

    if not train_parts:
        return pd.DataFrame(), pd.DataFrame()
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df
