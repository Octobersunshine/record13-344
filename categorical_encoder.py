import pandas as pd
import pickle
import json
import os
import warnings
from typing import List, Dict, Union, Optional


class CategoricalEncoder:
    """分类变量编码器，支持独热编码和标签编码。

    针对高基数类别（> max_categories 个唯一值）的独热编码会自动检测并处理，
    避免维度爆炸问题。
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        encoding_type: str = "onehot",
        max_categories: int = 100,
        high_cardinality_strategy: str = "auto",
    ):
        """
        初始化编码器。

        Args:
            columns: 需要编码的列名列表，若为 None 则自动识别所有对象类型列
            encoding_type: 编码类型，"onehot" 或 "label"
            max_categories: 独热编码最大类别数阈值，超过则触发高基数处理
            high_cardinality_strategy: 高基数处理策略:
                - "auto": 自动降级为标签编码（默认）
                - "warn": 仅发出警告，仍使用独热编码
                - "raise": 抛出 ValueError
        """
        if encoding_type not in ("onehot", "label"):
            raise ValueError("encoding_type 必须是 'onehot' 或 'label'")
        if high_cardinality_strategy not in ("auto", "warn", "raise"):
            raise ValueError("high_cardinality_strategy 必须是 'auto'、'warn' 或 'raise'")
        if not isinstance(max_categories, int) or max_categories <= 0:
            raise ValueError("max_categories 必须是正整数")

        self.columns = columns
        self.encoding_type = encoding_type
        self.max_categories = max_categories
        self.high_cardinality_strategy = high_cardinality_strategy
        self._fitted = False
        self._mapping: Dict[str, Dict] = {}
        self._onehot_columns: Dict[str, List[str]] = {}
        self._actual_encoding: Dict[str, str] = {}

    def fit(self, data: pd.DataFrame) -> "CategoricalEncoder":
        """
        从数据中学习编码映射。

        Args:
            data: 输入的 DataFrame

        Returns:
            self

        Raises:
            ValueError: 高基数策略为 "raise" 且某列类别数超过阈值时
        """
        target_cols = self._resolve_columns(data)

        for col in target_cols:
            unique_values = sorted(data[col].dropna().unique().tolist())
            n_unique = len(unique_values)

            actual_type = self.encoding_type
            if self.encoding_type == "onehot" and n_unique > self.max_categories:
                strategy = self.high_cardinality_strategy
                if strategy == "raise":
                    raise ValueError(
                        f"列 '{col}' 有 {n_unique} 个唯一类别，超过 max_categories={self.max_categories}，"
                        f"会导致维度爆炸。建议改用标签编码或调大 max_categories。"
                    )
                elif strategy == "warn":
                    warnings.warn(
                        f"列 '{col}' 有 {n_unique} 个唯一类别，超过 max_categories={self.max_categories}，"
                        f"独热编码可能导致维度爆炸。"
                    )
                elif strategy == "auto":
                    warnings.warn(
                        f"列 '{col}' 有 {n_unique} 个唯一类别，超过 max_categories={self.max_categories}，"
                        f"已自动降级为标签编码以避免维度爆炸。"
                    )
                    actual_type = "label"

            self._actual_encoding[col] = actual_type

            if actual_type == "label":
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self._mapping[col] = mapping
            else:
                self._mapping[col] = {val: val for val in unique_values}
                self._onehot_columns[col] = [f"{col}_{val}" for val in unique_values]

        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对数据进行编码转换。

        Args:
            data: 输入的 DataFrame

        Returns:
            编码后的 DataFrame
        """
        if not self._fitted:
            raise RuntimeError("编码器尚未 fit，请先调用 fit() 或 fit_transform()")

        result = data.copy()
        target_cols = [col for col in self._actual_encoding.keys() if col in result.columns]

        for col in target_cols:
            actual_type = self._actual_encoding[col]
            if actual_type == "label":
                mapping = self._mapping[col]
                result[col] = result[col].map(
                    lambda x: mapping.get(x, -1) if pd.notna(x) else x
                )
            else:
                dummies = pd.get_dummies(result[col], prefix=col)
                expected_cols = self._onehot_columns.get(col, [])
                for ec in expected_cols:
                    if ec not in dummies.columns:
                        dummies[ec] = 0
                dummies = dummies[expected_cols].astype(int)
                result = pd.concat([result.drop(columns=[col]), dummies], axis=1)

        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        学习编码映射并转换数据。

        Args:
            data: 输入的 DataFrame

        Returns:
            编码后的 DataFrame
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        反向解码，将编码后的数据还原为原始分类值。

        Args:
            data: 编码后的 DataFrame

        Returns:
            解码后的 DataFrame
        """
        if not self._fitted:
            raise RuntimeError("编码器尚未 fit，请先调用 fit() 或 fit_transform()")

        result = data.copy()

        for col, actual_type in self._actual_encoding.items():
            if actual_type == "label":
                if col in result.columns:
                    mapping = self._mapping[col]
                    reverse_map = {v: k for k, v in mapping.items()}
                    result[col] = result[col].map(
                        lambda x: reverse_map.get(int(x), x) if pd.notna(x) and int(x) != -1 else x
                    )
            else:
                encoded_cols = self._onehot_columns.get(col, [])
                if all(ec in result.columns for ec in encoded_cols):
                    idxmax = result[encoded_cols].idxmax(axis=1)
                    decoded = idxmax.map(lambda x: x[len(col) + 1:] if pd.notna(x) else x)
                    result = result.drop(columns=encoded_cols)
                    result[col] = decoded

        return result

    def get_mapping(self) -> Dict[str, Dict]:
        """
        获取编码映射关系。

        Returns:
            各列的编码映射字典
        """
        return self._mapping.copy()

    def get_actual_encoding(self) -> Dict[str, str]:
        """
        获取各列实际使用的编码类型。

        当高基数策略为 "auto" 时，部分列可能被自动降级为标签编码，
        通过此方法可以查看每列最终使用的编码方式。

        Returns:
            各列实际编码类型字典 {"列名": "onehot" 或 "label"}
        """
        return self._actual_encoding.copy()

    def save(self, filepath: str) -> None:
        """
        保存编码器到文件（pickle 格式）。

        Args:
            filepath: 保存路径
        """
        state = {
            "columns": self.columns,
            "encoding_type": self.encoding_type,
            "max_categories": self.max_categories,
            "high_cardinality_strategy": self.high_cardinality_strategy,
            "_fitted": self._fitted,
            "_mapping": self._mapping,
            "_onehot_columns": self._onehot_columns,
            "_actual_encoding": self._actual_encoding,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def save_json(self, filepath: str) -> None:
        """
        保存编码映射到 JSON 文件（便于跨语言使用）。

        Args:
            filepath: 保存路径
        """
        def convert_keys(d):
            return {str(k): v for k, v in d.items()}

        data = {
            "encoding_type": self.encoding_type,
            "columns": list(self._mapping.keys()),
            "mapping": {col: convert_keys(m) for col, m in self._mapping.items()},
        }
        if self._onehot_columns:
            data["onehot_columns"] = self._onehot_columns

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CategoricalEncoder":
        """
        从文件加载编码器。

        Args:
            filepath: 文件路径

        Returns:
            加载的编码器实例
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        max_categories = state.get("max_categories", 100)
        high_cardinality_strategy = state.get("high_cardinality_strategy", "auto")

        encoder = cls(
            columns=state["columns"],
            encoding_type=state["encoding_type"],
            max_categories=max_categories,
            high_cardinality_strategy=high_cardinality_strategy,
        )
        encoder._fitted = state["_fitted"]
        encoder._mapping = state["_mapping"]
        encoder._onehot_columns = state["_onehot_columns"]
        encoder._actual_encoding = state.get("_actual_encoding", {})
        return encoder

    def _resolve_columns(self, data: pd.DataFrame) -> List[str]:
        """解析需要编码的列。"""
        if self.columns is not None:
            missing = [c for c in self.columns if c not in data.columns]
            if missing:
                raise ValueError(f"数据中缺少列: {missing}")
            return list(self.columns)

        obj_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        if not obj_cols:
            raise ValueError("未找到可编码的分类列，请通过 columns 参数指定")
        return obj_cols


class MultiEncoder:
    """多策略编码器，可对不同列使用不同的编码策略。"""

    def __init__(
        self,
        strategy: Dict[str, str],
        max_categories: int = 100,
        high_cardinality_strategy: str = "auto",
    ):
        """
        初始化多策略编码器。

        Args:
            strategy: 策略字典，键为列名，值为编码类型 ("onehot" 或 "label")
            max_categories: 独热编码最大类别数阈值，超过则触发高基数处理
            high_cardinality_strategy: 高基数处理策略 ("auto"、"warn"、"raise")
        """
        self.strategy = strategy
        self.max_categories = max_categories
        self.high_cardinality_strategy = high_cardinality_strategy
        self.encoders: Dict[str, CategoricalEncoder] = {}

        for col, enc_type in strategy.items():
            if enc_type not in ("onehot", "label"):
                raise ValueError(f"列 {col} 的编码类型必须是 'onehot' 或 'label'")
            self.encoders[col] = CategoricalEncoder(
                columns=[col],
                encoding_type=enc_type,
                max_categories=max_categories,
                high_cardinality_strategy=high_cardinality_strategy,
            )

    def fit(self, data: pd.DataFrame) -> "MultiEncoder":
        """学习所有列的编码映射。"""
        for col in self.strategy.keys():
            if col not in data.columns:
                raise ValueError(f"数据中缺少列: {col}")
            self.encoders[col].fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """对所有列按策略编码。"""
        result = data.copy()
        for col in self.strategy.keys():
            encoder = self.encoders[col]
            result = encoder.transform(result)
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def save(self, dir_path: str) -> None:
        """将所有编码器保存到目录。"""
        os.makedirs(dir_path, exist_ok=True)
        for col, encoder in self.encoders.items():
            encoder.save(os.path.join(dir_path, f"{col}.pkl"))
        meta = {
            "strategy": self.strategy,
            "max_categories": self.max_categories,
            "high_cardinality_strategy": self.high_cardinality_strategy,
        }
        with open(os.path.join(dir_path, "strategy.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "MultiEncoder":
        """从目录加载多策略编码器。"""
        with open(os.path.join(dir_path, "strategy.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        strategy = meta["strategy"]
        max_categories = meta.get("max_categories", 100)
        high_cardinality_strategy = meta.get("high_cardinality_strategy", "auto")
        encoder = cls(
            strategy=strategy,
            max_categories=max_categories,
            high_cardinality_strategy=high_cardinality_strategy,
        )
        for col in strategy.keys():
            encoder.encoders[col] = CategoricalEncoder.load(os.path.join(dir_path, f"{col}.pkl"))
        return encoder
