import pandas as pd
import pickle
import json
import os
from typing import List, Dict, Union, Optional


class CategoricalEncoder:
    """分类变量编码器，支持独热编码和标签编码。"""

    def __init__(self, columns: Optional[List[str]] = None, encoding_type: str = "onehot"):
        """
        初始化编码器。

        Args:
            columns: 需要编码的列名列表，若为 None 则自动识别所有对象类型列
            encoding_type: 编码类型，"onehot" 或 "label"
        """
        if encoding_type not in ("onehot", "label"):
            raise ValueError("encoding_type 必须是 'onehot' 或 'label'")

        self.columns = columns
        self.encoding_type = encoding_type
        self._fitted = False
        self._mapping: Dict[str, Dict] = {}
        self._onehot_columns: Dict[str, List[str]] = {}

    def fit(self, data: pd.DataFrame) -> "CategoricalEncoder":
        """
        从数据中学习编码映射。

        Args:
            data: 输入的 DataFrame

        Returns:
            self
        """
        target_cols = self._resolve_columns(data)

        if self.encoding_type == "label":
            for col in target_cols:
                unique_values = sorted(data[col].dropna().unique().tolist())
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self._mapping[col] = mapping
        else:
            for col in target_cols:
                unique_values = sorted(data[col].dropna().unique().tolist())
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
        target_cols = [col for col in self._mapping.keys() if col in result.columns]

        if self.encoding_type == "label":
            for col in target_cols:
                mapping = self._mapping[col]
                result[col] = result[col].map(
                    lambda x: mapping.get(x, -1) if pd.notna(x) else x
                )
        else:
            for col in target_cols:
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

        if self.encoding_type == "label":
            for col, mapping in self._mapping.items():
                if col in result.columns:
                    reverse_map = {v: k for k, v in mapping.items()}
                    result[col] = result[col].map(
                        lambda x: reverse_map.get(int(x), x) if pd.notna(x) and int(x) != -1 else x
                    )
        else:
            for col, encoded_cols in self._onehot_columns.items():
                if all(ec in result.columns for ec in encoded_cols):
                    values = list(self._mapping[col].keys())
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

    def save(self, filepath: str) -> None:
        """
        保存编码器到文件（pickle 格式）。

        Args:
            filepath: 保存路径
        """
        state = {
            "columns": self.columns,
            "encoding_type": self.encoding_type,
            "_fitted": self._fitted,
            "_mapping": self._mapping,
            "_onehot_columns": self._onehot_columns,
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

        encoder = cls(columns=state["columns"], encoding_type=state["encoding_type"])
        encoder._fitted = state["_fitted"]
        encoder._mapping = state["_mapping"]
        encoder._onehot_columns = state["_onehot_columns"]
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

    def __init__(self, strategy: Dict[str, str]):
        """
        初始化多策略编码器。

        Args:
            strategy: 策略字典，键为列名，值为编码类型 ("onehot" 或 "label")
        """
        self.strategy = strategy
        self.encoders: Dict[str, CategoricalEncoder] = {}

        for col, enc_type in strategy.items():
            if enc_type not in ("onehot", "label"):
                raise ValueError(f"列 {col} 的编码类型必须是 'onehot' 或 'label'")
            self.encoders[col] = CategoricalEncoder(columns=[col], encoding_type=enc_type)

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
        with open(os.path.join(dir_path, "strategy.json"), "w", encoding="utf-8") as f:
            json.dump(self.strategy, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dir_path: str) -> "MultiEncoder":
        """从目录加载多策略编码器。"""
        with open(os.path.join(dir_path, "strategy.json"), "r", encoding="utf-8") as f:
            strategy = json.load(f)
        encoder = cls(strategy)
        for col in strategy.keys():
            encoder.encoders[col] = CategoricalEncoder.load(os.path.join(dir_path, f"{col}.pkl"))
        return encoder
