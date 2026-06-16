import pandas as pd
import os
import shutil
from categorical_encoder import CategoricalEncoder, MultiEncoder


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_sample_data():
    """创建示例数据。"""
    data = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "green", "red", None],
        "size": ["S", "M", "L", "M", "S", "XL", "L", "M"],
        "city": ["北京", "上海", "广州", "深圳", "北京", "上海", "广州", "深圳"],
        "price": [10, 20, 30, 40, 50, 60, 70, 80],
    })
    return data


def demo_label_encoding():
    """演示标签编码。"""
    print_section("1. 标签编码 (Label Encoding)")

    data = create_sample_data()
    print("原始数据:")
    print(data)

    encoder = CategoricalEncoder(columns=["color", "size"], encoding_type="label")
    encoded = encoder.fit_transform(data)

    print("\n编码后数据:")
    print(encoded)

    print("\n编码映射:")
    for col, mapping in encoder.get_mapping().items():
        print(f"  {col}: {mapping}")

    print("\n反向解码:")
    decoded = encoder.inverse_transform(encoded)
    print(decoded[["color", "size"]])


def demo_onehot_encoding():
    """演示独热编码。"""
    print_section("2. 独热编码 (One-Hot Encoding)")

    data = create_sample_data()
    print("原始数据:")
    print(data[["color", "city", "price"]])

    encoder = CategoricalEncoder(columns=["color", "city"], encoding_type="onehot")
    encoded = encoder.fit_transform(data)

    print("\n编码后数据:")
    print(encoded)

    print("\n反向解码:")
    decoded = encoder.inverse_transform(encoded)
    print(decoded[["color", "city", "price"]])


def demo_auto_detect_columns():
    """演示自动识别分类列。"""
    print_section("3. 自动识别分类列编码")

    data = create_sample_data()
    print("原始数据 dtypes:")
    print(data.dtypes)

    encoder = CategoricalEncoder(encoding_type="label")
    encoded = encoder.fit_transform(data)

    print("\n自动编码后 (所有 object/category 类型列):")
    print(encoded)


def demo_new_data_transform():
    """演示用已训练编码器转换新数据（含未见类别）。"""
    print_section("4. 新数据转换（含未见类别处理）")

    train_data = pd.DataFrame({
        "color": ["red", "blue", "green"],
        "size": ["S", "M", "L"],
    })
    print("训练数据:")
    print(train_data)

    encoder = CategoricalEncoder(encoding_type="label")
    encoder.fit(train_data)

    new_data = pd.DataFrame({
        "color": ["red", "yellow", "blue"],
        "size": ["M", "XXL", "S"],
    })
    print("\n新数据 (含未见类别 yellow, XXL):")
    print(new_data)

    encoded = encoder.transform(new_data)
    print("\n新数据编码 (未见类别标记为 -1):")
    print(encoded)


def demo_save_load():
    """演示编码器的保存与加载。"""
    print_section("5. 编码器的保存与加载")

    data = create_sample_data()

    encoder = CategoricalEncoder(columns=["color", "size"], encoding_type="label")
    encoder.fit_transform(data)

    os.makedirs("temp_encoder", exist_ok=True)

    encoder.save("temp_encoder/encoder.pkl")
    encoder.save_json("temp_encoder/encoder.json")

    print("已保存编码器到 temp_encoder/ 目录")

    loaded_encoder = CategoricalEncoder.load("temp_encoder/encoder.pkl")
    print("\n加载编码器后对数据转换:")
    encoded = loaded_encoder.transform(data[["color", "size"]])
    print(encoded)

    with open("temp_encoder/encoder.json", "r", encoding="utf-8") as f:
        import json
        json_content = json.load(f)
    print("\nJSON 映射内容:")
    print(json.dumps(json_content, ensure_ascii=False, indent=2))

    shutil.rmtree("temp_encoder", ignore_errors=True)


def demo_multi_encoder():
    """演示多策略编码器。"""
    print_section("6. 多策略编码器 (MultiEncoder)")

    data = create_sample_data()
    print("原始数据:")
    print(data[["color", "size", "city"]])

    strategy = {
        "color": "onehot",
        "size": "label",
        "city": "label",
    }
    print(f"\n编码策略: {strategy}")

    encoder = MultiEncoder(strategy)
    encoded = encoder.fit_transform(data)

    print("\n编码后数据:")
    print(encoded)

    os.makedirs("temp_multi", exist_ok=True)
    encoder.save("temp_multi")
    print("\n已保存多策略编码器到 temp_multi/ 目录")

    loaded = MultiEncoder.load("temp_multi")
    print("加载后对数据重新编码验证一致:")
    re_encoded = loaded.transform(data)
    print(re_encoded.equals(encoded))

    shutil.rmtree("temp_multi", ignore_errors=True)


def demo_high_cardinality_auto():
    """演示高基数列自动降级为标签编码。"""
    print_section("7. 高基数自动降级 (max_categories=5)")

    import numpy as np
    n_rows = 20
    data = pd.DataFrame({
        "category_low": np.random.choice(["A", "B", "C"], size=n_rows),
        "category_high": [f"cat_{i}" for i in range(n_rows)],
        "value": list(range(n_rows)),
    })
    print(f"原始数据 (category_low: {data['category_low'].nunique()} 个类别, "
          f"category_high: {data['category_high'].nunique()} 个类别):")
    print(data.head(10))

    print("\n使用 onehot 编码，max_categories=5，超过阈值自动降级为 label")
    import warnings
    warnings.filterwarnings("always")
    encoder = CategoricalEncoder(
        columns=["category_low", "category_high"],
        encoding_type="onehot",
        max_categories=5,
        high_cardinality_strategy="auto",
    )
    encoded = encoder.fit_transform(data)

    print("\n编码后数据形状:", encoded.shape)
    print("各列实际编码类型:", encoder.get_actual_encoding())
    print("\n编码后数据 (前 10 行):")
    print(encoded.head(10))

    print("\n反向解码验证:")
    decoded = encoder.inverse_transform(encoded)
    print(decoded[["category_low", "category_high"]].head(10))


def demo_high_cardinality_raise():
    """演示高基数列触发错误。"""
    print_section("8. 高基数 raise 策略 (主动报错)")

    data = pd.DataFrame({
        "category_high": [f"item_{i}" for i in range(150)],
    })
    print(f"数据: {len(data)} 行，{data['category_high'].nunique()} 个唯一类别")

    try:
        encoder = CategoricalEncoder(
            encoding_type="onehot",
            max_categories=100,
            high_cardinality_strategy="raise",
        )
        encoder.fit(data)
    except ValueError as e:
        print(f"\n触发 ValueError (预期行为):\n  {e}")


def main():
    print("分类变量编码服务 - 使用示例")
    print("Python + Pandas 实现")

    demo_label_encoding()
    demo_onehot_encoding()
    demo_auto_detect_columns()
    demo_new_data_transform()
    demo_save_load()
    demo_multi_encoder()
    demo_high_cardinality_auto()
    demo_high_cardinality_raise()

    print_section("全部演示完成")


if __name__ == "__main__":
    main()
