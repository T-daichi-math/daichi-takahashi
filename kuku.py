def print_kuku():
    """九九の表をきれいに整列して表示する関数"""
    for i in range(1, 10):
        row = []
        for j in range(1, 10):
            # 各数値を2桁で右寄せにしてスペースで区切る
            row.append(f"{i * j:2d}")
        print(" ".join(row))

if __name__ == "__main__":
    print_kuku()
