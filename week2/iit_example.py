import numpy as np
import itertools

def entropy(prob):
    """シャノンエントロピーの計算"""
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def joint_entropy(prob_matrix):
    """2変数以上の同時分布に対するエントロピー"""
    return entropy(prob_matrix.flatten())

def conditional_entropy(prob_joint, prob_marginal):
    """条件付きエントロピー H(Y|X) = H(X, Y) - H(X)"""
    # H(X, Y)
    h_joint = joint_entropy(prob_joint)
    # H(X)
    h_marginal = entropy(prob_marginal)
    return h_joint - h_marginal

def mutual_information(prob_joint):
    """相互情報量 I(X; Y) = H(X) + H(Y) - H(X, Y)"""
    prob_x = np.sum(prob_joint, axis=1)
    prob_y = np.sum(prob_joint, axis=0)
    h_x = entropy(prob_x)
    h_y = entropy(prob_y)
    h_joint = joint_entropy(prob_joint)
    return h_x + h_y - h_joint

class IITSimpleSystem:
    """
    3つのバイナリ素子 (A, B, C) からなる簡単なダイナミカルシステムの
    簡易的な統合情報量 (Phi) を計算するクラス。
    """
    def __init__(self, noise=0.05):
        # 状態空間: 000 (0) から 111 (7)
        self.num_nodes = 3
        self.states = list(itertools.product([0, 1], repeat=self.num_nodes))
        self.num_states = len(self.states)
        self.noise = noise
        self.tpm = self._build_transition_probability_matrix()

    def _next_state_logic(self, current_state):
        """
        システムの論理ルールを定義。
        A(t+1) = B(t) OR C(t)
        B(t+1) = A(t) AND C(t)
        C(t+1) = NOT A(t)
        """
        a, b, c = current_state
        next_a = int(b or c)
        next_b = int(a and c)
        next_c = int(not a)
        return (next_a, next_b, next_c)

    def _build_transition_probability_matrix(self):
        """
        ノイズを含めた遷移確率行列 (TPM) を構築。
        tpm[i, j] = P(State(t+1) = j | State(t) = i)
        """
        tpm = np.zeros((self.num_states, self.num_states))
        for i, curr_state in enumerate(self.states):
            target_state = self._next_state_logic(curr_state)
            for j, next_state in enumerate(self.states):
                # ターゲット状態と一致しないビット数（ハミング距離）を計算
                diff_bits = sum(s1 != s2 for s1, s2 in zip(target_state, next_state))
                # 各ビットが独立に noise の確率で反転すると仮定
                prob = ((1 - self.noise) ** (self.num_nodes - diff_bits)) * (self.noise ** diff_bits)
                tpm[i, j] = prob
        return tpm

    def calculate_phi(self):
        """
        簡易的なシステム統合情報量 (Phi) を計算する。
        Phi = I(X_t; X_{t+1}) - I_MIP(X_t; X_{t+1})
        """
        # 1. 状態の事前分布（一様分布と仮定）
        p_xt = np.ones(self.num_states) / self.num_states
        
        # 2. 同時確率分布 P(X_t, X_{t+1}) の計算
        # P(X_t = i, X_{t+1} = j) = P(X_t = i) * P(X_{t+1} = j | X_t = i)
        p_joint = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            p_joint[i, :] = p_xt[i] * self.tpm[i, :]
            
        # 3. システム全体の相互情報量 I(X_t; X_{t+1})
        i_whole = mutual_information(p_joint)
        
        print(f"システム全体の一連のエントロピー/情報量:")
        print(f"  H(X_t)       = {entropy(p_xt):.4f} bits")
        p_xt1 = np.sum(p_joint, axis=0)
        print(f"  H(X_t+1)     = {entropy(p_xt1):.4f} bits")
        print(f"  H(X_t,Xt+1)  = {joint_entropy(p_joint):.4f} bits")
        print(f"  I(X_t; X_t+1)= {i_whole:.4f} bits (システム全体の情報量)\n")

        # 4. 可能なすべての二分割 (Bipartitions) を定義
        # A, B, C のノードインデックスはそれぞれ 0, 1, 2
        # 自明でない一意な二分割は以下の3通り:
        # P1: {0} と {1, 2}
        # P2: {1} と {0, 2}
        # P3: {2} と {0, 1}
        partitions = [
            (([0], [1, 2]), "Part1: {A} ⊥ {B, C}"),
            (([1], [0, 2]), "Part2: {B} ⊥ {A, C}"),
            (([2], [0, 1]), "Part3: {C} ⊥ {A, B}")
        ]

        mip_partition = None
        min_i_parts = float('inf')
        mip_phi = 0.0

        print("各二分割における相互情報量の合計 (I_P) の計算:")
        for (part_l, part_r), label in partitions:
            # 各パーツごとの相互情報量を個別に計算する
            # 例: part = [0] (ノード A) のとき、P(A_t, A_{t+1}) を求める。
            i_part_l = self._marginal_mutual_information(p_joint, part_l)
            i_part_r = self._marginal_mutual_information(p_joint, part_r)
            
            # 分割されたシステムの相互情報量の合計
            i_p = i_part_l + i_part_r
            phi_p = i_whole - i_p
            
            print(f"  {label}:")
            print(f"    I(part_L) = {i_part_l:.4f} bits, I(part_R) = {i_part_r:.4f} bits")
            print(f"    I_P (合計) = {i_p:.4f} bits")
            print(f"    Φ_P        = {phi_p:.4f} bits")
            
            # 最小情報分割 (MIP) を探索 (最も Φ_P が小さくなる、すなわち最も情報損失の少ない分割)
            if phi_p < min_i_parts:
                min_i_parts = phi_p
                mip_partition = label
                mip_phi = phi_p

        print("\n=== 計算結果 ===")
        print(f"最小情報分割 (MIP): {mip_partition}")
        print(f"統合情報量 (Φ)     : {mip_phi:.4f} bits")
        if mip_phi > 0:
            print("⇒ このシステムは統合されています (システムは部分の和を超えた情報を持っています)。")
        else:
            print("⇒ このシステムは統合されていません (完全に分解可能です)。")

    def _marginal_mutual_information(self, p_joint, nodes):
        """
        指定されたノード群 (例: [0, 2] は A と C) に対する周辺相互情報量 I(Nodes_t; Nodes_{t+1}) を計算する。
        """
        # 状態から指定ノードの部分状態(0と1の組み合わせ)へのマッピングを構築
        # 例: nodes = [0, 2] の場合、元の状態 (A, B, C) = (1, 0, 1) は 部分状態 (1, 1) に写像される。
        sub_states = list(itertools.product([0, 1], repeat=len(nodes)))
        num_sub_states = len(sub_states)
        
        # 周辺同時確率分布 P(Nodes_t, Nodes_{t+1}) の初期化
        p_marginal_joint = np.zeros((num_sub_states, num_sub_states))
        
        for i, state_t in enumerate(self.states):
            # tにおける部分状態のインデックスを取得
            sub_t = tuple(state_t[n] for n in nodes)
            idx_t = sub_states.index(sub_t)
            
            for j, state_t1 in enumerate(self.states):
                # t+1における部分状態のインデックスを取得
                sub_t1 = tuple(state_t1[n] for n in nodes)
                idx_t1 = sub_states.index(sub_t1)
                
                # 同時確率を加算
                p_marginal_joint[idx_t, idx_t1] += p_joint[i, j]
                
                # numpyの浮動小数点誤差を防ぐため極小値を維持
                
        return mutual_information(p_marginal_joint)

if __name__ == "__main__":
    print("==================================================")
    print(" 統合情報理論 (IIT) 簡易計算デモンストレーション ")
    print("==================================================\n")
    print("3つの相互作用する素子 (A, B, C) のネットワークにおいて、")
    print("最小情報分割 (MIP) と 統合情報量 (Φ) を計算します。\n")
    
    # システムを生成 (ノイズ率 5%)
    system = IITSimpleSystem(noise=0.05)
    system.calculate_phi()
