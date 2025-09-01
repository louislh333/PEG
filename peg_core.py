# peg_pipeline/peg_core.py
import math
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from pyvis.network import Network as PyVisNetwork
from joblib import Parallel, delayed
from tqdm import tqdm
from .peg_vis import create_simple_html_vis


def compute_node_relevance(W: dict, id_to_activity: dict):
    """
    计算节点重要性，基于边权重的绝对值之和
    """
    rel = defaultdict(float)
    for (u, v_idx), w in W.items():
        v = id_to_activity.get(v_idx, str(v_idx))
        rel[u] += abs(w)
        rel[v] += abs(w)
    return dict(rel)


def agg_attention_matrix(att_mats, method="mean"):
    if att_mats is None or len(att_mats) == 0:
        return None
    arr = np.array(att_mats)
    if arr.ndim == 3:
        if method == "mean":
            return np.mean(arr, axis=0)
        else:
            return np.max(arr, axis=0)
    elif arr.ndim == 4:
        if method == "mean":
            return np.mean(arr, axis=(0, 1))
        else:
            return np.max(arr, axis=(0, 1))
    else:
        return None


def compute_activity_attention(prefix, A, id_to_activity):
    if A is None:
        return {}
    col_sum = np.sum(A, axis=0)
    activity_scores = defaultdict(float)
    for pos, aid in enumerate(prefix):
        act = id_to_activity.get(aid, str(aid))
        activity_scores[act] += float(col_sum[pos])
    total = sum(abs(v) for v in activity_scores.values()) + 1e-12
    for k in activity_scores:
        activity_scores[k] = activity_scores[k] / total
    return dict(activity_scores)


def single_prefix_analysis(prefix, model_wrapper, id_to_activity, cfg):
    max_len = cfg.get("max_seq_len", 100) - 1
    if len(prefix) > max_len:
        prefix = prefix[:max_len]

    # # 调试信息：检查输入数据
    # print(f"处理前缀长度: {len(prefix)}")
    # print(f"前缀内容: {prefix[:10]}...")  # 只显示前10个

    try:
        proba = model_wrapper.predict_proba([prefix], batch_size=cfg.get("batch_size", 64))[0]
        # print(f"预测概率数组长度: {len(proba)}")
    except Exception as e:
        print(f"预测失败: {e}")
        return []

    att_mats = model_wrapper.get_attention_matrices(prefix)
    A = agg_attention_matrix(att_mats, method=cfg.get("att_agg_method", "mean"))
    activity_scores = compute_activity_attention(prefix, A, id_to_activity) if A is not None else {}

    # 过滤预测结果，排除无效索引
    Pr_idx = [i for i, p in enumerate(proba)
              if p > cfg.get("delta_pred", 0.02)
              and i in id_to_activity]  # 确保索引在映射字典中

    # print(f"有效预测索引数量: {len(Pr_idx)}")

    Ar = []
    if activity_scores:
        # 过滤掉无效活动（如 "0" 等）
        valid_activities = {k: v for k, v in activity_scores.items()
                            if k in id_to_activity.values() and str(k) != "0"}

        if valid_activities:
            acts = list(valid_activities.keys())
            scores = np.array([valid_activities[a] for a in acts])
            if len(scores) > 0:
                thr = np.percentile(scores, cfg.get("delta_att_pctile", 75))
                Ar = [acts[i] for i in range(len(acts)) if scores[i] > thr]

    # 如果没有注意力活动，使用最近的有效活动
    if not Ar and len(prefix) > 0:
        recent_positions = min(3, len(prefix))
        recent_activities = []
        for i in range(1, recent_positions + 1):
            aid = prefix[-i]
            act = id_to_activity.get(aid, str(aid))
            # 只添加有效活动
            if act in id_to_activity.values() and str(act) != "0":
                recent_activities.append(act)
        Ar = list(set(recent_activities))

    # print(f"重要活动 (Ar): {Ar}")

    results = []
    positions_by_act = defaultdict(list)
    for pos, aid in enumerate(prefix):
        act = id_to_activity.get(aid, str(aid))
        if act in Ar and str(act) != "0":  # 过滤无效活动
            positions_by_act[act].append(pos)

    if not positions_by_act:
        # print("没有找到有效的活动位置")
        return []

    p_base = proba
    for u_act, pos_list in positions_by_act.items():
        per_v_deltas = defaultdict(list)
        for pos in pos_list:
            try:
                p_mask_input = model_wrapper.predict_with_embedding_zero(prefix, [pos])
                p_mask_att = model_wrapper.predict_with_attention_mask(prefix, [pos])
                for v in Pr_idx:
                    if v < len(p_base) and v < len(p_mask_input) and v < len(p_mask_att):
                        delta_input = float(p_base[v] - p_mask_input[v])
                        delta_att = float(p_base[v] - p_mask_att[v])
                        delta = (delta_input + delta_att) / 2.0
                        per_v_deltas[v].append(delta)
            except Exception as e:
                print(f"掩蔽预测失败: {e}")
                continue

        for v, vals in per_v_deltas.items():
            if len(vals) == 0:
                continue
            avg_d = float(np.mean(vals))
            att_score = activity_scores.get(u_act, 0.1)
            s_loc = math.copysign(abs(avg_d) * att_score, avg_d)
            sig = 1 if abs(avg_d) > cfg.get("delta_effect", 0.01) else 0

            # 确保 v 是有效的活动索引
            if v in id_to_activity:
                results.append((u_act, v, s_loc, sig))

    # print(f"单前缀分析结果数量: {len(results)}")
    return results


def aggregate_local_results(local_results):
    S = defaultdict(float)
    C = defaultdict(int)
    for res in local_results:
        for u_act, v, s_loc, sig in res:
            if sig == 0:
                continue
            S[(u_act, v)] += s_loc
            C[(u_act, v)] += 1
    return S, C


def normalize_and_prune(S: dict, C: dict, cfg: dict, total_prefixes: int, id_to_activity: dict):
    print(f" 原始候选边: {len(S)}")

    # 过滤无效边：包含 "0" 或不在活动映射中的边
    valid_activities = set(id_to_activity.values())
    S_valid = {}
    C_valid = {}

    for (u, v), weight in S.items():
        # 检查 u 是否为有效活动名称
        u_valid = u in valid_activities and str(u) != "0"
        # 检查 v 是否为有效活动索引
        v_valid = v in id_to_activity

        if u_valid and v_valid:
            # 将 v 转换为活动名称进行一致性检查
            v_act = id_to_activity[v]
            if str(v_act) != "0":
                S_valid[(u, v)] = weight
                C_valid[(u, v)] = C.get((u, v), 0)

    print(f" 过滤无效活动后的边: {len(S_valid)}")

    min_support = cfg.get("min_support", 5)
    min_support_ratio = cfg.get("min_support_ratio", 0.01)
    min_count = max(min_support, int(total_prefixes * min_support_ratio))
    print(f" 支持度阈值: min_support={min_support}, min_support_ratio={min_support_ratio}, min_count={min_count}")

    # 调试：输出支持度分布
    if C_valid:
        supports = list(C_valid.values())
        print(f" 支持度分布: 最小={min(supports)}, 最大={max(supports)}, 平均={np.mean(supports):.2f}")
        plt.figure(figsize=(8, 6))
        plt.hist(supports, bins=30, color='lightgreen', edgecolor='black')
        plt.title("Support Distribution")
        plt.xlabel("Support Count")
        plt.ylabel("Frequency")
        plt.savefig("support_distribution.png", dpi=500)
        plt.close()
        print("支持度分布图保存到: support_distribution.png")

    S_filt = {k: v for k, v in S_valid.items() if C_valid.get(k, 0) >= min_count}
    print(f" 满足支持度阈值 ({min_count}) 的边: {len(S_filt)}")

    if not S_filt:
        print(" 警告：没有边满足支持度要求")
        return {}

    abs_vals = np.array([abs(v) for v in S_filt.values()])
    max_abs = float(abs_vals.max()) if abs_vals.size else 1.0
    W_norm = {k: float(v / (max_abs + 1e-12)) for k, v in S_filt.items()}
    abs_norm = np.array([abs(v) for v in W_norm.values()])

    if abs_norm.size == 0:
        return {}

    original_pctile = cfg.get("prune_edge_pctile", 75)
    thr = np.percentile(abs_norm, original_pctile)
    W_pruned = {k: v for k, v in W_norm.items() if abs(v) > thr}

    attempts = 0
    while len(W_pruned) == 0 and original_pctile > 10 and attempts < 5:
        original_pctile -= 15
        thr = np.percentile(abs_norm, original_pctile)
        W_pruned = {k: v for k, v in W_norm.items() if abs(v) > thr}
        attempts += 1
        if len(W_pruned) > 0:
            print(f" 自动放宽剪枝到 pctile {original_pctile}")

    if len(W_pruned) == 0 and len(W_norm) > 0:
        k = min(20, len(W_norm))
        sorted_items = sorted(W_norm.items(), key=lambda x: abs(x[1]), reverse=True)
        W_pruned = dict(sorted_items[:k])
        print(f" 保留 top-{k} 边")

    print(f" 最终剪枝阈值: {thr:.4f}, 保留边数: {len(W_pruned)}")

    # # 调试信息：打印最终保留的边
    # print("\n最终保留的边:")
    # for (u, v), w in sorted(W_pruned.items(), key=lambda x: abs(x[1]), reverse=True):
    #     v_act = id_to_activity.get(v, f"未知_{v}")
    #     effect = "促进" if w > 0 else "抑制"
    #     print(f"  {u} -> {v_act}: {w:.4f} ({effect})")

    return W_pruned


def propagate_refine(W: dict, cfg: dict):
    prop_alpha = cfg.get("propagation_alpha", 0.1)
    prop_iters = cfg.get("prop_iters", 5)
    w_max = cfg.get("w_max", 1.0)
    W_new = W.copy()
    for t in range(prop_iters):
        rel = defaultdict(float)
        for (u, v), w in W_new.items():
            rel[v] += abs(w)
        max_rel = max(rel.values()) if rel else 1.0
        norm_rel = {k: math.tanh(v / (max_rel + 1e-12)) for k, v in rel.items()}
        W_new = {
            k: min(max(w + prop_alpha * w * norm_rel.get(k[1], 0.0), -w_max), w_max)
            for k, w in W_new.items()
        }
    return W_new


def export_graph(W: dict, id_to_activity: dict, out_dir: str, cfg: dict):
    G = nx.DiGraph()
    valid_edges = 0
    invalid_edges = 0

    for (u_act, v_idx), w in W.items():
        v_act = id_to_activity.get(v_idx, f"未知活动_{v_idx}")

        # 过滤无效边
        if str(u_act) == "0" or str(v_act) == "0" or "未知活动" in str(v_act):
            invalid_edges += 1
            continue

        G.add_edge(u_act, v_act, weight=w)
        valid_edges += 1

    if G.number_of_edges() == 0:
        print("警告：图中没有有效边，跳过可视化")
        return None

    # 计算节点重要性用于动态节点大小
    node_rel = compute_node_relevance(W, id_to_activity)
    max_rel = max(node_rel.values()) if node_rel else 1.0
    node_sizes = {n: 500 + 1000 * (node_rel.get(n, 0.0) / max_rel) for n in G.nodes()}

    def calculate_edge_label_position(u_pos, v_pos, offset_factor=0.0):
        """计算边标签的最佳位置，避免重叠"""
        mid_x = (u_pos[0] + v_pos[0]) / 2
        mid_y = (u_pos[1] + v_pos[1]) / 2

        # 计算边的方向向量
        dx = v_pos[0] - u_pos[0]
        dy = v_pos[1] - u_pos[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length > 0:
            # 垂直于边的方向偏移
            perp_x = -dy / length * offset_factor
            perp_y = dx / length * offset_factor
            return (mid_x + perp_x, mid_y + perp_y)
        else:
            return (mid_x, mid_y)

    # NetworkX可视化 - 修复边粗细和标签问题
    try:
        plt.figure(figsize=(20, 16))

        # 使用更好的布局算法，增加节点间距
        pos = nx.spring_layout(G, seed=42, k=4, iterations=100)

        # 获取所有边的权重
        edges_data = list(G.edges(data=True))
        weights = np.array([abs(d["weight"]) for _, _, d in edges_data])

        if weights.size == 0:
            print("没有边数据")
            return None

        # 计算边的粗细 - 使用更合理的缩放
        max_w = weights.max()
        min_w = weights.min()
        if max_w > min_w:
            # 线性映射到 1.0-10.0 的范围，确保所有边都可见
            edge_widths = 1.0 + (weights - min_w) / (max_w - min_w) * 9.0
        else:
            edge_widths = np.full(len(weights), 3.0)

        # 边颜色
        edge_colors = ['green' if d["weight"] > 0 else 'red' for _, _, d in edges_data]

        # 创建节点标签，不显示概率值
        node_labels = {n: n for n in G.nodes()}

        # 绘制节点
        nx.draw_networkx_nodes(G, pos,
                               node_size=[node_sizes[n] for n in G.nodes()],
                               node_color='lightblue', alpha=0.9,
                               linewidths=3, edgecolors='darkblue')

        # 绘制节点标签（只显示名称）
        nx.draw_networkx_labels(G, pos, labels=node_labels,
                                font_size=11, font_weight='bold',
                                font_color='black')

        # 绘制边 - 使用修正的粗细
        nx.draw_networkx_edges(G, pos,
                               arrowstyle='->', arrowsize=25,
                               width=edge_widths,
                               edge_color=edge_colors, alpha=0.8,
                               connectionstyle="arc3,rad=0.15")

        # 绘制所有边标签 - 使用智能标签放置策略
        edge_labels = {}
        for u, v, d in edges_data:
            weight = d['weight']
            edge_labels[(u, v)] = f"{weight:.3f}"

        # 边标签智能放置
        edge_label_positions = {}
        occupied_positions = set()

        for i, (u, v, d) in enumerate(edges_data):
            weight = d['weight']
            label = f"{weight:.3f}"

            # 尝试多个位置找到不重叠的位置
            base_pos = calculate_edge_label_position(pos[u], pos[v], 0)
            offsets = [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
            label_pos = None

            for offset in offsets:
                test_pos = calculate_edge_label_position(pos[u], pos[v], offset)
                # 检查是否与已有位置冲突
                conflict = False
                for occupied in occupied_positions:
                    if abs(test_pos[0] - occupied[0]) < 0.15 and abs(test_pos[1] - occupied[1]) < 0.15:
                        conflict = True
                        break

                if not conflict:
                    label_pos = test_pos
                    occupied_positions.add(label_pos)
                    break

            # 如果没找到合适位置，使用基础位置
            if label_pos is None:
                label_pos = base_pos
                occupied_positions.add(label_pos)

            # 绘制边标签
            x, y = label_pos
            bg_color = 'lightgreen' if weight > 0 else 'lightcoral'

            plt.text(x, y, label, fontsize=10, fontweight='bold',
                     ha='center', va='center', zorder=15,
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor=bg_color, alpha=0.98,
                               edgecolor='black', linewidth=1))

        plt.title("Process Explanation Graph (PEG)", fontsize=20, fontweight='bold', pad=40)

        # 添加更详细的图例说明
        legend_text = (f"Green=Promote, Red=Inhibit\n"
                       f"Edges: {G.number_of_edges()}, Nodes: {G.number_of_nodes()}\n"
                       f"Edge width ∝ |weight|\n"
                       f"Weight range: [{min_w:.3f}, {max_w:.3f}]\n"
                       f"Node size ∝ relevance")

        plt.text(0.02, 0.98, legend_text,
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

        plt.axis('off')
        plt.tight_layout()

        png_path = f"{out_dir}/peg.png"
        plt.savefig(png_path, bbox_inches='tight', dpi=400, facecolor='white')
        plt.close()
        print(f"NetworkX 图保存到: {png_path}")

        # 单独生成节点重要性图表
        if node_rel:
            plt.figure(figsize=(12, 8))
            nodes = list(node_rel.keys())
            relevances = [node_rel[n] for n in nodes]

            # 创建条形图显示节点重要性
            bars = plt.bar(range(len(nodes)), relevances,
                           color=['lightblue' if r > np.mean(relevances) else 'lightcoral' for r in relevances],
                           alpha=0.8, edgecolor='black', linewidth=1)

            # 添加数值标签
            for i, (node, rel) in enumerate(zip(nodes, relevances)):
                plt.text(i, rel + max(relevances) * 0.01, f'{rel:.3f}',
                         ha='center', va='bottom', fontweight='bold', fontsize=10)

            plt.xlabel('Nodes', fontsize=14, fontweight='bold')
            plt.ylabel('Relevance Score', fontsize=14, fontweight='bold')
            plt.title('Node Relevance Scores in Process Explanation Graph',
                      fontsize=16, fontweight='bold', pad=20)
            plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')

            # 添加统计信息
            avg_rel = np.mean(relevances)
            plt.axhline(y=avg_rel, color='red', linestyle='--', alpha=0.7,
                        label=f'Average: {avg_rel:.3f}')
            plt.legend()

            plt.tight_layout()
            relevance_path = f"{out_dir}/node_relevance.png"
            plt.savefig(relevance_path, dpi=400, facecolor='white')
            plt.close()
            print(f"节点重要性图表保存到: {relevance_path}")

        # 边权重分布直方图
        if weights.size > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(weights, bins=cfg.get("hist_bins", 30),
                     color='skyblue', edgecolor='black', alpha=0.7)
            plt.title("Edge Weight Distribution", fontsize=14)
            plt.xlabel("Absolute Weight", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = f"{out_dir}/weight_distribution.png"
            plt.savefig(hist_path, dpi=400, facecolor='white')
            plt.close()
            print(f"权重分布直方图保存到: {hist_path}")

    except Exception as e:
        print(f"NetworkX 可视化失败: {e}")
        import traceback
        traceback.print_exc()

    # HTML via pyvis
    html_path = f"{out_dir}/peg.html"
    try:
        net = PyVisNetwork(height="800px", width="100%", directed=True, bgcolor="#ffffff")

        # 添加节点
        for n in G.nodes():
            size = max(20, node_sizes.get(n, 25) / 20.0)
            net.add_node(n, label=n, title=f"Node: {n}\nRelevance: {node_rel.get(n, 0.0):.3f}",
                         size=size,
                         color={'background': '#97C2FC', 'border': '#2B7CE9'})

        # 添加边
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 0.0)
            color = "#00AA00" if w > 0 else "#FF4444"
            abs_w = abs(w)
            if max_w > 0:
                width = max(1, (abs_w / max_w) * 10)
            else:
                width = 2

            net.add_edge(u, v,
                         value=abs_w,
                         title=f"Weight: {w:.4f}\nAbs weight: {abs_w:.4f}",
                         color=color,
                         width=width,
                         arrows={'to': {'enabled': True, 'scaleFactor': 1.5}})

        # 设置物理引擎参数
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springConstant": 0.04,
              "springLength": 200
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "curvedCW",
              "roundness": 0.2
            }
          }
        }
        """)

        net.save_graph(html_path)
        print(f"PyVis 图保存到: {html_path}")

    except Exception as e:
        print(f"PyVis 可视化失败: {e}")
        try:
            html_content = create_simple_html_vis(G, W)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"简单 HTML 图保存到: {html_path}")
        except Exception as e:
            print(f"HTML 可视化失败: {e}")

    # GEXF 输出
    gexf_path = f"{out_dir}/peg_graph.gexf"
    try:
        nx.write_gexf(G, gexf_path, encoding="utf-8")
        print(f"GEXF 图保存到: {gexf_path}")
    except Exception as e:
        print(f"GEXF 导出失败: {e}")

    # 生成边权重详细表格
    try:
        import pandas as pd
        edge_table_data = []
        for u, v, d in G.edges(data=True):
            weight = d['weight']
            edge_table_data.append({
                'Source': u,
                'Target': v,
                'Weight': weight,
                'Effect': 'Promote' if weight > 0 else 'Inhibit',
                'Strength': abs(weight)
            })

        df = pd.DataFrame(edge_table_data)
        df = df.sort_values('Strength', ascending=False)

        table_path = f"{out_dir}/edge_weights_table.csv"
        df.to_csv(table_path, index=False)
        print(f"边权重详细表格保存到: {table_path}")

    except ImportError:
        print("pandas 不可用，跳过 CSV 表格生成")

    return gexf_path