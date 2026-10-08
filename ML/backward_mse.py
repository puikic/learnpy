"""MSE 损失的反向传播版本（基于 backward.py 改造，原文件未动）。

Loss 换成均方误差：
    MSE = (1/n) * Σ (pred_i - target_i)^2

改造要点：
1. 新增基础算子 add/sub/mul/square：任何运算都返回计算图节点，图可以任意拼装；
2. mse_loss() 把 n 条平方误差路径拼成一个损失节点，对其调用 backward_recursive()，
   链式法则自动得到 dMSE/dpred_i = 2*(pred_i - target_i)/n；
3. backward_recursive() 相比原版先清零所有可达节点的梯度，
   便于在训练循环里反复调用；
4. 参数（w、b）出现在多条路径上时，梯度靠 += 自动累加（菱形依赖）。
"""


class Node:
    """标量计算图节点，支持基于链式法则的反向传播。

    parents 为 (父节点, 本节点对该父节点的局部梯度) 元组列表。
    """

    def __init__(self, value, parents=None):
        self.value = value
        self.parents = parents if parents else []
        self.grad = 0.0

    def backward_recursive(self):
        """反向传播（递归版）：先清零可达节点的梯度，再沿每条路径递归传播"""
        for node in self._topological_order():
            node.grad = 0.0
        self.grad = 1.0
        self._propagate(1.0)

    def _topological_order(self):
        """DFS 构建拓扑序（父节点在前，自己在后）"""
        order = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def _propagate(self, grad):
        """把 grad 沿自己的各条父路径递归下传并累加"""
        for parent, local_grad in self.parents:
            g = grad * local_grad
            parent.grad += g
            parent._propagate(g)


# ---------------- 基础算子：把常用运算也建成计算图节点 ----------------

def add(a, b):
    """a + b"""
    return Node(a.value + b.value, [(a, 1.0), (b, 1.0)])


def sub(a, b):
    """a - b"""
    return Node(a.value - b.value, [(a, 1.0), (b, -1.0)])


def mul(a, b):
    """a * b"""
    return Node(a.value * b.value, [(a, b.value), (b, a.value)])


def square(a):
    """a ** 2"""
    return Node(a.value ** 2, [(a, 2.0 * a.value)])


# ---------------- MSE 损失 ----------------

def mse_loss(preds, targets):
    """均方误差：MSE = (1/n) * Σ (pred_i - target_i)^2

    preds   : 预测值节点列表（梯度沿图反传回各参数）
    targets : 真实值节点列表（常数节点，也会收到梯度但不使用）
    返回代表 MSE 的节点，对其调用 backward_recursive() 即可。
    """
    n = len(preds)
    squared = [square(sub(p, t)) for p, t in zip(preds, targets)]
    # MSE = (1/n) * Σ squared_i，故每个 squared_i 的局部梯度是 1/n
    return Node(sum(s.value for s in squared) / n,
                [(s, 1.0 / n) for s in squared])


def _closed_form_grads(preds, targets):
    """手工推导的解析梯度：dMSE/dpred_i = 2*(pred_i - target_i)/n"""
    n = len(preds)
    return [2.0 * (p.value - t.value) / n for p, t in zip(preds, targets)]


def main():
    # ---- 1. 梯度核对：图上反传 vs 手工公式 ----
    preds = [Node(2.0), Node(5.0), Node(-1.0)]
    targets = [Node(1.0), Node(7.0), Node(0.0)]
    loss = mse_loss(preds, targets)
    print(f"== 梯度核对 ==  MSE = {loss.value}")

    loss.backward_recursive()
    grads_rec = [p.grad for p in preds]

    grads_formula = _closed_form_grads(preds, targets)

    for i in range(len(preds)):
        print(f"pred[{i}]: 递归={grads_rec[i]:+.4f}  公式={grads_formula[i]:+.4f}")
    assert all(abs(g - f) < 1e-9 for g, f in zip(grads_rec, grads_formula))
    print("递归反传与手工公式全部一致\n")

    # ---- 2. 用 MSE 反向传播做梯度下降：拟合 y = 2x + 1 ----
    xs = [Node(x) for x in [1.0, 2.0, 3.0, 4.0]]
    ys = [Node(y) for y in [3.0, 5.0, 7.0, 9.0]]
    w, b = Node(0.0), Node(0.0)

    lr, epochs = 0.05, 300
    print("== 梯度下降拟合 y = 2x + 1 ==")
    for epoch in range(epochs + 1):
        preds = [add(mul(w, x), b) for x in xs]
        loss = mse_loss(preds, ys)

        loss.backward_recursive()  # w.grad、b.grad 自动累加 4 条样本路径的梯度

        w.value -= lr * w.grad
        b.value -= lr * b.grad

        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss={loss.value:8.4f}  "
                  f"w={w.value:+.4f}  b={b.value:+.4f}")

    print(f"拟合结果: y ≈ {w.value:.3f}*x + {b.value:.3f}（期望 2x + 1）")


if __name__ == "__main__":
    main()
