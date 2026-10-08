class Node:
    """标量计算图节点，支持基于链式法则的反向传播。

    parents 为 (父节点, 局部梯度) 元组列表。

    关于局部梯度（∂本节点值 / ∂父节点值）：
    1. 局部梯度就是偏导数的直观含义——父节点的值变一点点，本节点的值跟着变
       多少（∂z/∂x = 4 ⇒ x 增 ε，z 约增 4ε）。它在建图（前向计算）时就随
       parents 一起存好；连线本身没有任何参数，边只记录"谁参与了谁的运算"。
    2. 例如 z = x*y：∂z/∂x = y —— 边上的局部梯度其实是另一个父节点的值，
       而不是独立的"边权重"。只有仿射层 y = Wx+b 里 ∂y/∂x = W 碰巧等于
       权重矩阵，那是线性运算的特例，不是局部梯度的定义。
    3. 神经网络要学习的 w 在计算图里是叶节点（和数据 x 一样参与运算），
       不是边的属性。反向传播后 w.grad 就是 ∂L/∂w，再做梯度下降即"学习"：

           w = Node(0.0)
           pred = Node(w.value * x.value, [(w, x.value), (x, w.value)])
           loss.backward()          # 之后 w.grad == ∂L/∂w
           w.value -= lr * w.grad   # 梯度下降更新（完整示例见 backward_mse.py）
    """

    def __init__(self, value, parents=None):
        self.value = value
        self.parents = parents if parents else [] # parents 是 (父节点, 局部梯度) 元组列表，局部梯度就是当前节点对其父节点的梯度
        self.grad = 0.0 # 就是 L 对当前节点的梯度

    def backward(self):
        """反向传播（迭代版）：按拓扑逆序传播，每个节点只处理一次"""
        topo_order = self._topological_order()

        for node in topo_order:
            node.grad = 0.0
        self.grad = 1.0

        for node in reversed(topo_order):
            for parent, local_grad in node.parents:
                parent.grad += node.grad * local_grad

    def backward_recursive(self):
        """反向传播（递归版）：沿每条路径递归传播梯度，多路径自动累加"""
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
