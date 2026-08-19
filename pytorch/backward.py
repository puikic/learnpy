class Node:
    """标量计算图节点，支持基于链式法则的反向传播。

    parents 为 (父节点, 本节点对该父节点的局部梯度) 元组列表。
    """

    def __init__(self, value, parents=None):
        self.value = value
        self.parents = parents if parents else []
        self.grad = 0.0

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
