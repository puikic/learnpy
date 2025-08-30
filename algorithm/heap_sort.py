class MaxHeap:
    def __init__(self, arr = None):
        self.heap = arr if arr else []
        self.init()

    def push(self, val):
        self.heap.append(val)
        self._up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 1:
            return self.heap.pop()
        res = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._down(0)
        return res

    def init(self):
        for i in range(len(self.heap)//2 - 1, -1, -1):
            self._down(i)

    def _down(self, i):
        n = len(self.heap)
        while True:
            l, r = 2*i+1, 2*i+2
            small = i
            if l < n and self.heap[l] < self.heap[small]:
                small = l
            if r < n and self.heap[r] < self.heap[small]:
                small = r
            if small == i:
                break
            self.heap[i], self.heap[small] = self.heap[small], self.heap[i]
            i = small

        def _up(self, i):
            while i > 0:
                parent = (i-1)//2
                if self.heap[parent] >= self.heap[i]:
                    break
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent

if __name__ == '__main__':
    h = MaxHeap([4, 2, 3, 14, 5, 16, 1, 8, 0])

    for i in range(len(h.heap)):
        print(h.pop())
