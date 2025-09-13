import heapq
arr = [5, 1, 8, 3]
heapq.heapify(arr) #最小堆
print(arr)
heapq.heappush(arr, 2)
print(arr)
heapq.heappush(arr, -2)
print(arr)
smallest = heapq.heappop(arr)
print(smallest)
print(arr)
val = heapq.heappushpop(arr, 0) #先插入后弹出
print(val)
print(arr)
res = heapq.nlargest(3, arr)
print(res)