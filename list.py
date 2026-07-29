arr = [10, 20, 30, 40, 50, 70]
i = 1
merged = [21, 22]

arr[i:i+3] = merged
print(arr)


# list index out of range
# l = []
# print(l[0])
a = [1, 2, 3] 
b= a[1:] 
print(a, b)
b[0]=100
print(a, b)
# 浅拷贝
a = [[1], [2], [3]]
b = a[1:]
print(a, b)
b[0].append(999)
print(a, b)