def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = split(arr)
    l = quick_sort(arr[:pivot])
    r = quick_sort(arr[pivot+1:])
    return l + [arr[pivot]] + r

def split(arr):
    pivot = arr[0]
    low = 0
    high = len(arr)-1
    while low < high:
        while arr[high] >= pivot and low < high:
            high -= 1
        arr[low] = arr[high]
        while arr[low] <= pivot and low < high:
            low += 1
        arr[high] = arr[low]

    arr[low] = pivot
    return low


if __name__ == '__main__':
    l = [5,14,13,22,1,999,-2]
    x = quick_sort(l)
    print(l)
    print(x)
    # split(l)
    # print(l)