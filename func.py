class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def move_pointer(p):
    p = p.next  # 只改变函数内部的 p，不影响外部

def modify_content(p):
    p.val = 100  # 修改节点内容（影响外部）
    temp = ListNode(666)
    if p.next:
        p.next = temp  # 修改下一个节点（影响外部）

head = ListNode(1, ListNode(2, ListNode(3)))
print("原始头节点值:", head.val)  # 输出 1
move_pointer(head)
print("调用后头节点值:", head.val)  # 输出 1（未改变）
modify_content(head)
print("调用后节点值:", head.val, head.next.val)  # 输出 100（改变）

def change_arr(nums):
    nums[0] = 999

l = [1,2,3]
change_arr(l)
print(l)
