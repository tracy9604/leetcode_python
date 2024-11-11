from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListProblems:
    # https://leetcode.com/problems/add-two-numbers
    def add_two_numbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        sum_head = ListNode()
        carry, current_node = 0, sum_head

        while l1 or l2 or carry:
            sum_node = (l1.val if l1 else 0) + (l2.val if l2 else 0) + carry
            carry, value = divmod(sum_node, 10)

            current_node.next = ListNode(value)
            current_node = current_node.next

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return sum_head.next

    def call_method(self):
        l1 = ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9)))))))
        l2 = ListNode(9, ListNode(9, ListNode(9, ListNode(9))))
        print(self.add_two_numbers(l1, l2))
        return