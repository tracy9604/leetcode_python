from typing import Optional

from sorting.common import ListNode


class InsertionSort:
    def insertion_sort_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        dummy_node = ListNode(0, head)
        prev, curr = head, head.next

        while curr:
            if prev.val <= curr.val:
                prev, curr = curr, curr.next
            else:
                insert_pos = dummy_node
                while insert_pos.next.val <= curr.val:
                    insert_pos = insert_pos.next

                tmp = curr.next
                curr.next = insert_pos.next
                insert_pos.next = curr
                prev.next = tmp

                curr= tmp

        return dummy_node.next

    def test_insertion_sort_list(self) -> None:
        node3 = ListNode(3)
        node1 = ListNode(1, node3 )
        node2 = ListNode(2, node1)
        node4 = ListNode(4, node2)

        rs = self.insertion_sort_list(node4)
        print(rs)