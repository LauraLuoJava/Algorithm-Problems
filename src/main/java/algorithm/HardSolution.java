package algorithm;

import datastructure.ListNode;
import datastructure.TreeNode;

public class HardSolution {
//https://leetcode.com/problems/binary-tree-maximum-path-sum/
    static int maxSum;
    public int maxPathSum(TreeNode root) {
        maxSum=Integer.MIN_VALUE;
        maxPathSumRecursion(root);
        return maxSum;
    }
    public static int maxPathSumRecursion(TreeNode root){
        int left,right;
        int value = root.val;
        left = root.left == null? 0 : maxPathSumRecursion(root.left);
        right = root.right == null? 0 : maxPathSumRecursion(root.right);
        maxSum = Math.max((Math.max(left, 0))+value+(Math.max(right, 0)),maxSum);
        return Math.max(Math.max(left,right),0) + value;
    }

    //https://leetcode.com/problems/minimum-window-substring
    public String minWindow(String s, String t) {
        int slen = s.length();
        int tlen = t.length();

        if (tlen > slen) {
            return "";
        }

        if (tlen == slen && s.equals(t)) {
            return s;
        }

        int minLeft = -1;
        int left = 0;
        int minLen = Integer.MAX_VALUE;
        int match = t.length();
        int[] count = new int[128];

        char[] array = s.toCharArray();

        for (int i = 0; i < t.length(); i++) {
            count[t.charAt(i)]++;
        }

        for (int i = 0; i < array.length; i++) {
            char c = array[i];
            count[c]--;

            if (count[c] >= 0) {
                match--;
            }

            while (match == 0) {
                if (i - left + 1 < minLen) {
                    minLeft = left;
                    minLen = i - left + 1;
                }

                char l = array[left];
                count[l]++;
                left++;

                if (count[l] > 0) {
                    match++;
                }
            }

        }

        return minLeft == -1 ? "" : s.substring(minLeft, minLeft + minLen);
    }

    //https://leetcode.com/problems/merge-k-sorted-lists/
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        int interval = 1;
        while (interval < lists.length) {
            for (int i = 0; i < lists.length - interval; i += interval * 2) {
                lists[i] = mergeTwoLists(lists[i], lists[i + interval]);
            }
            interval *= 2;
        }
        return lists[0];
    }
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) return null;
        else if (l1 == null) return l2;
        else if (l2 == null) return l1;
        ListNode lr = new ListNode();
        ListNode lc = new ListNode();
        boolean isStart = true;
        while (true) {
            if (l1.val <= l2.val) {
                if (isStart) {
                    isStart = false;
                    lr = l1;
                } else
                    lc.next = l1;
                lc = l1;
                if (l1.next != null)
                    l1 = l1.next;
                else {
                    lc.next = l2;
                    break;
                }
            } else {
                if (isStart) {
                    isStart = false;
                    lr = l2;
                } else
                    lc.next = l2;
                lc = l2;
                if (l2.next != null)
                    l2 = l2.next;
                else {
                    lc.next = l1;
                    break;
                }
            }
        }
        return lr;
    }

    //https://leetcode.com/problems/median-of-two-sorted-arrays/
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int middleIndex = (m + n) >> 1;
        boolean isOdd = ((m + n) & 1) == 1;
        int i = 0, j = 0, tempNum = 0;
        while (i < m || j < n) {
            while (i < m && (j >= n || nums1[i] <= nums2[j])) {
                if (i + j == middleIndex)
                    return isOdd ? nums1[i] : (nums1[i] + tempNum) / 2f;
                tempNum = nums1[i++];
            }
            while (j < n && (i >= m || nums2[j] <= nums1[i])) {
                if (i + j == middleIndex)
                    return isOdd ? nums2[j] : (nums2[j] + tempNum) / 2f;
                tempNum = nums2[j++];
            }
        }
        return 0;
    }
}
