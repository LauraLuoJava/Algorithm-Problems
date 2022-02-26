package algorithm;

import datastructure.ListNode;
import datastructure.Node;
import datastructure.TreeNode;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class EasySolution {
    //https://leetcode.com/problems/two-sum
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[]{-1, -1};
        Map<Integer, Integer> numsMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int balance = target - nums[i];
            if (numsMap.containsKey(balance)) {
                result[0] = numsMap.get(balance);
                result[1] = i;
            } else
                numsMap.put(nums[i], i);
        }
        return result;
    }

    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        int oint = x;
        int pop = 0;
        int rev = 0;
        while (x != 0) {
            pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 8)) return false;
            rev = rev * 10 + pop;
        }
        return oint == rev;
    }

    //https://leetcode.com/problems/roman-to-integer/
    public int romanToInt(String s) {
        int integer = 0, vor = 0, aft = 0;
        Function<Character, Integer> charToInt = c -> {
            switch (c) {
                case 'I':
                    return 1;
                case 'V':
                    return 5;
                case 'X':
                    return 10;
                case 'L':
                    return 50;
                case 'C':
                    return 100;
                case 'D':
                    return 500;
                case 'M':
                    return 1000;
                default:
                    System.out.println("it's not a roman number!");
                    return null;
            }
        };

        switch (s.length()) {
            case 0:
                return -1;
            case 1:
                return charToInt.apply(s.charAt(0));
            default:
                vor = charToInt.apply(s.charAt(0));
                for (int i = 1; i < s.length(); i++) {
                    aft = charToInt.apply(s.charAt(i));
                    if (aft > vor) {
                        switch (aft - vor) {
                            case 4:
                            case 9:
                            case 40:
                            case 90:
                            case 400:
                            case 900:
                                if (integer != 0) integer -= vor;
                                integer += aft - vor;
                                break;
                            default:
                                System.out.println("it's not a roman number!");
                        }

                    } else {
                        if (integer == 0) integer = vor;
                        integer += aft;
                    }
                    vor = aft;

                }
                return integer;
        }

    }

    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (strs[j].length() == i || strs[j].charAt(i) != c)
                    return strs[0].substring(0, i);
            }
        }
        return strs[0];
    }

    public boolean validParentheses(String s) {
        if (s.length() % 2 != 0) return false;
        char[] arr = s.toCharArray();
        Stack<Character> sta = new Stack<>();
        for (char c : arr) {
            if (c == '(' || c == '{' || c == '[')
                sta.push(c);
            else if (c == ')') {
                if (!sta.empty() && sta.peek() == '(') {
                    sta.pop();
                } else
                    return false;
            } else if (c == ']') {
                if (!sta.empty() && sta.peek() == '[') {
                    sta.pop();
                } else
                    return false;
            } else if (c == '}') {
                if (!sta.empty() && sta.peek() == '{') {
                    sta.pop();
                } else
                    return false;
            } else
                return false;
        }
        return sta.empty();
    }

    public int removeDuplicates(int[] nums) {
        if (nums == null) return 0;
        int k = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[k]) {
                if (i > k + 1)
                    nums[k + 1] = nums[i];
                k++;
            }
        }
        return k + 1;
    }

    public int removeElement(int[] nums, int val) {
        if (nums.length == 0) return 0;
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val) {
                if (i != k) {
                    nums[k] = nums[i];
                }
                k++;
            }
        }
        return k;
    }

    public int strStr(String haystack, String needle) {
        if (needle.isEmpty()) {
            return 0;
        }
        int[] borderArray = GetBorderArray(needle);
        int i = 0;
        int j = 0;
        int m = needle.length();
        int n = haystack.length();
        while (i <= n - m) {
            while (j < m && haystack.charAt(i + j) == needle.charAt(j)) {
                ++j;
            }
            if (j == m) {
                return i;
            }
            i = i + (j - borderArray[j]);
            j = Integer.max(borderArray[j], 0);
        }
        return -1;
    }

    public int[] GetBorderArray(String word) {
        int length = word.length();
        int[] border = new int[length + 1];
        border[0] = -1;
        for (int i = 1; i <= length; i++) {
            int t = border[i - 1];
            while (t >= 0 && word.charAt(t) != word.charAt(i - 1)) {
                t = border[t];
            }
            ++t;
            border[i] = t;
        }
        return border;
    }

    public int searchInsert(int[] nums, int target) {
        if (nums.length == 1) {
            if (target <= nums[0])
                return 0;
            else
                return 1;
        } else
            return searchInsertRecursion(nums, target, 0, nums.length - 1);
    }

    public int searchInsertRecursion(int[] nums, int target, int left, int right) {
        if (right == left) return left;
        else {
            if (target <= nums[left])
                return left;
            else if (target == nums[right])
                return right;
            else if (target > nums[right])
                return right + 1;
            else {
                int middle = (left + right) / 2;
                if (target == nums[middle])
                    return middle;
                else if (target < nums[middle])
                    return searchInsertRecursion(nums, target, left, middle);
                else
                    return searchInsertRecursion(nums, target, middle + 1, right);
            }
        }
    }

    public int maxSubArray(int[] nums) {
        int max = nums[0];
        int sum = max;
        for (int i = 1; i < nums.length; i++) {
            sum = Math.max(sum + nums[i], nums[i]);
            max = Math.max(sum, max);
        }
        return max;
    }

    public int lengthOfLastWord(String s) {
        return s.trim().substring(s.trim().lastIndexOf(" ") + 1).length();
    }

    public int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; ) {
            if (digits[i] + 1 == 10) {
                digits[i] = 0;
                if (i == 0) {
                    int[] result = new int[digits.length + 1];
                    result[0] = 1;
                    System.arraycopy(digits, 0, result, 1, digits.length);
                    return result;
                } else i--;
            } else {
                digits[i]++;
                return digits;
            }
        }
        return digits;
    }

    public String addBinary(String a, String b) {
    /*  another way: change between string and int
            BigInteger abig = new BigInteger(a, 2);
            BigInteger newbig = abig.add(new BigInteger(b, 2));
            return newbig.toString(2);
    */
        StringBuilder result = new StringBuilder();
        int alen = a.length() - 1, blen = b.length() - 1, bit = 0, carry = 0;
        while (alen >= 0 || blen >= 0) {
            bit = carry;
            if (alen >= 0) bit += a.charAt(alen--) - '0';
            if (blen >= 0) bit += b.charAt(blen--) - '0';
            result.append(bit & 1);
            carry = bit >> 1;
        }
        if (bit >= 2) result.append(carry);
        return result.reverse().toString();
    }

    public int mySqrt(int x) {
        /* Math Function
        return (int)Math.sqrt(x);
         */
        long start = 0;
        long end = x;
        long res = -1;
        while (start <= end) {
            long mid = start + (end - start) / 2;
            long msq = mid * mid;
            if (msq > x) {
                end = mid - 1;
            } else {
                start = mid + 1;
                res = mid;
            }
        }
        return (int) res;
    }

    public int climbStairs(int n) {
        int result = 1;
        int ex = 1;
        int exex = 1;
        for (int i = 2; i <= n; i++) {
            result = ex + exex;
            exex = ex;
            ex = result;
        }
        return result;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode result = head;
        ListNode undupNode = head;
        while (true) {
            if (undupNode.val != head.val) {
                head.next = undupNode;
                head = undupNode;
            }
            if (undupNode.next == null) break;
            else
                undupNode = undupNode.next;
        }
        head.next = null;
        return result;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int[] temp = new int[m];
        System.arraycopy(nums1, 0, temp, 0, m);
        int index1 = 0, index2 = 0, indexr = 0;
        while (index1 < m && index2 < n) {
            if (temp[index1] <= nums2[index2]) {
                nums1[indexr] = temp[index1];
                index1++;
            } else {
                nums1[indexr] = nums2[index2];
                index2++;
            }
            indexr++;
        }
        if (index1 < m)
            System.arraycopy(temp, index1, nums1, indexr, m - index1);
        else if (index2 < n)
            System.arraycopy(nums2, index2, nums1, indexr, n - index2);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        result.add(root.val);
        if (root.left != null)
            result.addAll(0, inorderTraversal(root.left));
        if (root.right != null)
            result.addAll(inorderTraversal(root.right));
        return result;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root.left == null && root.right == null)
            return true;
        if (root.left == null || root.right == null)
            return false;
        else
            return isSymmetricChild(root.left, root.right);
    }

    public boolean isSymmetricChild(TreeNode leftChild, TreeNode rightChild) {
        if (leftChild.val != rightChild.val)
            return false;
        if ((leftChild.left == null && rightChild.right != null) || (leftChild.left != null && rightChild.right == null))
            return false;
        if ((leftChild.right == null && rightChild.left != null) || (leftChild.right != null && rightChild.left == null))
            return false;
        if (leftChild.left == null) {
            if (leftChild.right == null)
                return true;
            else
                return isSymmetricChild(leftChild.right, rightChild.left);
        } else {
            if (isSymmetricChild(leftChild.left, rightChild.right)) {
                if (leftChild.right == null)
                    return true;
                else
                    return isSymmetricChild(leftChild.right, rightChild.left);
            } else
                return false;
        }
    }

    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTRecursion(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTRecursion(int[] nums, int start, int end) {
        TreeNode root = new TreeNode();
        if (start == end) {
            root.val = nums[start];
            return root;
        }
        int middle = (start + end) / 2;
        root.val = nums[middle];
        if (middle > start)
            root.left = sortedArrayToBSTRecursion(nums, start, middle - 1);
        if (middle < end)
            root.right = sortedArrayToBSTRecursion(nums, middle + 1, end);
        return root;
    }

    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] - minPrice > maxProfit)
                maxProfit = prices[i] - minPrice;
            else if (prices[i] < minPrice)
                minPrice = prices[i];
        }
        return maxProfit;
    }

    public int singleNumber(int[] nums) {
        int num = nums[0];
        for (int i = 1; i < nums.length; i++) {
            num ^= nums[i];
        }
        return num;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode single = head;
        ListNode couple = head.next;
        while (couple != null && couple.next != null) {
            if (single == couple)
                return true;
            else {
                single = single.next;
                couple = couple.next.next;
            }
        }
        return false;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode tempA = headA, tempB = headB;
        while (tempA != tempB) {
            tempA = (tempA == null) ? headB : tempA.next;
            tempB = (tempB == null) ? headA : tempB.next;
        }
        return tempA;
    }

    public int majorityElement(int[] nums) {
        int count = 0, element = 0;
        for (int i = 0; i < nums.length; i++) {
            if (count == 0) {
                element = nums[i];
                count++;
            } else if (nums[i] != element)
                count--;
            else
                count++;
        }
        return element;
    }

    public ListNode reverseList(ListNode head) {
        ListNode parent = null;
        ListNode child = head;
        while (child != null) {
            ListNode tempNode = child.next;
            child.next = parent;
            parent = child;
            child = tempNode;
        }
        return parent;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;

        TreeNode saveRoot = root.left;
        root.left = root.right;
        root.right = saveRoot;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    public boolean isPalindrome(ListNode head) {
        List<Integer> vals = new ArrayList<>();
        while (head != null) {
            vals.add(head.val);
            head = head.next;
        }
        int i = vals.size();
        for (int j = 0; j < i / 2; j++) {
            if (!vals.get(j).equals(vals.get(i - j - 1)))
                return false;
        }
        return true;
    }

    public void moveZeroes(int[] nums) {
        int lastNonZeroFoundAt = 0;
        // If the current element is not 0, then we need to
        // append it just in front of last non 0 element we found.
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[lastNonZeroFoundAt++] = nums[i];
            }
        }
        // After we have finished processing new elements,
        // all the non-zero elements are already at beginning of array.
        // We just need to fill remaining array with 0's.
        for (int i = lastNonZeroFoundAt; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        ans[0] = 0;
        for (int i = 1; i <= n; i++) {
            if (i % 2 == 0)
                ans[i] = ans[i / 2];
            else
                ans[i] = ans[i / 2] + 1;
        }
        return ans;
    }

    int diameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        maxDepthDiameter(root);
        return diameter;
    }

    public int maxDepthDiameter(TreeNode root) {
        if (root == null) return 0;
        int maxLeft = maxDepth(root.left);
        int maxRight = maxDepth(root.right);
        if (maxLeft + maxRight > diameter)
            diameter = maxLeft + maxRight;
        return 1 + Math.max(maxLeft, maxRight);
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        if (root1 == null) return root2;
        if (root2 == null) return root1;
        return new TreeNode(root1.val + root2.val, mergeTrees(root1.left, root2.left), mergeTrees(root1.right, root2.right));
    }

    List<List<Integer>> pascalTri = new ArrayList<>();

    public List<List<Integer>> generate(int numRows) {
        if (numRows == 0) return pascalTri;
        List<Integer> pascalRow = new ArrayList<>();
        pascalRow.add(1);
        pascalTri.add(pascalRow);
        if (numRows == 1) return pascalTri;
        pascalNumSet(numRows, 2, pascalRow);
        return pascalTri;
    }

    public void pascalNumSet(int numRows, int currentNum, List<Integer> prevRow) {
        if (currentNum <= numRows) {
            List<Integer> currentRow = new ArrayList<>();
            for (int j = 0; j < currentNum; j++) {
                if (j == 0) {
                    currentRow.add(1);
                } else if (j == currentNum - 1) {
                    currentRow.add(1);
                    pascalTri.add(currentRow);
                } else
                    currentRow.add(prevRow.get(j - 1) + prevRow.get(j));
            }
            pascalNumSet(numRows, currentNum + 1, currentRow);
        }
    }

    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (!Character.isLetterOrDigit(s.charAt(left))) {
                left++;
                continue;
            }
            if (!Character.isLetterOrDigit(s.charAt(right))) {
                right--;
                continue;
            }
            if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right)))
                return false;
            left++;
            right--;
        }
        return true;
    }

    public int titleToNumber(String columnTitle) {
        int n = columnTitle.length() - 1;
        int result = columnTitle.charAt(n) - 'A' + 1;
        for (int i = 1; i <= n; i++) {
            result += (columnTitle.charAt(n - i) - 'A' + 1) * Math.pow(26, i);
        }
        return result;
    }

    public int reverseBits(int n) {
        int result = 0; //Let the final answer be called "result" and we initialize it to zero.
        for (int i = 0; i < 32; i++) { //To be able to access each and every digit of n.
            int lsb = n & 1;            //To get the last digit of the given number
            int reverseLSB = lsb << (31 - i); //Making a bitmask.
            result = result | reverseLSB;   //OR the result with reverseLSB to form the req. no.
            n = n >> 1;
        }
        return result;
    }

    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            if ((n & 1) == 1)
                count++;
            n >>>= 1;
        }
        return count;
    }

    public boolean isHappy(int n) {
        int slowRunner = n;
        int fastRunner = getNext(n);
        while (fastRunner != 1 && slowRunner != fastRunner) {
            slowRunner = getNext(slowRunner);
            fastRunner = getNext(getNext(fastRunner));
        }
        return fastRunner == 1;
    }

    public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }

    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1])
                return true;
        }
        return false;
    }

    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public boolean isAnagram(String s, String t) {
        char[] sArray = s.toCharArray();
        char[] tArray = t.toCharArray();
        Arrays.sort(sArray);
        Arrays.sort(tArray);
        return Arrays.equals(sArray, tArray);
    }

    public int missingNumber(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums)
            sum += num;
        return n * (n + 1) / 2 - sum;
    }

    public boolean isPowerOfThree(int n) {
        if (n == 0) return false;
        while (n % 27 == 0) n = n / 27;
        while (n % 9 == 0) n = n / 9;
        while (n % 3 == 0) n = n / 3;
        return n == 1;
    }

    public void reverseString(char[] s) {
        char temp;
        int n = s.length - 1;
        int i = 0;
        while (i < n) {
            temp = s[i];
            s[i] = s[n];
            s[n] = temp;
            i++;
            n--;
        }
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        int[] arr1 = new int[1001];
        int[] ans = new int[1001];

        for (int i = 0; i < nums1.length; i++) {
            arr1[nums1[i]]++;
        }
        int k = 0;
        for (int i = 0; i < nums2.length; i++) {
            if (arr1[nums2[i]] > 0) {
                ans[k++] = nums2[i];
                arr1[nums2[i]]--;
            }
        }
        return Arrays.copyOfRange(ans, 0, k);
    }

    public int firstUniqChar(String s) {
        int[] freq = new int[26];

        for (char c : s.toCharArray())
            freq[c - 'a']++;

        for (int i = 0; i < s.length(); i++) {
            if (freq[s.charAt(i) - 'a'] == 1)
                return i;
        }

        return -1;
    }

    public List<String> fizzBuzz(int n) {
        /* IntStream
        return IntStream.range(1,n+1).mapToObj((IntFunction<String>)
                i->i%15 == 0?"FizzBuzz":(i%5== 0? "Buzz":
                        (i%3== 0? "Fizz":String.valueOf(i)))).collect(Collectors.toList());

         */
        List<String> answer = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            answer.add(i % 15 == 0 ? "FizzBuzz" : (i % 5 == 0 ? "Buzz" : (i % 3 == 0 ? "Fizz" : String.valueOf(i))));
        }
        return answer;
    }
}