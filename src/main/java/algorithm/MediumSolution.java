package algorithm;

import datastructure.ListNode;
import datastructure.Node;
import datastructure.TreeNode;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class MediumSolution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0 && (i == 0 || nums[i] != nums[i - 1])) {
                int l = i + 1;
                int r = nums.length - 1;
                while (r > l) {
                    if (nums[i] + nums[l] + nums[r] < 0)
                        l++;
                    else if (nums[i] + nums[l] + nums[r] > 0)
                        r--;
                    else {
                        result.add(Arrays.asList(nums[i], nums[l], nums[r]));
                        Boolean duplicate = true;
                        for (int j = l + 1; j < r; j++) {
                            if (nums[j] != nums[l]) {
                                l = j;
                                duplicate = false;
                                break;
                            }
                        }
                        if (duplicate) break;
                    }
                }
            }
        }
        return result;
    }

    public int reverse(int x) {
        int rev = 0, pop = 0;
        while (x != 0) {
            pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7)) return 0;
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

//https://leetcode.com/problems/house-robber-ii/
    public int rob2(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        return Math.max(rob2(nums,0,n-2),rob2(nums,1,n-1));
    }
    public int rob2(int[] nums, int start, int end) {
        int robMe = 0;
        int robNotMe = 0;
        int maxRob = 0;
        for (int i = start; i <= end; i++){
            int temp = robMe;
            robMe = robNotMe + nums[i];
            robNotMe = Math.max(temp,robNotMe);
            maxRob = Math.max(maxRob,Math.max(robMe,robNotMe));
        }
        return maxRob;
    }

    //https://leetcode.com/problems/add-two-numbers/
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        int value = 0;
        boolean isFirst = true;
        ListNode result = l1 == null ? l2 : l1;
        ListNode current = result;
        while (l1 != null || l2 != null) {
            value = carry;
            if (l1 != null) {
                value += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                value += l2.val;
                l2 = l2.next;
            }
            if (isFirst) {
                current.val = value % 10;
                isFirst = false;
            } else {
                current.next = new ListNode(value % 10, null);
                current = current.next;
            }

            carry = value / 10;

        }
        if (carry == 1)
            current.next = new ListNode(1, null);
        return result;
    }

    public int lengthOfLongestSubstring(String s) {
        int[] chars = new int[128];
        int left = 0, right = 0, maxLength = 0;
        while (right < s.length()) {
            char r = s.charAt(right);
            chars[r]++;

            while (chars[r] > 1) {
                char l = s.charAt(left);
                chars[l]--;
                left++;

            }
            maxLength = Math.max(maxLength, right - left + 1);

            right++;
        }
        return maxLength;
    }

    public String longestPalindrome(String s) {
        String longestPalindrome = String.valueOf(s.charAt(0));
        for (int i = 0; i < s.length() - 1; i++) {

            if (s.charAt(i) == s.charAt(i + 1)) {
                int d = i + 2;
                for (; d < s.length(); d++) {
                    if (s.charAt(i) != s.charAt(d)) {
                        break;
                    }
                }
                d -= i + 1;
                int j = 1;
                for (; j <= i && i + j + d < s.length(); j++) {
                    if (s.charAt(i + j + d) != s.charAt(i - j)) {
                        break;
                    }
                }
                if (j * 2 + d > longestPalindrome.length())
                    longestPalindrome = s.substring(i - j + 1, i + j + d);
                continue;
            }
            if (i < s.length() - 2 && s.charAt(i) == s.charAt(i + 2)) {
                int j = 1;
                for (; j <= i && i + j + 2 < s.length(); j++) {
                    if (s.charAt(i + j + 2) != s.charAt(i - j)) {
                        break;
                    }
                }
                if ((j * 2 + 1) > longestPalindrome.length())
                    longestPalindrome = s.substring(i - j + 1, i + j + 2);
            }
        }
        return longestPalindrome;
    }

    public int myAtoi(String s) {

        long ans = 0;

        int multiplier = 1;

        s = s.trim();

        if (s.length() == 0)
            return 0;

        int min = Integer.MIN_VALUE;
        int max = Integer.MAX_VALUE;

        if (s.charAt(0) == '-')
            multiplier = -1;

        int i = (s.charAt(0) == '-' || s.charAt(0) == '+') ? 1 : 0;

        while (i < s.length()) {
            if (s.charAt(i) == ' ' || !Character.isDigit(s.charAt(i)))
                break;
            ans = ans * 10 + (s.charAt(i) - '0');
            if (multiplier == -1 && -1 * ans < min)
                return min;
            if (multiplier == 1 && 1 * ans > max)
                return max;

            i++;
        }
        return (int) (multiplier * ans);
    }

    public int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            if (height[left] < height[right]) {
                maxArea = Math.max(maxArea, height[left] * (right - left));
                left++;
            } else {
                maxArea = Math.max(maxArea, height[right] * (right - left));
                right--;
            }
        }
        return maxArea;
    }

    public List<String> letterCombinations(String digits) {
        List<String> firstStringList = new ArrayList<>();
        if (digits.length() == 0) return firstStringList;
        firstStringList = letterCombination(Integer.parseInt(digits.substring(0, 1)));
        if (digits.length() == 1) return firstStringList;
        List<String> nextStringList = letterCombinations(digits.substring(1));
        List<String> stringList = new ArrayList<>();
        for (String s : firstStringList) {
            for (String n : nextStringList) {
                stringList.add(s + n);
            }
        }
        return stringList;
    }

    public List<String> letterCombination(int digit) {
        List<String> stringList = new ArrayList<>();
        if (digit < 7) {
            stringList.add(Character.toString('a' + ((digit - 2) * 3)));
            stringList.add(Character.toString('a' + ((digit - 2) * 3 + 1)));
            stringList.add(Character.toString('a' + ((digit - 2) * 3 + 2)));
        } else if (digit == 7) {
            stringList.add("p");
            stringList.add("q");
            stringList.add("r");
            stringList.add("s");
        } else if (digit == 8) {
            stringList.add("t");
            stringList.add("u");
            stringList.add("v");
        } else if (digit == 9) {
            stringList.add("w");
            stringList.add("x");
            stringList.add("y");
            stringList.add("z");
        }
        return stringList;

    }

    List<String> res = new ArrayList<>();

    public List<String> generateParenthesis(int n) {
        bt(new StringBuilder(), 0, 0, n);

        return res;
    }

    public void bt(StringBuilder curr, int open, int close, int max) {
        if (curr.length() == max * 2) {
            res.add(curr.toString());
            return;
        }

        if (open < max) {
            //StringBuilder newCurr = new StringBuilder(curr);
            curr.append("(");

            bt(curr, open + 1, close, max);

            curr.deleteCharAt(curr.length() - 1);//REMEMBER PLEASE !!
        }

        if (close < open) {
            curr.append(")");

            bt(curr, open, close + 1, max);

            curr.deleteCharAt(curr.length() - 1);//REMEMBER PLEASE !!
        }
    }

    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1)
            return Integer.MAX_VALUE;
        int absDividend = Math.abs(dividend);
        int absDivisor = Math.abs(divisor);
        int result = 0;
        while (absDividend - absDivisor >= 0) {
            int count = 0;
            while (absDividend - (absDivisor << 1 << count) >= 0) {
                count++;
            }
            result += (1 << count);
            absDividend -= (absDivisor << count);
        }
        return (dividend < 0) == (divisor >= 0) ? -result : result;
    }

    public int searchRotatedSorted(int[] nums, int target) {
        int length = nums.length;
        int leftEnd = length / 2;
        int rightStart = leftEnd;
        int left = length / 4;
        int right = length * 3 / 4;
        int n = 0;
        while (nums[left] < nums[leftEnd] && nums[rightStart] < nums[right]) {
            leftEnd = left;
            left = (left + 1) / 2;
            rightStart = right;
            right = Math.min(right + left, length - 1);
        }
        if (nums[left] > nums[leftEnd]) {
            n = findRotatedIndex(nums, left, leftEnd);
        }
        if (nums[rightStart] > nums[right]) {
            n = findRotatedIndex(nums, rightStart, right);
        }
        int result = Arrays.binarySearch(nums, 0, n + 1, target);
        if (result >= 0)
            return result;
        else {
            result = Arrays.binarySearch(nums, n + 1, length, target);
            return result >= 0 ? result : -1;
        }
    }

    public int findRotatedIndex(int[] nums, int start, int end) {
        int middle = (start + end) / 2;
        if (nums[start] == nums[middle]) return start;
        if (nums[start] > nums[middle]) {
            return findRotatedIndex(nums, start, middle);
        }
        if (nums[middle] > nums[end]) {
            return findRotatedIndex(nums, middle, end);
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        int start = binarySearch(nums, 0, nums.length - 1, target);
        if (start == -1) return new int[]{-1, -1};
        int end = start;
        while (start > 0 && nums[start - 1] == target) {
            start--;
        }
        while (end < nums.length - 1 && nums[end + 1] == target) {
            end++;
        }
        return new int[]{start, end};
    }

    public int binarySearch(int[] nums, int start, int end, int target) {
        if (start > end) return -1;
        int middle = (start + end) / 2;
        if (nums[middle] == target) return middle;
        if (nums[start] <= target && target < nums[middle]) {
            return binarySearch(nums, start, middle - 1, target);
        } else {
            return binarySearch(nums, middle + 1, end, target);
        }
    }

    public boolean isValidSudoku(char[][] board) {
        for (int row = 0; row < board.length; row++) {
            Set<Character> seriesRow = new HashSet<>();
            Set<Character> seriesColumn = new HashSet<>();
            for (int column = 0; column < board[row].length; column++) {

                if (board[row][column] != '.')
                    if (!seriesRow.add(board[row][column]))
                        return false;
                if (board[column][row] != '.')
                    if (!seriesColumn.add(board[column][row]))
                        return false;
            }
        }
        for (int r = 0; r < 3; r++) {
            for (int i = 0; i < 3; i++) {
                Set<Character> seriesSquare = new HashSet<>();
                for (int j = 0; j < 3; j++) {
                    for (int s = 0; s < 3; s++) {
                        if (board[3 * r + j][3 * i + s] != '.')
                            if (!seriesSquare.add(board[3 * r + j][3 * i + s]))
                                return false;
                    }
                }
            }
        }
        return true;
    }

    public String countAndSay(int n) {
        if (n == 1) return "1";
        StringBuilder result = new StringBuilder();
        int count = 1;
        String prev = countAndSay(n - 1);
        for (int i = 0; i < prev.length() - 1; i++) {
            if (prev.charAt(i) == prev.charAt(i + 1))
                count++;
            else {
                result.append(count).append(prev.charAt(i));
                count = 1;
            }
        }
        result.append(count).append(prev.charAt(prev.length() - 1));
        return result.toString();
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> permuteList = new ArrayList<>();
        List<Integer> permute = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        permuteSingle(permuteList, permute, nums, visited);
        return permuteList;
    }

    public void permuteSingle(List<List<Integer>> permuteList, List<Integer> permute, int[] nums, boolean[] visited) {
        if (permute.size() == nums.length) {
            permuteList.add(new ArrayList<>(permute));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!visited[i]) {
                permute.add(nums[i]);
                visited[i] = true;
                permuteSingle(permuteList, permute, nums, visited);
                permute.remove(permute.size() - 1);
                visited[i] = false;
            }
        }
    }

    public void rotate90Degree(int[][] matrix) {
        int n = matrix.length - 1;
        int temp1, temp2;
        for (int i = 0; i < (n + 1) / 2; i++) {
            for (int j = 0; j < n - i * 2; j++) {
                temp1 = matrix[i + j][n - i];
                matrix[i + j][n - i] = matrix[i][i + j];
                temp2 = matrix[n - i][n - i - j];
                matrix[n - i][n - i - j] = temp1;
                temp1 = matrix[n - i - j][i];
                matrix[n - i - j][i] = temp2;
                matrix[i][i + j] = temp1;
            }
        }
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        Function<String, String> sortString = new Function<String, String>() {
            @Override
            public String apply(String s) {
                char[] tempArray = s.toCharArray();
                Arrays.sort(tempArray);
                return new String(tempArray);
            }
        };
        return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(sortString)).values());
    }

    public double myPow(double x, int n) {
        double ans = 1;
        while (n != 0) {
            if (n % 2 != 0) {
                ans = (n > 0) ? ans * x : ans * (1 / x);
            }
            x = x * x;
            n /= 2;
        }
        return ans;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length - 1;
        int n = matrix[0].length - 1;
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i <= Math.min(m / 2, n / 2); i++) {
            for (int j = i; j <= n - i; j++)
                result.add(matrix[i][j]);
            for (int j = i + 1; j <= m - i; j++)
                result.add(matrix[j][n - i]);
            for (int j = n - i - 1; j >= i && m - i != i; j--)
                result.add(matrix[m - i][j]);
            for (int j = m - i - 1; j > i && n - i != i; j--)
                result.add(matrix[j][i]);
        }
        return result;
    }

    public boolean canJump(int[] nums) {
        if (nums.length == 1)
            return true;
        else
            return canJumpRecursion(nums, nums.length - 1);
    }

    public boolean canJumpRecursion(int[] nums, int lastIndex) {
        int zeroIndex = lastIndex - 1;
        for (; zeroIndex >= 0; zeroIndex--) {
            if (nums[zeroIndex] == 0) break;
        }
        if (zeroIndex < 0) return true;
        else {

            for (int i = zeroIndex - 1; i >= 0; i--) {
                if (nums[i] > zeroIndex - i)
                    return canJumpRecursion(nums, i);
            }
        }
        return false;
    }

    public int[][] merge(int[][] intervals) {
        List<int[]> mergeList = new ArrayList<>();
        mergeList.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            int start = intervals[i][0];
            int end = intervals[i][1];
            mergeToList(start, end, 0, mergeList);
        }
        int[][] resultArray = new int[mergeList.size()][2];
        return mergeList.toArray(resultArray);
    }

    public void mergeToList(int start, int end, int index, List<int[]> mergeList) {
        if (end < mergeList.get(index)[0])
            mergeList.add(index, new int[]{start, end});
        else if (mergeList.get(index)[0] <= end && end <= mergeList.get(index)[1])
            mergeList.get(index)[0] = Math.min(mergeList.get(index)[0], start);
        else if (start <= mergeList.get(index)[1]) {
            mergeList.get(index)[0] = Math.min(mergeList.get(index)[0], start);
            for (int j = index + 1; j <= mergeList.size(); ) {
                if (mergeList.size() == index + 1 || end < mergeList.get(j)[0]) {
                    mergeList.get(index)[1] = end;
                    break;
                }
                if (mergeList.get(j)[0] <= end && end <= mergeList.get(j)[1]) {
                    mergeList.get(index)[1] = mergeList.get(j)[1];
                    mergeList.remove(j);
                    break;
                }
                if (end > mergeList.get(j)[1]) {
                    mergeList.remove(j);
                    continue;
                }
            }
        } else if (index == mergeList.size() - 1)
            mergeList.add(new int[]{start, end});
        else
            mergeToList(start, end, index + 1, mergeList);
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> mergelist = new ArrayList<>();
        int start = newInterval[0];
        int end = newInterval[1];
        int endIndex = 0;
        for (; endIndex < intervals.length; endIndex++) {
            if (intervals[endIndex][1] < start)
                mergelist.add(intervals[endIndex]);
            else if (end < intervals[endIndex][0])
                break;

            else {
                start = Math.min(start, intervals[endIndex][0]);
                end = Math.max(end, intervals[endIndex][1]);
                if (end <= intervals[endIndex][1]) {
                    endIndex++;
                    break;
                }
            }
        }
        mergelist.add(new int[]{start, end});
        for (int i = endIndex; i < intervals.length; i++)
            mergelist.add(intervals[i]);
        return mergelist.toArray(new int[0][]);
    }

    public int uniquePaths(int m, int n) {
        if (n == 1 || m == 1) return 1;
        int[] permutation = new int[m];
        for (int i = 0; i < m; i++)
            permutation[i] = 1;
        n -= 2;
        while (n > 0) {
            for (int i = 1; i < m; i++)
                permutation[i] = permutation[i] + permutation[i - 1];
            n--;
        }
        return Arrays.stream(permutation).sum();
    }

    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int column = matrix[0].length;
        boolean firstColumn = false;
        boolean firstRow = false;
        for (int[] ints : matrix) {
            if (ints[0] == 0) {
                firstColumn = true;
                break;
            }
        }
        for (int i : matrix[0]) {
            if (i == 0) {
                firstRow = true;
                break;
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < column; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (matrix[0][0] == 0) {
            if (firstRow)
                Arrays.fill(matrix[0], 0);
            if (firstColumn) {
                for (int i = 1; i < row; i++) {
                    matrix[i][0] = 0;
                }
            }
        }
    }

    public boolean exist(char[][] board, String word) {
        for (int startRow = 0; startRow < board.length; startRow++) {
            for (int startColumn = 0; startColumn < board[startRow].length; startColumn++) {
                if (board[startRow][startColumn] == word.charAt(0)
                        && existRecursion(board, word, startRow, startColumn, 0))
                    return true;
            }
        }
        return false;
    }

    public boolean existRecursion(char[][] board, String word, int row, int column, int charIndex) {
        if (row < 0 || column < 0 || row == board.length || column == board[0].length) return false;
        if (board[row][column] == word.charAt(charIndex)) {
            board[row][column] += 27;
            if (charIndex == word.length() - 1)
                return true;
            if (existRecursion(board, word, row, column + 1, charIndex + 1)
                    || existRecursion(board, word, row + 1, column, charIndex + 1)
                    || existRecursion(board, word, row - 1, column, charIndex + 1)
                    || existRecursion(board, word, row, column - 1, charIndex + 1)) {
                return true;
            }
            board[row][column] -= 27;
        }
        return false;
    }

    public int numDecodings(String s) {
        if (s.charAt(0) == '0') return 0;
        int l = s.length();
        if (l == 1) return 1;
        int num1 = 1, num2 = s.charAt(1) == '0' ? 0 : 1;

        for (int i = 2; i < l; i++) {
            if (s.charAt(i) == '0') {
                num1 = num2;
                num2 = 0;
            } else {
                int temp = num2;
                num2 = (s.charAt(i - 2) - 48) * 10 + s.charAt(i - 1) - 48 < 27 ? num1 + num2 : num2;
                num1 = temp;
            }
        }
        return (s.charAt(l - 2) - 48) * 10 + s.charAt(l - 1) - 48 < 27 ? num1 + num2 : num2;
    }

    public boolean isValidBST(TreeNode root) {
        return isValidBSTRecursion(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    public boolean isValidBSTRecursion(TreeNode root, int start, int end) {
        if (root == null) return true;
        int value = root.val;
        if (value < start || value > end)
            return false;
        return (value == start ? root.left == null : isValidBSTRecursion(root.left, start, value - 1))
                && (value == end ? root.right == null : isValidBSTRecursion(root.right, value + 1, end));
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null ^ q == null) return false;
        if (p == null) return true;
        if (p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        return levelOrderRecursion(root, 0, new ArrayList<>());
    }

    public List<List<Integer>> levelOrderRecursion(TreeNode root, int level, List<List<Integer>> nodeList) {
        if (root != null) {
            if (nodeList.size() == level)
                nodeList.add(new ArrayList<>());
            nodeList.get(level).add(root.val);
            levelOrderRecursion(root.left, level + 1, nodeList);
            levelOrderRecursion(root.right, level + 1, nodeList);
        }
        return nodeList;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeRecursion(preorder, inorder, new int[]{0}, 0, inorder.length - 1);
    }

    public TreeNode buildTreeRecursion(int[] preorder, int[] inorder, int[] index, int start, int end) {
        TreeNode root = new TreeNode(preorder[index[0]]);
        if (start == end) return root;
        int rootIndex = -1;
        for (int i = start; i <= end; i++) {
            if (inorder[i] == preorder[index[0]]) {
                rootIndex = i;
                break;
            }
        }
        if (rootIndex > start) {
            index[0] += 1;
            root.left = buildTreeRecursion(preorder, inorder, index, start, rootIndex - 1);
        }
        if (rootIndex < end) {
            index[0] += 1;
            root.right = buildTreeRecursion(preorder, inorder, index, rootIndex + 1, end);
        }
        return root;
    }


    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        if (nums.length <= 0) return n;
        Arrays.sort(nums);
        int longest = 1, cLong = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] == nums[i - 1] + 1) {
                cLong++;
            } else if (nums[i] > nums[i - 1] + 1) {
                longest = Math.max(longest, cLong);
                cLong = 1;
            }
        }
        return Math.max(longest, cLong);
    }

    Map<Node, Node> cloneList = new HashMap<>();

    public Node cloneGraph(Node node) {
        if (node == null) return null;
        Node copyNode = new Node();
        copyNode.val = node.val;
        cloneList.put(node, copyNode);
        for (Node n : node.neighbors) {
            if (cloneList.get(n) != null)
                copyNode.neighbors.add(cloneList.get(n));
            else
                copyNode.neighbors.add(cloneGraph(n));
        }
        return copyNode;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        List<Integer> checked = new ArrayList<>();
        checked.add(0);
        return wordBreakRecursion(s, wordDict, 0, checked);
    }

    public boolean wordBreakRecursion(String s, List<String> wordDict, int checkedIndex, List<Integer> checked) {
        int start = checked.get(checkedIndex);
        for (String word : wordDict) {
            if (s.substring(start).startsWith(word)) {
                if (s.length() - word.length() == start)
                    return true;
                if (!checked.contains(start + word.length()))
                    checked.add(start + word.length());
            }
        }
        if (checkedIndex == checked.size() - 1)
            return false;
        return wordBreakRecursion(s, wordDict, checkedIndex + 1, checked);
    }

    public void reorderList(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode second = slow.next;
        slow.next = null;
        reorderListRecursion(head, second);
    }

    public ListNode reorderListRecursion(ListNode first, ListNode second) {
        if (second == null) return first;
        ListNode currentFirst = reorderListRecursion(first, second.next);
        ListNode aftercurrentFirst = currentFirst.next;
        currentFirst.next = second;
        second.next = aftercurrentFirst;
        return aftercurrentFirst;
    }

    public int maxProduct(int[] nums) {
        int maxOverall = nums[0];
        int maxHere = maxOverall;
        int minHere = maxOverall;
        for (int i = 1; i < nums.length; i++) {
            int temp = maxHere;
            maxHere = Math.max(Math.max(maxHere * nums[i], nums[i]), minHere * nums[i]);
            minHere = Math.min(Math.min(temp * nums[i], nums[i]), minHere * nums[i]);
            maxOverall = Math.max(maxHere, maxOverall);
        }
        return maxOverall;
    }

    public int findMin(int[] nums) {
        int len = nums.length;
        if (len == 1) return nums[0];
        int middle = len / 2;
        if (nums[middle - 1] > nums[middle]) return nums[middle];
        if (middle + 1 < len && nums[middle] > nums[middle + 1]) return nums[middle + 1];
        return findMinRecursion(nums, middle / 2, (middle + len) / 2);
    }

    public int findMinRecursion(int[] nums, int start, int end) {
        if (nums[end] < nums[start]) {
            int min = nums[end];
            for (int i = end - 1; i >= 0; i--) {
                if (nums[i] < min)
                    min = nums[i];
                else
                    break;
            }
            return min;
        } else if (end == nums.length - 1) {
            int min = nums[start];
            for (int i = start - 1; i >= 0; i--) {
                if (nums[i] < min)
                    min = nums[i];
                else
                    break;
            }
            return min;
        } else
            return findMinRecursion(nums, start / 2, (end + nums.length) / 2);
    }

    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        int robMe = nums[1];
        int robNotMe = nums[0];
        int maxRob = Math.max(robMe, robNotMe);
        for (int i = 2; i < n; i++) {
            int temp = robMe;
            robMe = robNotMe + nums[i];
            robNotMe = Math.max(temp, robNotMe);
            maxRob = Math.max(maxRob, Math.max(robMe, robNotMe));
        }
        return maxRob;
    }

    public int robCycle(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
    }

    public int rob(int[] nums, int start, int end) {
        int robMe = 0;
        int robNotMe = 0;
        int maxRob = 0;
        for (int i = start; i <= end; i++) {
            int temp = robMe;
            robMe = robNotMe + nums[i];
            robNotMe = Math.max(temp, robNotMe);
            maxRob = Math.max(maxRob, Math.max(robMe, robNotMe));
        }
        return maxRob;
    }

    public int numIslands(char[][] grid) {
        int num = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1' && isIsland(grid, i, j, (char) (num + 65))) {
                    num++;
                }
            }
        }
        return num;
    }

    public boolean isIsland(char[][] grid, int row, int column, char no) {
        if (row < 0 || row >= grid.length || column < 0 || column >= grid[0].length
                || grid[row][column] == '0' || grid[row][column] == no) return true;
        if (grid[row][column] != '1') return false;
        grid[row][column] = no;
        return isIsland(grid, row - 1, column, no)
                && isIsland(grid, row + 1, column, no)
                && isIsland(grid, row, column - 1, no)
                && isIsland(grid, row, column + 1, no);
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        boolean[] checked = new boolean[numCourses];
        boolean[] recycle = new boolean[numCourses];
        List<List<Integer>> prerequisiteList = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            prerequisiteList.add(new ArrayList<>());
        for (int[] prerequisite : prerequisites) {
            int course = prerequisite[0];
            int pre = prerequisite[1];
            prerequisiteList.get(course).add(pre);

        }
        for (int course = 0; course < numCourses; course++) {
            if (!checked[course] && !canFinishRecursion(recycle, prerequisiteList, course, checked))
                return false;
        }
        return true;
    }

    public boolean canFinishRecursion(boolean[] recycle, List<List<Integer>> prerequisiteList, int course, boolean[] checked) {
        if (recycle[course])
            return false;

        if (checked[course] || prerequisiteList.get(course).size() == 0)
            return true;
        recycle[course] = true;
        checked[course] = true;
        for (int i : prerequisiteList.get(course)) {
            if (!canFinishRecursion(recycle, prerequisiteList, i, checked))
                return false;
        }
        recycle[course] = false;
        return true;
    }

    public int dominator(int[] A) {
        // write your code in Java SE 8
        if (A == null || A.length == 0) return -1;
        int n = A.length, count;
        Map<Integer, Integer> group = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int a = A[i];
            if (group.containsKey(a)) {
                count = group.get(a);
                if (count >= n / 2) return i;
                group.replace(a, count + 1);
            } else {
                group.put(a, 1);
            }

        }
        return -1;
    }

    public int codilityTask3(int[] A) {
        int start = A[0], count = 1;
        boolean isStart = true, isHill = false;
        for (
                int i = 1;
                i < A.length; i++) {
            if (isStart) {
                if (A[i] != start) {
                    count++;
                    isHill = A[i] > start;
                    isStart = false;
                } else continue;
            } else if (isHill && A[i] < start){
                count++;
            }
            else if (!isHill && A[i] > start) {
                count++;
            }
            start = A[i];
        }
        return count;
    }

    public int codilityTask2(int M, int[] A) {
        int N = A.length;
        int[] count = new int[M + 1];
        for (int i = 0; i <= M; i++)
            count[i] = 0;
        int maxOccurence = 1;
        int index = -1;
        for (int i = 0; i < N; i++) {
            if (count[A[i]] > 0) {
                int tmp = count[A[i]];
                if (tmp > maxOccurence) {
                    maxOccurence = tmp;
                    index = i;
                }
                count[A[i]] = tmp + 1;
            } else {
                count[A[i]] = 1;
            }
        }
        return A[index];
    }

}
