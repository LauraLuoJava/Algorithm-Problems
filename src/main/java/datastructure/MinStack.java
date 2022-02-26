package datastructure;

import java.util.Stack;

public class MinStack {
        Stack<Node> minStack;
        int min = 0;
        class Node{
            int val; int min;
            Node(int val, int min){
                this.val = val;
                this.min = min;
            }
        }

        public MinStack() {
            minStack = new Stack<>();
        }

        public void push(int val) {
            if (minStack.isEmpty())
                min = val;
            else
                min = Math.min(val, minStack.peek().min);
            minStack.push(new Node(val, min));
        }

        public void pop() {
            //         if (!minStack.isEmpty())
            minStack.pop();
        }

        public int top() {
            //         if (!minStack.isEmpty())
            return minStack.peek().val;
        }

        public int getMin() {
            return minStack.peek().min;
        }
}
