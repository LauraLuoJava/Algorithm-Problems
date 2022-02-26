package concurrency;

//https://leetcode.com/problems/print-zero-even-odd/

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

class ZeroEvenOdd {
    private int n;

    public ZeroEvenOdd(int n) {
        this.n = n;
    }

    AtomicInteger printN = new AtomicInteger(0);
    boolean zfree = false;
    boolean nfree = true;

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void zero(IntConsumer printNumber) throws InterruptedException {
        while (printN.get() < n) {
            synchronized(this){
                if (nfree) {
                    printNumber.accept(0);
                    zfree = true;
                    nfree = false;
                    notifyAll();
                }
                else
                    wait();
            }
        }
    }

    public void even(IntConsumer printNumber) throws InterruptedException {
        while (printN.get() < n) {
            synchronized(this) {
                if (zfree && (printN.get() & 1) == 1) {
                    printNumber.accept(printN.incrementAndGet());
                    zfree = false;
                    nfree = true;
                    notifyAll();
                }
                else
                    wait();
            }
        }
    }

    public void odd(IntConsumer printNumber) throws InterruptedException {
        while (printN.get() < n) {
            synchronized(this) {
                if (zfree && (printN.get() & 1) == 0) {
                    printNumber.accept(printN.incrementAndGet());
                    zfree = false;
                    nfree = true;
                    notifyAll();
                }
                else
                    wait();
            }
        }
    }
}
