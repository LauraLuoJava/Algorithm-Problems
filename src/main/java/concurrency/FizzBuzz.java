package concurrency;

import java.util.concurrent.Semaphore;
import java.util.function.IntConsumer;

//https://leetcode.com/problems/fizz-buzz-multithreaded/
class FizzBuzz {
    private int n;

    public FizzBuzz(int n) {
        this.n = n;
    }

    Semaphore nPrint = new Semaphore(1);
    Semaphore fPrint = new Semaphore(0);
    Semaphore bPrint = new Semaphore(0);
    Semaphore fbPrint = new Semaphore(0);

    boolean done = false;

    // printFizz.run() outputs "fizz".
    public void fizz(Runnable printFizz) throws InterruptedException {

        while (true) {
            fPrint.acquire();
            if (done) break;
            printFizz.run();
            nPrint.release();
        }

    }

    // printBuzz.run() outputs "buzz".
    public void buzz(Runnable printBuzz) throws InterruptedException {
        while (true) {
            bPrint.acquire();
            if (done) break;
            printBuzz.run();
            nPrint.release();
        }

    }

    // printFizzBuzz.run() outputs "fizzbuzz".
    public void fizzbuzz(Runnable printFizzBuzz) throws InterruptedException {
        while (true) {
            fbPrint.acquire();
            if (done) break;
            printFizzBuzz.run();
            nPrint.release();
        }
    }

    // printNumber.accept(x) outputs "x", where x is an integer.
    public void number(IntConsumer printNumber) throws InterruptedException {
        for (int i = 1; i <= n; i++) {
            nPrint.acquire();
            if (i % 15 == 0) {
                fbPrint.release();
            } else if (i % 5 == 0) {
                bPrint.release();
            } else if (i % 3 == 0) {
                fPrint.release();
            } else {
                printNumber.accept(i);
                nPrint.release();
            }
        }
        nPrint.acquire();
        done = true;
        fbPrint.release();
        bPrint.release();
        fPrint.release();
    }
}
