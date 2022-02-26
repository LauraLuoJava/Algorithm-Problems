package concurrency;

import java.util.concurrent.Semaphore;

//https://leetcode.com/problems/print-foobar-alternately/
class FooBar {
    private int n;

    public FooBar(int n) {
        this.n = n;
    }

    Semaphore semaphoreFoo = new Semaphore(0);
    Semaphore semaphoreBar = new Semaphore(0);

    public void foo(Runnable printFoo) throws InterruptedException {
        printFoo.run();
        semaphoreFoo.release();
        for (int i = 1; i < n; i++) {
            semaphoreBar.acquire();
            // printFoo.run() outputs "foo". Do not change or remove this line.
            printFoo.run();
            semaphoreFoo.release();
        }
    }

    public void bar(Runnable printBar) throws InterruptedException {

        for (int i = 0; i < n; i++) {
            semaphoreFoo.acquire();
            // printBar.run() outputs "bar". Do not change or remove this line.
            printBar.run();
            semaphoreBar.release();
        }
    }
}