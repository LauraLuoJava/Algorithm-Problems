package concurrency;

import java.util.concurrent.*;

//https://leetcode.com/problems/building-h2o/
class H2O {

    private final CyclicBarrier barrier = new CyclicBarrier(3);
    private final Semaphore hSem = new Semaphore(2);
    private final Semaphore oSem = new Semaphore(1);

    public H2O() {

    }

    public void hydrogen(Runnable releaseHydrogen) throws InterruptedException {
        try {
            hSem.acquire();

            // releaseHydrogen.run() outputs "H". Do not change or remove this line.
            releaseHydrogen.run();
            barrier.await();
        } catch(Exception ignore) {

        } finally {
            hSem.release();
        }

    }

    public void oxygen(Runnable releaseOxygen) throws InterruptedException {
        try {
            oSem.acquire();

            // releaseOxygen.run() outputs "O". Do not change or remove this line.
            releaseOxygen.run();
            barrier.await();
        } catch(Exception ignore) {

        } finally {
            oSem.release();
        }
    }
}
