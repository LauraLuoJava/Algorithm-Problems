package concurrency;

import java.util.concurrent.Semaphore;

//https://leetcode.com/problems/the-dining-philosophers/
class DiningPhilosophers {

    private final Semaphore[] forks = new Semaphore[5];

    public DiningPhilosophers() {
        forks[0] = new Semaphore(1);
        forks[1] = new Semaphore(1);
        forks[2] = new Semaphore(1);
        forks[3] = new Semaphore(1);
        forks[4] = new Semaphore(1);
    }

    public void wantsToEat(int philosopher,
                           Runnable pickLeftFork,
                           Runnable pickRightFork,
                           Runnable eat,
                           Runnable putLeftFork,
                           Runnable putRightFork) throws InterruptedException {

        if (philosopher == 3) {
            forks[3].acquire();
            pickRightFork.run();
            forks[4].acquire();
            pickLeftFork.run();
        } else {
            forks[(philosopher + 1) % 5].acquire();
            pickLeftFork.run();
            forks[philosopher].acquire();
            pickRightFork.run();
        }
        eat.run();

        if (philosopher == 3) {
            putLeftFork.run();
            forks[4].release();
            putRightFork.run();
            forks[3].release();
        }
        else{
            putRightFork.run();
            forks[philosopher].release();
            putLeftFork.run();
            forks[(philosopher + 1) % 5].release();
        }

    }

}

