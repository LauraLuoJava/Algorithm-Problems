
import algorithm.EasySolution;
import algorithm.HardSolution;
import algorithm.MediumSolution;
import org.junit.jupiter.api.*;

import static org.junit.Assert.assertEquals;

class SolutionTest {
    EasySolution easySolution = new EasySolution();
    MediumSolution mediumSolution = new MediumSolution();
    HardSolution hardSolution = new HardSolution();

    @Test
    void dominator() {
        int result = mediumSolution.dominator(new int[]{2,2,3, 4, 3, 2, 3, -1, 3, 3});
        assertEquals(-1,result);
    }

    @Test
    void task3() {
        int result = mediumSolution.codilityTask3(new int[] {2, 2, 3, 4, 3, 3, 2, 2, 1, 1, 2, 5});
        assertEquals(5,result);
    }
}