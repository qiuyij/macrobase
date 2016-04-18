package macrobase.analysis.stats.density;

import macrobase.util.SampleData;
import org.junit.Test;

import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;

public class NTreeKDETest {
    @Test
    public void simpleTest() throws Exception {
        List<double[]> data = SampleData.toMetrics(SampleData.loadVerySimple());

        NKDTree tree = new NKDTree().setLeafCapacity(3);
        NTreeKDE kde = new NTreeKDE(tree).setTolerance(0.0);
        kde.train(data);

        KDESimple kdeSimple = new KDESimple();
        kdeSimple.train(data);

        assertArrayEquals(kdeSimple.getBandwidth(), kde.getBandwidth(), 1e-10);
        for (double[] datum : data) {
            assertEquals(kdeSimple.score(datum), kde.score(datum), 1e-10);
        }
    }

    public void approxTest(
            List<double[]> data,
            double tol,
            double cutoff
    ) {
        NKDTree tree = new NKDTree().setLeafCapacity(3);
        NTreeKDE kde = new NTreeKDE(tree)
                .setTolerance(tol)
                .setCutoff(cutoff);
        kde.train(data);
        tol = Math.max(tol, 1e-10);

        KDESimple kdeSimple = new KDESimple();
        kdeSimple.train(data);

        for (double[] d: data) {
            double trueDensity = kdeSimple.density(d);
            double estDensity = kde.density(d);

            if (trueDensity < cutoff) {
                assertEquals(trueDensity, estDensity, tol);
            }
        }
    }

    @Test
    public void testTolerance() throws Exception {
        List<double[]> data = SampleData.toMetrics(SampleData.loadVerySimple());
        approxTest(data, 1e-5, 0.0);
    }

    @Test
    public void testCutoff() throws Exception {
        List<double[]> data = SampleData.toMetrics(SampleData.loadVerySimple());
        approxTest(data, 0.0, 7e-4);
    }

    @Test
    public void testToleranceCutoff() throws Exception {
        List<double[]> data = SampleData.toMetrics(SampleData.loadVerySimple());
        approxTest(data, 1e-5, 7e-4);
    }
}
