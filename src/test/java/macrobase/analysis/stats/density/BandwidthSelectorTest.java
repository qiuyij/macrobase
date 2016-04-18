package macrobase.analysis.stats.density;

import macrobase.util.SampleData;
import org.junit.Test;

import java.util.List;

import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertThat;

public class BandwidthSelectorTest {
    @Test
    public void testVerySimple() throws Exception {
        List<double[]> data = SampleData.toMetrics(SampleData.loadVerySimple());
        BandwidthSelector s = new BandwidthSelector();
        double[] bw = s.findBandwidth(data);
        for (int i=0; i < bw.length; i++) {
            assertThat(bw[i], lessThan(5.0));
            assertThat(bw[i], greaterThan(2.0));
        }
    }
}
