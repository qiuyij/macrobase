package macrobase.util;

import com.google.common.collect.Lists;
import macrobase.conf.MacroBaseConf;
import macrobase.datamodel.Datum;
import macrobase.ingest.CSVIngester;
import macrobase.ingest.DataIngester;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper methods for easily loading some sample data
 */
public class SampleData {
    public static List<Datum> loadSimple() throws Exception {
        MacroBaseConf conf = new MacroBaseConf();
        conf.set(MacroBaseConf.CSV_INPUT_FILE, "src/test/resources/data/simple.csv");
        conf.set(MacroBaseConf.ATTRIBUTES, Lists.newArrayList("A2", "A5"));
        conf.set(MacroBaseConf.LOW_METRICS, new ArrayList<>());
        conf.set(MacroBaseConf.HIGH_METRICS, Lists.newArrayList("A1", "A3", "A4"));
        DataIngester loader = new CSVIngester(conf);
        return loader.getStream().drain();
    }


    public static List<Datum> loadVerySimple() throws Exception {
        MacroBaseConf conf = new MacroBaseConf();
        conf.set(MacroBaseConf.CSV_INPUT_FILE, "src/test/resources/data/verySimple.csv");
        conf.set(MacroBaseConf.ATTRIBUTES, Lists.newArrayList("x", "y", "z"));
        conf.set(MacroBaseConf.LOW_METRICS, new ArrayList<>());
        conf.set(MacroBaseConf.HIGH_METRICS, Lists.newArrayList("x", "y", "z"));
        DataIngester loader = new CSVIngester(conf);
        return loader.getStream().drain();
    }

    public static List<double[]> toMetrics(List<Datum> data) {
        ArrayList<double[]> metrics = new ArrayList<>(data.size());
        for (Datum datum : data) {
            metrics.add(datum.getMetrics().toArray());
        }
        return metrics;
    }
}
