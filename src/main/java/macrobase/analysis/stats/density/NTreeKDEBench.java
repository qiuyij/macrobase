package macrobase.analysis.stats.density;

import com.fasterxml.jackson.dataformat.yaml.snakeyaml.Yaml;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import macrobase.conf.MacroBaseConf;
import macrobase.datamodel.Datum;
import macrobase.ingest.CSVIngester;
import macrobase.ingest.DataIngester;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

public class NTreeKDEBench {
    // *******
    // PARAMETERS
    // *******

    // data
    public String inputFile;
    public List<String> columns;
    public List<String> attributes;

    // KDE
    public Supplier<Kernel> kernelFactory;
    public double percentile;
    public int trainSize;
    public int testSize;
    public int sampleSize;
    public double bwMultiplier;

    // KDTree
    public double relTolerance;
    public double relCutoff;
    public boolean splitByWidth;

    // Grid parameters
    public boolean useGrid;
    public List<Double> gridSizeList;

    // UI
    public boolean waitForUser;
    public boolean showTreeTraversal;
    public boolean dumpScores;

    // calculated params
    double[] bw;
    double quantileEstimate;

    private static final Logger log = LoggerFactory.getLogger(NTreeKDEBench.class);

    public static void main(String[] args) throws Exception {
        new NTreeKDEBench().run();
    }

    public void loadParams(String path) throws IOException {
        Yaml yaml = new Yaml();
        BufferedReader in = Files.newBufferedReader(Paths.get(path));
        Map<String, Object> testConf = (Map<String, Object>) yaml.load(in);

        inputFile = (String)testConf.get("inputFile");
        columns = (List<String>)testConf.get("columns");
        attributes = (List<String>)testConf.get("attributes");

        String kernelStr = (String)testConf.get("kernel");
        if (kernelStr.equals("gaussian")) {
            kernelFactory = GaussianKernel::new;
        } else {
            kernelFactory = EpaKernel::new;
        }
        percentile = (double)testConf.get("percentile");
        trainSize = (int)testConf.get("trainSize");
        testSize = (int)testConf.get("testSize");
        relTolerance = (double)testConf.get("relTolerance");
        relCutoff = (double)testConf.get("relCutoff");
        sampleSize = (int)testConf.get("sampleSize");
        bwMultiplier = (double)testConf.get("bwMultiplier");

        splitByWidth = (boolean)testConf.get("splitByWidth");
        useGrid = (boolean)testConf.get("useGrid");
        gridSizeList = (List<Double>)testConf.get("gridSizes");

        waitForUser = (boolean)testConf.get("waitForUser");
        showTreeTraversal = (boolean)testConf.get("showTreeTraversal");
        dumpScores = (boolean)testConf.get("dumpScores");

    }

    public void run() throws Exception {
        loadParams("src/test/resources/conf/test_kde.yaml");
        StopWatch sw = new StopWatch();

        sw.start();
        List<double[]> metrics = loadMetrics();
        sw.stop();
        log.info("Loaded Data and shuffled in {}", sw.toString());

        // Calculating basic parameters
        sw.reset();
        sw.start();
        this.bw = new BandwidthSelector()
                .setMultiplier(bwMultiplier)
                .findBandwidth(metrics);
        sw.stop();
        log.info("Bandwidth: {} in {}", Arrays.toString(bw), sw.toString());

        sw.reset();
        sw.start();
        estimateQuantile(metrics.subList(0, sampleSize));
        sw.stop();
        log.info("Estimated p{}: {} in {}", percentile, quantileEstimate, sw.toString());

        // Initialize cutoff grids
        Kernel k = kernelFactory.get().initialize(bw);
        List<SoftGridCutoff> grids = new ArrayList<>();
        double rawGridCutoff = trainSize * relCutoff * quantileEstimate;
        for (double gridScaleFactor : gridSizeList) {
            double[] gridSize = bw.clone();
            for (int i = 0; i < gridSize.length; i++) {
                gridSize[i] *= gridScaleFactor;
            }
            SoftGridCutoff g = new SoftGridCutoff(k, gridSize, rawGridCutoff);
            grids.add(g);
//            SoftGridCutoff g2 = new SoftGridCutoff(k, gridSize, rawGridCutoff).setOffset(.5);
//            grids.add(g2);
        }

        // Final score calculations
        List<double[]> train = metrics.subList(0, trainSize);
        List<double[]> test = metrics.subList(metrics.size()/2, metrics.size());

        NKDTree tree = new NKDTree()
                .setLeafCapacity(20)
                .setSplitByWidth(splitByWidth);
        NTreeKDE kde = new NTreeKDE(tree)
                .setKernel(kernelFactory.get())
                .setBandwidth(bw)
                .setTolerance(relTolerance * quantileEstimate)
                .setCutoff(relCutoff * quantileEstimate)
                .setDebugTraversal(showTreeTraversal);

        sw.reset();
        sw.start();
        kde.train(train);
        sw.stop();
        log.info("Trained tree in {}", sw.toString());

        sw.reset();
        sw.start();
        for (SoftGridCutoff g : grids) {
            for (double[] d : train) {
                g.add(d);
            }
        }
        grids.stream().forEach(SoftGridCutoff::prune);
        log.info("Trained grid in {}", sw.toString());
        log.info("Grid sizes: {}",
                Arrays.toString(grids.stream().map(SoftGridCutoff::getNumCells).toArray())
        );

        if (waitForUser) {
            // Makes it easier to attach profiler
            System.in.read();
        }

        long start = System.currentTimeMillis();
        double[] scores = new double[testSize];
        ArrayList<double[]> outliers = new ArrayList<>((int)(2*testSize*percentile));
        ArrayList<double[]> almostOutliers = new ArrayList<>(testSize);

        int testDataSize = test.size();
        int numUsingGrid = 0;
        for (int i = 0; i < testSize; i++) {
            double[] d = test.get(i % testDataSize);
            double gScore = 0.0;
            if (useGrid && grids.size() > 0) {
                for (SoftGridCutoff g : grids) {
                    double curScore = g.getDenseValue(d);
                    if (curScore > 0) {
                        gScore = curScore;
                        break;
                    }
                }
            }
            if (gScore > 0.0) {
                numUsingGrid++;
                scores[i] = gScore / trainSize;
            } else {
                scores[i] = kde.density(d);
                if (scores[i] < quantileEstimate) {
                    outliers.add(d);
                } else {
                    almostOutliers.add(d);
                }
            }
        }

        log.info("{} points eliminated using grid", numUsingGrid);
        long elapsed = System.currentTimeMillis() - start;
        log.info("Scored {} @ {} / s", testSize, (float)testSize * 1000/(elapsed));
        kde.showDiagnostics();

        Percentile p = new Percentile();
        p.setData(scores);
        System.out.println("KDTree Score Quantiles:");
        System.out.println(p.evaluate(percentile*100));
        System.out.println(".1:"+p.evaluate(.1));
        System.out.println("10:"+p.evaluate(10));

        if (dumpScores) {
            dumpToFile(scores);
            dumpOutliersToFile(outliers, almostOutliers);
        }
    }

    public void estimateQuantile(List<double[]> sample) {
        KDESimple kde = new KDESimple()
                .setKernel(kernelFactory.get())
                .setBandwidth(bw);

        int n = sample.size();
        List<double[]> train = sample.subList(0, n/2);
        List<double[]> test = sample.subList(n/2, n);
        kde.train(train);

        double[] densities = new double[test.size()];
        for (int i=0; i<test.size(); i++) {
            densities[i] = kde.density(test.get(i));
        }

        Percentile p = new Percentile();
        p.setData(densities);

        System.out.println("Estimates:");
        double quantile = p.evaluate(percentile*100);
        System.out.println(quantile);
        for (double pp=1; pp<5; pp+=1) {
            System.out.println(pp+":"+p.evaluate(pp));
        }
        for (double pp=95; pp<100; pp+=1) {
            System.out.println(pp+":"+p.evaluate(pp));
        }
        quantileEstimate = quantile;
    }

    private String getTestSuffix() {
        int gval = useGrid ? 1 : 0;
        return String.format(
                "cutoff%dp_tol%dp_grid%d",
                (int)(relCutoff*100),
                (int)(relTolerance*100),
                gval);
    }

    private void dumpOutliersToFile(
            List<double[]> outliers,
            List<double[]> almostOutliers
    ) throws IOException {
        String fName = "treekde_outliers_"+getTestSuffix()+".txt";
        PrintWriter out = new PrintWriter(Files.newBufferedWriter(Paths.get("target", fName)));
        out.println(Joiner.on(",").join(columns));
        for (double[] vec : outliers) {
            String line = Joiner.on(",").join(Arrays.stream(vec).mapToObj(Double::toString).iterator());
            out.println(line);
        }
        out.close();

        fName = "treekde_almostOutliers_"+getTestSuffix()+".txt";
        out = new PrintWriter(Files.newBufferedWriter(Paths.get("target", fName)));
        out.println(Joiner.on(",").join(columns));
        for (double[] vec : almostOutliers) {
            String line = Joiner.on(",").join(Arrays.stream(vec).mapToObj(Double::toString).iterator());
            out.println(line);
        }
        out.close();
    }

    private void dumpToFile(
            double[] scores
    ) throws IOException {
        String densityFileName = "treekde_densities_"+getTestSuffix()+".txt";
        PrintWriter out = new PrintWriter(Files.newBufferedWriter(Paths.get("target", densityFileName)));
        out.println("density");
        for (double s : scores) {
            out.println(s);
        }
        out.close();
    }

    public List<double[]> loadMetrics() throws Exception {
        MacroBaseConf conf = new MacroBaseConf();
        conf.set(
                MacroBaseConf.CSV_INPUT_FILE,
                "src/test/resources/data/"+inputFile
        );
        conf.set(MacroBaseConf.CSV_COMPRESSION, "GZIP");
        conf.set(MacroBaseConf.ATTRIBUTES, attributes);
        conf.set(MacroBaseConf.LOW_METRICS, new ArrayList<>());
        conf.set(MacroBaseConf.HIGH_METRICS, columns);
        DataIngester loader = new CSVIngester(conf);
        List<Datum> data = loader.getStream().drain();

        Random r = new Random();
        r.setSeed(0);
        Collections.shuffle(data, r);

        ArrayList<double[]> metrics = new ArrayList<>(data.size());
        for (Datum datum : data) {
            metrics.add(datum.getMetrics().toArray());
        }

        return metrics;
    }
}
