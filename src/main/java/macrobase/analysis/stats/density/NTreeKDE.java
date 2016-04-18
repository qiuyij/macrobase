package macrobase.analysis.stats.density;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class NTreeKDE {
    private static final Logger log = LoggerFactory.getLogger(NTreeKDE.class);

    // ** Basic stats parameters
    private int numPoints;
    private double[] bandwidth;
    private Kernel kernel;

    // ** Tree parameters
    private NKDTree tree;
    // Total density error tolerance
    private double tolerance = 0;
    // Cutoff point at which point we no longer need accuracy
    private double cutoff = Double.MAX_VALUE;

    // ** Other parameters
    private boolean debugTraversal = false;

    // ** Diagnostic Measurements
    public long finalCutoff[] = new long[10];
    public long numNodesProcessed[] = new long[10];
    public int numScored = 0;

    // ** Cached values
    private double unscaledTolerance;
    private double unscaledCutoff;

    public NTreeKDE(NKDTree tree) {
        this.tree = tree;
    }

    public NTreeKDE setTolerance(double t) {this.tolerance = t; return this;}
    public NTreeKDE setCutoff(double cutoff) {this.cutoff = cutoff; return this;}
    public NTreeKDE setBandwidth(double[] bw) {this.bandwidth = bw; return this;}
    public NTreeKDE setKernel(Kernel k) {this.kernel = k; return this;}
    public NTreeKDE setDebugTraversal(boolean f) {this.debugTraversal = f; return this;}

    public double[] getBandwidth() {return bandwidth;}

    public void train(List<double[]> data) {
        this.numPoints = data.size();
        this.unscaledTolerance = tolerance * numPoints;
        this.unscaledCutoff = cutoff * numPoints;

        if (bandwidth == null) {
            bandwidth = new BandwidthSelector().findBandwidth(data);
        }
        if (kernel == null) {
            kernel = new GaussianKernel();
        }
        kernel.initialize(bandwidth);

        log.debug("training kd-tree KDE on {} points", data.size());
        this.tree.build(data);
    }

    /**
     * Calculates density * N
     * @param d query point
     * @return unnormalized density
     */
    private double pqScore(double[] d) {
        ScoreEstimate curEstimate = new ScoreEstimate(this.kernel, this.tree, d);
        Comparator<ScoreEstimate> c = (o1, o2) -> {
            if (o1.totalWMax < o2.totalWMax) {
                return 1;
            } else if (o1.totalWMax > o2.totalWMax) {
                return -1;
            } else {
                return 0;
            }
        };
        PriorityQueue<ScoreEstimate> pq = new PriorityQueue<>(100, c);
        pq.add(curEstimate);

        double totalWMin = curEstimate.totalWMin;
        double totalWMax = curEstimate.totalWMax;
        long curNodesProcessed = 1;

//        System.out.println("\nScoring : "+Arrays.toString(d));
//        System.out.println("tolerance: "+unscaledTolerance);
//        System.out.println("cutoff: "+unscaledCutoff);
        while (!pq.isEmpty()) {
//            System.out.println("minmax: "+totalWMin+", "+totalWMax);
            if (totalWMax - totalWMin < unscaledTolerance) {
                numNodesProcessed[0] += curNodesProcessed;
                finalCutoff[0]++;
                break;
            } else if (totalWMin > unscaledCutoff) {
                numNodesProcessed[1] += curNodesProcessed;
                finalCutoff[1]++;
                break;
            }
            curEstimate = pq.poll();
//            System.out.println("current box:\n"+ DiagnosticsUtils.array2dToString(curEstimate.tree.getBoundaries()));
//            System.out.println("split: "+curEstimate.tree.getSplitDimension() + ":"+curEstimate.tree.getSplitValue());
            totalWMin -= curEstimate.totalWMin;
            totalWMax -= curEstimate.totalWMax;

            if (curEstimate.tree.isLeaf()) {
                double exact = exactDensity(curEstimate.tree, d);
                totalWMin += exact;
                totalWMax += exact;
            } else {
                ScoreEstimate[] children = curEstimate.split(this.kernel, d);
                curNodesProcessed += 2;
                for (ScoreEstimate child : children) {
                    totalWMin += child.totalWMin;
                    totalWMax += child.totalWMax;
                    pq.add(child);
                }
            }
        }
        if (pq.isEmpty()) {
            finalCutoff[3]++;
        }
        numScored++;
        return (totalWMin + totalWMax) / 2;
    }

    private double exactDensity(NKDTree t, double[] d) {
        double score = 0.0;
        for (double[] dChild : t.getItems()) {
            double[] diff = d.clone();
            for (int i = 0; i < diff.length; i++) {
                diff[i] -= dChild[i];
            }
            score += kernel.density(diff);
        }
        return score;

    }

    /**
     * Return the negative log pdf density, this avoids underflow errors while still being
     * an interpretable quantity. Use density if you need the actual negative pdf.
     */
    public double score(double[] d) {
        double unscaledScore = pqScore(d);
        // Note: return score with a minus sign, s.t. outliers are selected not inliers.
        return -(Math.log(unscaledScore) - Math.log(numPoints));
    }

    public void showDiagnostics() {
        log.info("Final Loop Cutoff: tol {}, totalcutoff {}, completion {}",
                finalCutoff[0],
                finalCutoff[1],
                finalCutoff[2]);
        log.info("Avg # of nodes processed: tol {}, totalcutoff {}",
                (double)numNodesProcessed[0]/finalCutoff[0],
                (double)numNodesProcessed[1]/finalCutoff[1]
                );
    }

    /**
     * Returns normalized pdf
     */
    public double density(double[] d) {
        return pqScore(d) / numPoints;
    }
}
