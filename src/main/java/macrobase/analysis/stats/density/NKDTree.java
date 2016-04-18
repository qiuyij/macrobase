package macrobase.analysis.stats.density;

import com.google.common.base.Strings;
import macrobase.util.AlgebraUtils;

import java.util.*;

public class NKDTree {
    // Core Data
    protected NKDTree loChild;
    protected NKDTree hiChild;
    protected int k;
    protected double[][] items;

    // Parameters
    private int leafCapacity = 20;
    private boolean splitByWidth = false;

    // Statistics
    private int splitDimension;
    protected int nBelow;
    protected double[] mean;
    private double splitValue;
    // Array of (k,2) dimensions, of (min, max) pairs in all k dimensions
    private double[][] boundaries;

    public NKDTree() {
        splitDimension = 0;
    }

    public NKDTree(NKDTree parent) {
        this.k = parent.k;
        splitDimension = (parent.splitDimension + 1) % k;

        leafCapacity = parent.leafCapacity;
        splitByWidth = parent.splitByWidth;
    }

    public NKDTree setSplitByWidth(boolean f) {
        this.splitByWidth = f;
        return this;
    }
    public NKDTree setLeafCapacity(int leafCapacity) {
        this.leafCapacity = leafCapacity;
        return this;
    }

    public NKDTree build(List<double[]> data) {
        this.k = data.get(0).length;
        this.boundaries = AlgebraUtils.getBoundingBoxRaw(data);

        if (data.size() > this.leafCapacity) {
            Collections.sort(data, Comparator.comparing((double[] arr) -> arr[splitDimension]));
            int splitIndex = pickSplitIndex(data);
            double belowSplit = data.get(splitIndex - 1)[splitDimension];
            double aboveSplit = data.get(splitIndex)[splitDimension];
            this.splitValue = 0.5 * (belowSplit + aboveSplit);

            this.loChild = new NKDTree(this).build(data.subList(0, splitIndex));
            this.hiChild = new NKDTree(this).build(data.subList(splitIndex, data.size()));
            this.nBelow = data.size();

            this.mean = new double[k];
            for (int i = 0; i < k; i++) {
                this.mean[i] = (loChild.mean[i] * loChild.getNBelow() + hiChild.mean[i] * hiChild.getNBelow())
                        / (loChild.getNBelow() + hiChild.getNBelow());
            }
        } else {
            this.items = data.toArray(new double[][] {});
            this.nBelow = data.size();

            double[] sum = new double[k];
            for (double[] d : data) {
                for (int i = 0; i < k; i++) {
                    sum[i] += d[i];
                }
            }

            for (int i = 0; i < k; i++) {
                sum[i] /= this.nBelow;
            }

            this.mean = sum;
        }
        return this;
    }

    /**
     * Pick the index at which we want to split our list of vectors
     * @param data List of metrics sorted by the splitDimension
     * @return right-inclusive index we wish to split at
     */
    protected int pickSplitIndex(List<double[]> data) {
        if (!this.splitByWidth) {
            return data.size() / 2;
        } else {
            int n = data.size();
            double midPoint = (data.get(n/10)[splitDimension] + data.get(9*n/10)[splitDimension])/2;
            int i = 0;
            for (i = 0; i < data.size(); i++) {
                if (data.get(i)[splitDimension] >= midPoint) {
                    break;
                }
            }

            if (i == 0 || i >= data.size()-1) {
                return data.size()/2;
            } else {
                return i;
            }
        }
    }

    /**
     * Estimates min and max difference absolute vectors from point to region
     * @return minVec, maxVec
     */
    public double[][] getMinMaxDistanceVectors(double[] q) {
        double[][] minMaxDiff = new double[2][k];

        for (int i=0; i<k; i++) {
            double d1 = q[i] - boundaries[i][0];
            double d2 = q[i] - boundaries[i][1];
            // outside to the right
            if (d2 >= 0) {
                minMaxDiff[0][i] = d2;
                minMaxDiff[1][i] = d1;
            }
            // inside, min distance is 0;
            else if (d1 >= 0) {
                minMaxDiff[1][i] = d1 > -d2 ? d1 : -d2;
            }
            // outside to the left
            else {
                minMaxDiff[0][i] = -d1;
                minMaxDiff[1][i] = -d2;
            }
        }

        return minMaxDiff;
    }

    /**
     * Estimates bounds on the distance to a region
     * @return array with min, max distances squared
     */
    public double[] getMinMaxDistances(double[] q) {
        double[][] diffVectors = getMinMaxDistanceVectors(q);
        double[] estimates = new double[2];
        for (int i = 0; i < k; i++) {
            double minD = diffVectors[0][i];
            double maxD = diffVectors[1][i];
            estimates[0] += minD * minD;
            estimates[1] += maxD * maxD;
        }
        return estimates;
    }

    public boolean isInsideBoundaries(double[] q) {
        for (int i=0; i<k; i++) {
            if (q[i] < this.boundaries[i][0] || q[i] > this.boundaries[i][1]) {
                return false;
            }
        }
        return true;
    }

    public double[][] getItems() {
        return this.items;
    }

    public double[] getMean() {
        return this.mean;
    }

    public int getNBelow() {
        return nBelow;
    }

    public double[][] getBoundaries() {
        return this.boundaries;
    }

    public NKDTree getLoChild() {
        return this.loChild;
    }

    public NKDTree getHiChild() {
        return this.hiChild;
    }

    public boolean isLeaf() {
        return this.loChild == null && this.hiChild == null;
    }

    public int getSplitDimension() {
        return splitDimension;
    }

    public double getSplitValue() {
        return splitValue;
    }

    public String toString(int indent) {
        int nextIndent = indent + 1;
        String tabs = Strings.repeat(" ", nextIndent);
        if (loChild != null && hiChild != null) {
            return String.format(
                    "KDNode: dim=%d split=%.3f \n%sLO: %s\n%sHI: %s",
                    this.splitDimension, this.splitValue,
                    tabs, this.loChild.toString(nextIndent),
                    tabs, this.hiChild.toString(nextIndent));
        }
        else if (hiChild!= null) {
            return String.format(
                    "KDNode: dim=%d split=%.3f \n%sHI: %s",
                    this.splitDimension, this.splitValue,
                    tabs, this.hiChild.toString(nextIndent));
        }
        else if (loChild != null) {
            return String.format(
                    "KDNode: dim=%d split=%.3f \n%sLO: %s",
                    this.splitDimension, this.splitValue,
                    tabs, this.loChild.toString(nextIndent));
        }
        else {
            String all = "KDNode:\n";
            for (double[] d: this.items) {
                all += String.format("%s - %s\n", tabs, Arrays.toString(d));
            }
            return all;
        }

    }

    public String toString() {
        return this.toString(0);
    }
}
