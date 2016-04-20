package macrobase.analysis.stats.mixture;

import macrobase.analysis.stats.BatchMixtureModel;
import macrobase.analysis.stats.distribution.MultivariateTDistribution;
import macrobase.analysis.stats.distribution.Wishart;
import macrobase.conf.MacroBaseConf;
import macrobase.conf.MacroBaseDefaults;
import macrobase.datamodel.Datum;
import macrobase.util.AlgebraUtils;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.special.Gamma;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Variational Dirichlet Process Mixture of Gaussians.
 */
public class VariationalDPMG extends BatchMixtureModel {
    private static final Logger log = LoggerFactory.getLogger(VariationalDPMG.class);
    private final int maxIterationsToConverge;

    // Number of truncated clusters.
    private int T;
    // Concentration parameter for the Dirichlet distribution.
    private double concentrationParameter;
    private double progressCutoff;

    // Variables governing atoms.
    // Omega and atomNu for Wishart distribution for the precision matrix of the clusters.
    private double atomNu[];
    private List<RealMatrix> atomOmega;
    // Parameters for Normal distribution for atom position, N(atomLocation, (atomBeta * Lambda))
    // where Lambda is Wishart distributed given parameters above.
    private double atomBeta[];
    private List<RealVector> atomLocation;

    // Parameters describing stick lengths, i.e. shape parameters of Beta distributions.
    private double shapeParams[][];

    // Parameters for multinomial distributions for data points. i.e. density corresponding to each cluster (atom).
    private double[][] r;

    // Parameters for Base Distribution of the DP process, which is Wishart-Gaussian
    private double baseNu;
    private RealMatrix baseOmega;
    private RealMatrix inverseBaseOmega;  // Use inverse of baseOmega, since it is used in the update equations.
    private double baseBeta;
    private RealVector baseLocation;

    private List<MultivariateTDistribution> predictiveDistributions;

    private void updatePredictiveDistributions() {
        int dimension = atomLocation.get(0).getDimension();

        predictiveDistributions = new ArrayList<>(T);
        for (int t = 0; t < T; t++) {
            double scale = (atomNu[t] + 1 - dimension) * atomBeta[t] / (1 + atomBeta[t]);
            RealMatrix ll = AlgebraUtils.invertMatrix(atomOmega.get(t).scalarMultiply(scale));
            // TODO: MultivariateTDistribution should support real values for 3rd parameters
            predictiveDistributions.add(new MultivariateTDistribution(atomLocation.get(t), ll, (int) (atomNu[t] + 1 - dimension)));
        }
    }

    public VariationalDPMG(MacroBaseConf conf) {
        super(conf);
        T = conf.getInt(MacroBaseConf.DPM_TRUNCATING_PARAMETER, MacroBaseDefaults.DPM_TRUNCATING_PARAMETER);
        concentrationParameter = conf.getDouble(MacroBaseConf.DPM_CONCENTRATION_PARAMETER, MacroBaseDefaults.DPM_CONCENTRATION_PARAMETER);
        progressCutoff = conf.getDouble(MacroBaseConf.EM_CUTOFF_PROGRESS, MacroBaseDefaults.EM_CUTOFF_PROGRESS);
        maxIterationsToConverge = conf.getInt(MacroBaseConf.MIXTURE_MAX_ITERATIONS_TO_CONVERGE, MacroBaseDefaults.MIXTURE_MAX_ITERATIONS_TO_CONVERGE);
    }

    private void initializeBaseDistribution(List<Datum> data) {
        int dimension = data.get(0).getMetrics().getDimension();
        baseNu = dimension;
        double[][] boundingBox = AlgebraUtils.getBoundingBox(data);
        double[] midpoints = new double[dimension];
        double[] dimensionWidth = new double[dimension];
        double R = 0;  // value of the widest dimension.
        for (int i = 0; i < dimension; i++) {
            dimensionWidth[i] = boundingBox[i][1] - boundingBox[i][0];
            midpoints[i] = boundingBox[i][0] + dimensionWidth[i];
            if (dimensionWidth[i] > R) {
                R = dimensionWidth[i];
            }
        }
        baseBeta = Math.pow(R, -2);
        baseLocation = new ArrayRealVector(midpoints);
        baseOmega = MatrixUtils.createRealIdentityMatrix(dimension);
        inverseBaseOmega = AlgebraUtils.invertMatrix(baseOmega);
    }

    @Override
    public void train(List<Datum> data) {
        int N = data.size();
        int dimension = data.get(0).getMetrics().getDimension();

        instanciateVariationalParameters(N, T);
        initializeBaseDistribution(data);

        // 0. Initialize all approximating factors

        // stick lengths
        for (int i = 0; i < T; i++) {
            shapeParams[i][0] = 1;
            shapeParams[i][1] = concentrationParameter;
        }

        // atoms
        atomLocation = gonzalezInitializeMixtureCenters(data, T);
        for (int i = 0; i < T; i++) {
            // initialize betas as if all points are from the first cluster.
            atomBeta[i] = 1;

            atomNu[i] = baseNu;
            atomOmega.add(0, AlgebraUtils.invertMatrix(inverseBaseOmega));
        }

        final double dimensionLn2 = dimension * Math.log(2);
        final double halfDimensionLn2Pi = 0.5 * dimension * Math.log(2 * Math.PI);

        List<RealMatrix> S;  // initialized and used in step 2.
        double logLikelihood = -Double.MAX_VALUE;
        double[] clusterWeight;
        List<RealVector> clusterMean;

        double ELBO = -Double.MAX_VALUE;

        for (int iteration = 1; ; iteration++) {
            // 0. Initialize volatile parameters
            clusterMean = new ArrayList<>(T);
            for (int t = 0; t < T; t++) {
                clusterMean.add(new ArrayRealVector(new double[dimension]));
            }
            clusterWeight = new double[T];

            //double newElbo = ex_lnp_sticks + ex_lnp_atoms + ex_lnp_data + ex_lnp_mult - ex_lnq_sticks - ex_lnq_atoms - ex_lnq_mult;

            double oldElbo = ELBO;

            //double ex_lnq_mult =

            // 1. calculate expectation of densities of each point coming from individual clusters - r[n][k]
            // 1. Reevaluate r[][]
            double[] lnLambdaContribution = new double[T];
            double[] lnMixingContribution = new double[T];
            double cumulativeAlreadyAssigned = 0;
            // for (int t=0; t<atomLocation.size(); t++) {
            for (int t = 0; t < T; t++) {
                // Calculate Per cluster 0.5 ln L_t - D/2 ln(2 pi) contributions.
                lnLambdaContribution[t] = dimensionLn2 + Math.log((new EigenDecomposition(atomOmega.get(t))).getDeterminant());
                for (int i = 0; i < dimension; i++) {
                    lnLambdaContribution[t] += Gamma.digamma((atomNu[t] - i) / 2);
                }
                lnLambdaContribution[t] /= 2;
                // Calculate Mixing coefficient contributions to r
                lnMixingContribution[t] = cumulativeAlreadyAssigned + (Gamma.digamma(shapeParams[t][0]) - Gamma.digamma(shapeParams[t][0] + shapeParams[t][1]));
                cumulativeAlreadyAssigned += Gamma.digamma(shapeParams[t][1]) - Gamma.digamma(shapeParams[t][0] + shapeParams[t][1]);
            }


            double lnXMuLambdaContribution;
            for (int n = 0; n < N; n++) {
                double normalizingConstant = 0;
                for (int t = 0; t < T; t++) {
                    RealVector _diff = data.get(n).getMetrics().subtract(atomLocation.get(t));
                    if (atomBeta[t] != 0) {
                        lnXMuLambdaContribution = dimension / atomBeta[t] + atomNu[t] * _diff.dotProduct(atomOmega.get(t).operate(_diff));
                    } else {
                        lnXMuLambdaContribution = atomNu[t] * _diff.dotProduct(atomOmega.get(t).operate(_diff));
                    }
                    r[n][t] = Math.exp(lnMixingContribution[t] - halfDimensionLn2Pi + lnLambdaContribution[t] - lnXMuLambdaContribution);
                    normalizingConstant += r[n][t];
                }
                for (int t = 0; t < atomLocation.size(); t++) {
                    if (normalizingConstant > 0) {
                        r[n][t] /= normalizingConstant;
                    }
                    // Calculate unnormalized cluster weight, cluster mean
                    clusterWeight[t] += r[n][t];
                    clusterMean.set(t, clusterMean.get(t).add(data.get(n).getMetrics().mapMultiply(r[n][t])));
                }
            }

            log.debug("after 1, ELBO = {}", calculateELBO(data));

            // 2. Reevaluate clusters based on densities that we have for each point.
            // 2. Reevaluate atoms and stick lengths.
            S = new ArrayList<>(T);

            for (int t = 0; t < T; t++) {
                S.add(new BlockRealMatrix(dimension, dimension));
                if (clusterWeight[t] > 0) {
                    clusterMean.set(t, clusterMean.get(t).mapDivide(clusterWeight[t]));
                } else {
                    continue;
                }
                for (int n = 0; n < N; n++) {
                    RealVector _diff = data.get(n).getMetrics().subtract(clusterMean.get(t));
                    S.set(t, S.get(t).add(_diff.outerProduct(_diff).scalarMultiply(r[n][t])));
                }
                S.set(t, S.get(t).scalarMultiply(1 / clusterWeight[t]));
            }

            for (int t = 0; t < T; t++) {
                shapeParams[t][0] = 1;
                shapeParams[t][1] = concentrationParameter;
                for (int n = 0; n < N; n++) {
                    shapeParams[t][0] += r[n][t];
                    for (int j = t + 1; j < T; j++) {
                        shapeParams[t][1] += r[n][j];
                    }
                }
                atomBeta[t] = baseBeta + clusterWeight[t];
                atomLocation.set(t, baseLocation.mapMultiply(baseBeta).add(clusterMean.get(t).mapMultiply(clusterWeight[t])).mapDivide(atomBeta[t]));
                atomNu[t] = baseNu + 1 + clusterWeight[t];
                RealMatrix wInverse = inverseBaseOmega.add(
                        S.get(t).scalarMultiply(clusterWeight[t])).add(
                        clusterMean.get(t).outerProduct(clusterMean.get(t)).scalarMultiply(baseBeta * clusterWeight[t] / (baseBeta + clusterWeight[t])));
                atomOmega.set(t, AlgebraUtils.invertMatrix(wInverse));
            }

            updatePredictiveDistributions();

            log.debug("after 2, ELBO = {}", calculateELBO(data));

            double oldLogLikelihood = logLikelihood;
            double sumWeights = 0;
            for (double w : getClusterWeights()) {
                sumWeights += w;
            }
            log.debug("cluster weights = {}, sum={}", getClusterWeights(), sumWeights);
            logLikelihood = 0;
            for (int n = 0; n < N; n++) {
                logLikelihood += Math.log(score(data.get(n)));
            }

            log.info("log likelihood after iteration {} is {}", iteration, logLikelihood);

            if (iteration >= maxIterationsToConverge) {
                log.info("Breaking because have already run {} iterations", iteration);
                break;
            }

            double improvement = (logLikelihood - oldLogLikelihood) / (-logLikelihood);
            if (improvement >= 0 && improvement < this.progressCutoff) {
                log.debug("Breaking because improvement was {} percent", improvement * 100);
                break;
            } else {
                log.debug("improvement is : {}%", improvement * 100);
            }
            log.debug(".........................................");
        }
    }

    private double calculateELBO(List<Datum> data) {
        return getDataExpectLogP(data)
                + getAssignExpectLogP()
                + getStickExpectLogP()
                + getAtomExpectLogP()
                - getAssignExpectLogQ()
                - getStickExpectLogQ()
                - getAtomExpectLogQ();
    }

    /**
     * depends on
     * - r
     * - atoms
     *
     * @return
     */
    private double getDataExpectLogP(List<Datum> data) {
        int D = atomOmega.get(0).getColumnDimension();
        double ex_lnp_data = D * Math.log(2 * Math.PI);
        for (int i = 1; i < T; i++) {
            double clusterWeight = calculateClusterWeight(i);
            RealVector center = calculateWeightedClusterSum(data, i).mapDivide(clusterWeight);
            RealMatrix S = calculateUnscaledS(data, i, center).scalarMultiply(1. / clusterWeight);
            RealVector _diff = center.subtract(atomLocation.get(i));

            ex_lnp_data += clusterWeight * (
                    new Wishart(atomOmega.get(i), atomNu[i])).getExpectationLogDeterminantLambda()
                    - D / atomBeta[i]
                    - atomNu[i] * S.multiply(atomOmega.get(i)).getTrace()
                    - atomNu[i] * (_diff.dotProduct(atomOmega.get(i).operate(_diff)));
        }
        return ex_lnp_data;
    }

    private double calculateClusterWeight(int cluster) {
        double weight = 0;
        for (double[] probas : r) {
            weight += probas[cluster];
        }
        return weight;
    }


    private RealMatrix calculateUnscaledS(List<Datum> data, int cluster, RealVector center) {
        int D = data.get(0).getMetrics().getDimension();
        RealMatrix unscaledS = new BlockRealMatrix(D, D);
        for (int n = 0; n < data.size(); n++) {
            RealVector _diff = data.get(n).getMetrics().subtract(center);
            unscaledS = unscaledS.add(_diff.outerProduct(_diff).scalarMultiply(r[n][cluster]));
        }
        return unscaledS;
    }

    private RealVector calculateWeightedClusterSum(List<Datum> data, int cluster) {
        RealVector sum = new ArrayRealVector(data.get(0).getMetrics().getDimension());
        for (int n = 0; n < data.size(); n++) {
            sum = sum.add(data.get(n).getMetrics().mapMultiply(r[n][cluster]));
        }
        return sum;
    }

    /**
     * depends on
     * - r
     * - sticks
     *
     * @return
     */
    private double getAssignExpectLogP() {
        int N = r.length;
        double logP = 0;
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < T; i++) {
                double q_greter = 0;
                for (int j = i + 1; j < T; j++) {
                    q_greter += r[n][j];
                }
                logP += q_greter * (Gamma.digamma(shapeParams[i][1]) - Gamma.digamma(shapeParams[i][0] + shapeParams[i][1]))
                        + r[n][i] * (Gamma.digamma(shapeParams[i][0] - Gamma.digamma(shapeParams[i][0] + shapeParams[i][1])));
            }
        }
        return logP;
    }

    /**
     * depends on
     * - sticks
     *
     * @return
     */
    private double getStickExpectLogP() {
        //return T * (Gamma.digamma(1) - Gamma.digamma(1 + concentrationParameter));  FIXME: WRONG!
        return 0;
    }

    /**
     * depends on
     * - atoms
     *
     * @return
     */
    private double getAtomExpectLogP() {
        int D = atomOmega.get(0).getColumnDimension();
        double ex_lnp_atom = T * (new Wishart(baseOmega, baseNu)).lnB()
                + T * 0.5 * D * Math.log(baseBeta / 2 / Math.PI);
        for (int i = 0; i < T; i++) {
            RealVector _diff = atomLocation.get(i).subtract(baseLocation);
            ex_lnp_atom += 0.5 * (baseNu - D) * (new Wishart(atomOmega.get(i), atomNu[i])).getExpectationLogDeterminantLambda()
                    - D * baseBeta / atomBeta[i]
                    - baseBeta * atomNu[i] * _diff.dotProduct(atomOmega.get(i).operate(_diff))
                    - 0.5 * atomNu[i] * inverseBaseOmega.multiply(atomOmega.get(i)).getTrace();
        }
        return ex_lnp_atom;
    }

    /**
     * depends on
     * - atoms
     *
     * @return
     */
    private double getAtomExpectLogQ() {
        int dataDimension = atomOmega.get(0).getColumnDimension();
        double ex_lnq_atoms = 0;
        for (int t = 0; t < T; t++) {
            Wishart wishart = new Wishart(atomOmega.get(t), atomNu[t]);
            ex_lnq_atoms += 0.5 * wishart.getExpectationLogDeterminantLambda()
                    + 0.5 * dataDimension * Math.log(atomBeta[t] / 2 / Math.PI)
                    - 0.5 * dataDimension
                    - wishart.getEntropy();
        }
        return ex_lnq_atoms;
    }

    /**
     * depends on
     * - r
     *
     * @return
     */
    private double getAssignExpectLogQ() {
        double ex_lnq_mult = 0;
        for (double[] clusterProbas : r) {
            for (double rnk : clusterProbas) {
                ex_lnq_mult += rnk * Math.log(rnk);
            }
        }
        return ex_lnq_mult;
    }

    /**
     * depends on
     * - sticks
     *
     * @return
     */
    private double getStickExpectLogQ() {
        return 0;
        //double ex_lnq_sticks = 0; //FIXME: Probably wrong.
        //for (int i = 0; i < T; i++) {
        //    ex_lnq_sticks += Gamma.digamma(shapeParams[i][0]) - Gamma.digamma(shapeParams[i][0] + shapeParams[i][1]);
        //}
        //return ex_lnq_sticks;
    }

    private void instanciateVariationalParameters(int numPoints, int maxNumClusters) {
        shapeParams = new double[maxNumClusters][2];
        r = new double[numPoints][maxNumClusters];
        atomLocation = new ArrayList<>(maxNumClusters);
        atomOmega = new ArrayList<>(maxNumClusters);
        atomNu = new double[maxNumClusters];
        atomBeta = new double[maxNumClusters];
    }

    @Override
    public double score(Datum datum) {
        double density = 0;
        double[] stickLengths = getClusterWeights();
        for (int i = 0; i < predictiveDistributions.size(); i++) {
            density += stickLengths[i] * predictiveDistributions.get(i).density(datum.getMetrics());
        }
        return density;
    }

    @Override
    public double getZScoreEquivalent(double zscore) {
        return 0;
    }

    public double[] getClusterWeights() {
        double[] proportions = new double[T];
        double stickRemaining = 1;
        double expectedBreak;
        for (int i = 0; i < T; i++) {
            expectedBreak = stickRemaining / (1 + shapeParams[i][1] / shapeParams[i][0]);
            stickRemaining -= expectedBreak;
            proportions[i] = expectedBreak;
        }
        // if (stickRemaining > 0) {
        //     for (int i = 0; i < T; i++) {
        //         proportions[i] /= (1 - stickRemaining);
        //     }
        // }
        return proportions;
    }

    public List<RealVector> getClusterCenters() {
        return atomLocation;
    }

    public List<RealMatrix> getClusterCovariances() {
        List<RealMatrix> covariances = new ArrayList<>(T);
        for (int t = 0; t < T; t++) {
            covariances.add(AlgebraUtils.invertMatrix(atomOmega.get(t).scalarMultiply(atomNu[t])));
        }
        return covariances;
    }
}