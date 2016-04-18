package macrobase.util;

import org.apache.commons.math3.linear.*;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;

public class AlgebraUtilsTest {
    private static final Logger log = LoggerFactory.getLogger(AlgebraUtilsTest.class);

    @Test
    /**
     * test that v^T M v == vec(M) vec(v v^T)
     */
    public void testVectorization() {
        double[][] matrixContents = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
        };
        double[] flattenedMatrixContents = {
                1, 4, 7, 2, 5, 8, 3, 6, 9,
        };
        double[] vectorContents = {
                1, 2, 3,
        };

        RealMatrix matrix = new BlockRealMatrix(matrixContents);
        RealVector vectorizedMatrix = AlgebraUtils.vectorize(matrix);
        RealVector vector = new ArrayRealVector(vectorContents);
        assertEquals(vectorizedMatrix, new ArrayRealVector(flattenedMatrixContents));
        assertEquals(vector.dotProduct(matrix.operate(vector)), vectorizedMatrix.dotProduct(AlgebraUtils.vectorize(vector.outerProduct(vector))));
    }

    @Test
    public void testBoundingBox() {
        double[][] matrixContents = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, -9},
        };
        List<double[]> data = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            data.add(matrixContents[i]);
        }
        double[][] bbox = AlgebraUtils.getBoundingBoxRaw(data);

        assertEquals(1.0, bbox[0][0], 1e-10);
        assertEquals(7.0, bbox[0][1], 1e-10);
        assertEquals(-9.0, bbox[2][0], 1e-10);
        assertEquals(6.0, bbox[2][1], 1e-10);
    }
}
