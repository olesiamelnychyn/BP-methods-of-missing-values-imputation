/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.company.imputationMethods.regressions;


/******************************************************************************
 *  Compilation:  javac -classpath jama.jar:. MultipleLinearRegression.java
 *  Execution:    java  -classpath jama.jar:. MultipleLinearRegression
 *  Dependencies: jama.jar
 *
 *  Compute least squares solution to X beta = y using Jama library.
 *  Assumes X has full column rank.
 *
 *       http://math.nist.gov/javanumerics/jama/
 *       http://math.nist.gov/javanumerics/jama/Jama-1.0.1.jar
 *
 * https://algs4.cs.princeton.edu/14analysis/MultipleLinearRegression.java.html
 ******************************************************************************/

import Jama.Matrix;
import Jama.QRDecomposition;


public class MultipleLinearRegressionJama {
    private final Matrix beta;  // regression coefficients
    private int nCoeff;
    private double sse;         // sum of squared
    private double sst;         // sum of squared

    /**
     * Performs a linear regression on the data points {@code (y[i], x[i][j])}.
     * @param  x the values of the predictor variables
     * @param  y the corresponding values of the response variable
     * @throws IllegalArgumentException if the lengths of the two arrays are not equal
     */
    public MultipleLinearRegressionJama (double[][] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("matrix dimensions don't agree");
        }

        // number of observations
        int n = y.length;

        Matrix matrixX = new Matrix(x);

        // create matrix from vector
        Matrix matrixY = new Matrix(y, n);

        // find least squares solution
        QRDecomposition qr = new QRDecomposition(matrixX);
        beta = qr.solve(matrixY);
        nCoeff = beta.getRowDimension();

        // mean of y[] values
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += y[i];
        double mean = sum / n;

        // total variation to be accounted for
        for (int i = 0; i < n; i++) {
            double dev = y[i] - mean;
            sst += dev * dev;
        }

        // variation not accounted for
        Matrix residuals = matrixX.times(beta).minus(matrixY);
        sse = residuals.norm2() * residuals.norm2();

    }

    /**
     * Returns the least squares estimate of &beta;<sub><em>j</em></sub>.
     *
     * @param  j the index
     * @return the estimate of &beta;<sub><em>j</em></sub>
     */
    public double beta (int j) {
        return beta.get(j, 0);
    }

    /**
     * Returns the coefficient of determination <em>R</em><sup>2</sup>.
     *
     * @return the coefficient of determination <em>R</em><sup>2</sup>,
     *         which is a real number between 0 and 1
     */
    public double R2 () {
        return 1.0 - sse / sst;
    }

    /**
     * Returns the expected response {@code y} given the value of the predictor
     * variable {@code x}.
     *
     * @param x the value of the predictor variable
     * @return the expected response {@code y} given the value of the predictor
     * variable {@code x}
     */
    public double predict (double[] x) {
        double y = 0.0;
        for (int i = 0; i < nCoeff; i++) {
            y += beta(i) * x[i];
        }
        return y;
    }

    /**
     * Unit tests the {@code MultipleLinearRegression} data type.
     *
     * @param args the command-line arguments
     */
//    public static void main (String[] args) {
//        double[][] x = {{1, 10, 20},
//                {1, 20, 40},
//                {1, 40, 15},
//                {1, 80, 100},
//                {1, 160, 23},
//                {1, 200, 18}};
//        double[] y = {243, 483, 508, 1503, 1764, 2129};
//        MultipleLinearRegressionJama regression = new MultipleLinearRegressionJama(x, y);
//
//        System.out.printf("%.2f + %.2f beta1 + %.2f beta2  (R^2 = %.2f)\n",
//                regression.beta(0), regression.beta(1), regression.beta(2), regression.R2());
//    }
}
