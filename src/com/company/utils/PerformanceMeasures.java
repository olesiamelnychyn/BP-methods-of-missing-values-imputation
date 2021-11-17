package com.company.utils;

import jsat.linear.Vec;
import static java.lang.Math.*;

public class PerformanceMeasures {

    /** Mean-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return mean-squared error
     */
    static public double MSError(Vec actual, Vec predicted){
        int n = actual.length();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double x = predicted.get(i) - actual.get(i);
            sum += x * x;
        }

        return sum/n;
    }

    /** Root Mean-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return root mean-squared error
     */
    static public double RMSError(Vec actual, Vec predicted){
        int n = actual.length();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double x = predicted.get(i) - actual.get(i);
            sum += x * x;
        }

        return sqrt(sum/n);
    }

    /** Mean-Absolute Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return mean-absolute error
     */
    static public double meanAbsoluteError(Vec actual, Vec predicted){
        int n = actual.length();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum+= abs(predicted.get(i) - actual.get(i));
        }

        return sum/n;
    }

    /** Relative-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @param meanTraining mean over training dataset
     * @return relative-squared error
     */
    static public double relativeSquaredError (Vec actual, Vec predicted, double meanTraining) {
        int n = actual.length();
        double sumTop = 0.0, sumBottom = 0.0;
        for (int i = 0; i < n; i++) {
            double x = predicted.get(i) - actual.get(i);
            sumTop += x * x;
            x = actual.get(i) - meanTraining;
            sumBottom += x * x;
        }
        return sumTop / sumBottom;
    }

    /** Root Relative-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @param meanTraining mean over training dataset
     * @return root relative-squared error
     */
    static public double rootRelativeSquaredError (Vec actual, Vec predicted, double meanTraining) {
        int n = actual.length();
        double sumTop = 0, sumBottom = 0;
        for (int i = 0; i < n; i++) {
            double x = predicted.get(i) - actual.get(i);
            sumTop += x * x;
            x = (actual.get(i) - meanTraining);
            sumBottom += x * x;
        }

        return sqrt(sumTop / sumBottom);
    }

    /** Relative-Absolute Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @param meanTraining mean over training dataset
     * @return relative-absolute error
     */
    static public double relativeAbsoluteError (Vec actual, Vec predicted, double meanTraining) {
        int n = actual.length();
        double sumTop = 0.0, sumBottom = 0.0;
        for (int i = 0; i < n; i++) {
            sumTop += abs(predicted.get(i) - actual.get(i));
            sumBottom += abs(actual.get(i) - meanTraining);
        }


        return sumTop / sumBottom;
    }

    /** Mean Absolute Percentage Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return relative-absolute error
     */
    static public double meanAbsolutePercentageError (Vec actual, Vec predicted) {
        int n = actual.length();
        int skipped = 0;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            if (actual.get(i) == 0.0) {
                n--;
                skipped++;
                continue;
            }
            sum += abs((actual.get(i) - predicted.get(i)) / actual.get(i));
        }

        if (skipped != 0) {
            System.out.println("Number of records skipped when calculating Mean Percentage Error due to equality to 0: " + skipped);
        }
        return sum * 100 / n;
    }
}
