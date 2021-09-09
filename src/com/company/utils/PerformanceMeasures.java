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
        int n= actual.length();
        double sum=0;
        for (int i = 0; i < n; i++) {
            sum+= pow(predicted.get(i) - actual.get(i), 2);
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
        int n= actual.length();
        double sum=0;
        for (int i = 0; i < n; i++) {
            sum+= pow(predicted.get(i) - actual.get(i), 2);
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
        int n= actual.length();
        double sum=0;
        for (int i = 0; i < n; i++) {
            sum+= abs(predicted.get(i) - actual.get(i));
        }

        return sum/n;
    }

    /** Relative-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return relative-squared error
     */
    static public double relativeSquaredError(Vec actual, Vec predicted){
        int n= actual.length();
        double meanTraining=actual.mean();
        double sumTop=0, sumBottom=0;
        for (int i = 0; i < n; i++) {
            sumTop=pow(predicted.get(i)- actual.get(i), 2);
            sumBottom=pow(actual.get(i) - meanTraining, 2);
        }

        return sumTop/sumBottom;
    }

    /** Root Relative-Squared Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return root relative-squared error
     */
    static public double rootRelativeSquaredError(Vec actual, Vec predicted){
        int n = actual.length();
        double meanTraining=actual.mean();
        double sumTop=0, sumBottom=0;
        for (int i = 0; i < n; i++) {
            sumTop=pow(predicted.get(i)- actual.get(i), 2);
            sumBottom=pow(actual.get(i) - meanTraining, 2);
        }

        return sqrt(sumTop/sumBottom);
    }

    /** Relative-Absolute Error
     *
     * @param predicted predicted values
     * @param actual original values
     * @return relative-absolute error
     */
    static public double relativeAbsoluteError(Vec actual, Vec predicted){
        int n= actual.length();
        double meanTraining=actual.mean();
        double sumTop=0, sumBottom=0;
        for (int i = 0; i < n; i++) {
            sumTop=abs(predicted.get(i)- actual.get(i));
            sumBottom=abs(actual.get(i) - meanTraining);
        }

        return sumTop/sumBottom;
    }
}
