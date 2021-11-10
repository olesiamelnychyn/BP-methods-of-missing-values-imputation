package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.io.CSV;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import static java.lang.Math.*;
import static jsat.linear.distancemetrics.PearsonDistance.correlation;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.linear.LUDecomposition;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class DatasetManipulation {

    private static String encodeMissingness (String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        filename = filename.replace(".csv", "_NULL.csv");

        BufferedWriter bw = new BufferedWriter(new FileWriter(filename));

        String line = "";
        while ((line = br.readLine()) != null) {
            String toWrite = "";
            String[] values = line.split(",", -1);

            ArrayList<String> arrayList = new ArrayList();
            for (String value : values) {
                if ("null".equalsIgnoreCase(value) || "?".equals(value) || ("na").equalsIgnoreCase(value) || ("nan").equalsIgnoreCase(value) || value.length() == 0) {
                    arrayList.add("");
                } else {
                    arrayList.add(value);
                }
            }

            for (String s : arrayList) {
                toWrite += s + ",";
            }
            toWrite = toWrite.substring(0, toWrite.length() - 1);
            bw.write(toWrite + "\n");
        }

        br.close();
        bw.close();
        return filename;
    }

    static public SimpleDataSet readDataset (String fileName, boolean containMissing) throws IOException {
        if (containMissing) {
            fileName = encodeMissingness(fileName);
        }

        SimpleDataSet simpleDataSet = CSV.read(Paths.get(fileName), ',', 0, ' ', new HashSet());
        List<DataPoint> list = new ArrayList<>();
        int nans = 0;
        for (DataPoint dp : simpleDataSet.getDataPoints()) {
            if (dp.getNumericalValues().countNaNs() != 0) {
                nans++;
            }
            list.add(dp);
        }
        System.out.println("Number of missing values: " + nans);
        return new SimpleDataSet(list);
    }

    static public SimpleDataSet createDeepCopy (SimpleDataSet dataset, int from, int to) {
        List<DataPoint> ll = new ArrayList<>();
        ll.add(dataset.getDataPoint(from++).clone());
        SimpleDataSet datasetCopy = new SimpleDataSet(ll);
        for (DataPoint obj : dataset.getDataPoints().subList(from, to)) {
            datasetCopy.add(obj.clone());
        }
        return datasetCopy;
    }

    static public SimpleDataSet createDeepCopy (SimpleDataSet dataset, int from1, int to1, int from2, int to2) {
        List<DataPoint> ll = new ArrayList<>();
        ll.add(dataset.getDataPoint(from1++).clone());
        SimpleDataSet datasetCopy = new SimpleDataSet(ll);
        for (DataPoint obj : dataset.getDataPoints().subList(from1, to1)) {
            datasetCopy.add(obj.clone());
        }
        for (DataPoint obj : dataset.getDataPoints().subList(from2, to2)) {
            datasetCopy.add(obj.clone());
        }
        return datasetCopy;
    }

    static public SimpleDataSet createDeepCopyAroundIndex (SimpleDataSet dataset, int index, int columnPredicted, int[] columnPredictors) {
        List<DataPoint> ll = new ArrayList<>();
        int firstIndex = index - 4;
        SimpleDataSet datasetCopy = null;
        int nTraining = 0;
        if (firstIndex != -4) {
            if (firstIndex < 0) {
                firstIndex = 0;
            }
            System.out.print(firstIndex + " ");
            ll.add(dataset.getDataPoint(firstIndex).clone());
            nTraining++;
            datasetCopy = new SimpleDataSet(ll);
            for (DataPoint obj : dataset.getDataPoints().subList(firstIndex + 1, index)) {
                datasetCopy.add(obj.clone());
                nTraining++;
                System.out.print(dataset.getDataPoints().indexOf(obj) + " ");
            }
        }

        int n = dataset.getSampleSize();
        for (int i = index + 1; i < n && nTraining < 8; i++) {
            DataPoint obj = dataset.getDataPoint(i);
            if (!Double.isNaN(obj.getNumericalValues().get(columnPredicted)) && getIntersection(getIndexesOfNull(obj), columnPredictors).length == 0) {
                if (datasetCopy == null) {
                    ll.add(obj.clone());
                    datasetCopy = new SimpleDataSet(ll);
                } else {
                    datasetCopy.add(obj.clone());
                }
                nTraining++;
                System.out.print(i + " ");

            }
        }
        return datasetCopy;
    }

    static public double[][] toArray (SimpleDataSet dataset, int[] columns) {
        double[][] array = new double[dataset.getSampleSize()][columns.length + 1];
        int i = 0;
        for (DataPoint obj : dataset.getDataPoints()) {
            int k = 0;
            array[i][k++] = 1;
            for (int j : columns) {
                array[i][k++] = obj.getNumericalValues().get(j);
            }
            i++;
        }
        return array;
    }

    static public void printDataset (SimpleDataSet dataSet) {
        for (DataPoint dp : dataSet.getDataPoints()) {
            System.out.println(dp.getNumericalValues());
        }
    }

    static public boolean isStrictlyIncreasing (SimpleDataSet dataSet, int columnPredicted) {
        double current = dataSet.getDataPoint(0).getNumericalValues().get(columnPredicted);
        for (int i = 1; i < dataSet.getSampleSize(); i++) {
            if (dataSet.getDataPoint(i).getNumericalValues().get(columnPredicted) < current) {
                return false;
            }
            current = dataSet.getDataPoint(i).getNumericalValues().get(columnPredicted);
        }
        return true;
    }

    static public boolean isStrictlyDecreasing (SimpleDataSet dataSet, int columnPredicted) {
        double current = dataSet.getDataPoint(0).getNumericalValues().get(columnPredicted);
        for (int i = 1; i < dataSet.getSampleSize(); i++) {
            if (dataSet.getDataPoint(i).getNumericalValues().get(columnPredicted) > current) {
                return false;
            }
            current = dataSet.getDataPoint(i).getNumericalValues().get(columnPredicted);
        }
        return true;
    }

    static public boolean isCloseToMean (SimpleDataSet dataSet, int columnPredicted) {
        double std = dataSet.getDataMatrix().getColumn(columnPredicted).standardDeviation();
        double mean = dataSet.getDataMatrix().getColumn(columnPredicted).mean();
//        System.out.println(std + " " + mean + " " + std / mean);
        if (std / mean <= 0.35) {
            return true;
        }
        return false;
    }

    static public boolean isCloseToMedian (SimpleDataSet dataSet, int columnPredicted) {
        double dist = 0.0;
        double median = dataSet.getDataMatrix().getColumn(columnPredicted).median();
        for (DataPoint dp : dataSet.getDataPoints()) {
            dist += abs(dp.getNumericalValues().get(columnPredicted) - median);
        }
//        System.out.println(dist + " " + median + " " + dist / 8 / median);
        if (dist / 8 / median <= 0.2) {
            return true;
        }
        return false;
    }

    static public boolean hasLinearRelationship (SimpleDataSet dataSet, int columnPredicted, int columnPredictor) {
        int[] col = new int[]{
                columnPredicted, columnPredictor
        };
        double corr = correlation(dataSet.getNumericColumn(columnPredicted), dataSet.getNumericColumn(columnPredictor), true);
        System.out.println(corr);
        if (abs(corr) < 0.4) {
            return false;
        }
        return true;
    }

    static public boolean hasLinearRelationship (SimpleDataSet dataSet, int columnPredicted, int[] columnPredictor) {

        int col = columnPredictor.length + 1;

        //count c^T
        RealMatrix mat = new BlockRealMatrix(dataSet.getSampleSize(), columnPredictor.length + 1);
        for (int i = 0; i < col - 1; i++) {
            mat.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
        }
        mat.setColumn(col - 1, dataSet.getDataMatrix().getColumn(columnPredicted).arrayCopy());
        PearsonsCorrelation corr = new PearsonsCorrelation(mat);
        RealMatrix vector = corr.getCorrelationMatrix().getSubMatrix(col - 1, col - 1, 0, corr.getCorrelationMatrix().getColumnDimension() - 2);

        // count Rxx
        RealMatrix mat1 = new BlockRealMatrix(dataSet.getSampleSize(), columnPredictor.length);
        for (int i = 0; i < columnPredictor.length; i++) {
            mat1.setColumn(i, dataSet.getDataMatrix().getColumn(columnPredictor[i]).arrayCopy());
        }
        PearsonsCorrelation corr1 = new PearsonsCorrelation(mat1);
        RealMatrix rxx = corr1.getCorrelationMatrix().scalarMultiply(1 / new LUDecomposition(corr1.getCorrelationMatrix()).getDeterminant());


        RealMatrix a = vector.multiply(vector.transpose());
        RealMatrix b = vector.multiply(rxx).multiply(vector.transpose());

        if (a.getEntry(0, 0) / b.getEntry(0, 0) > 0.45) {
            return false;
        }
        return true;
    }

    static public int getPolynomialOrder (SimpleDataSet dataSet, int columnPredicted, int columnPredictor) {
        int n = dataSet.getSampleSize();
        double[][] powPred = new double[4][n];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < n; j++) {
                double x = dataSet.getDataPoint(i).getNumericalValues().get(columnPredictor);
                powPred[i][j] = getPoly(x, i);
            }
        }
        double[] corr = new double[4];
        for (int i = 0; i < 4; i++) {
            corr[i] = correlation(dataSet.getNumericColumn(columnPredicted), new DenseVector(powPred[i]), true);
        }
        int iMax = getMax(corr);
        System.out.println(corr[iMax] + " " + iMax);
        if (abs(corr[iMax]) < 0.3) {
            return -1;
        }
        return iMax + 1;
    }

    static public SimpleDataSet reverseDataset (SimpleDataSet dataSet) {
        ArrayList<DataPoint> list = new ArrayList<>();
        for (int i = dataSet.getSampleSize() - 1; i >= 0; i--) {
            list.add(dataSet.getDataPoint(i));
        }
        return new SimpleDataSet(list);
    }

    public static SimpleDataSet addPowerColumns (SimpleDataSet dataset, int degree, int[] predictors, int columnPredicted) {
        List<DataPoint> list = new ArrayList<>();
        for (DataPoint dp : dataset.getDataPoints()) {
            Vec vec = new DenseVector(predictors.length + 1);
            vec.set(columnPredicted, dp.getNumericalValues().get(columnPredicted));
            int next = predictors.length / degree + 1;
            for (int i = 0; i < dataset.getDataMatrix().cols(); i++) {
                if (i != columnPredicted) {
                    vec.set(i, dp.getNumericalValues().get(i));
                    for (int j = 2; j <= degree; j++) {
                        vec.set(next, Math.pow(dp.getNumericalValues().get(i), j));
                        next++;
                    }
                }
            }

            list.add(new DataPoint(vec));
        }
        return new SimpleDataSet(list);
    }

    public static SimpleDataSet excludeNonPredictors (SimpleDataSet dataset, int[] predictors, int columnPredicted) {
        List<DataPoint> list = new ArrayList<>();
        for (DataPoint dp : dataset.getDataPoints()) {
            Vec vec = new DenseVector(predictors.length + 1);
            vec.set(0, dp.getNumericalValues().get(columnPredicted));
            int nextIndex = 1;
            for (int column : predictors) {
                vec.set(nextIndex++, dp.getNumericalValues().get(column));
            }
            list.add(new DataPoint(vec));
        }
        return new SimpleDataSet(list);
    }

    static public void printStatistics (SimpleDataSet dataset, int columnPredictor, int columnPredicted) {
        System.out.println("Statistics:\n\tStandard deviation of predictor: " + dataset.getDataMatrix().getColumn(columnPredictor).standardDeviation());
        System.out.println("\tStandard deviation of predicted: " + dataset.getDataMatrix().getColumn(columnPredicted).standardDeviation());
        System.out.println("\tCorrelation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(columnPredictor), dataset.getDataMatrix().getColumn(columnPredicted)) + "\n");
    }

    static private double getPoly (double x, int power) {
        double y = 0.0;
        for (int i = 0; i < power; i++) {
            y += pow(x, i);
        }
        return y;
    }

    static private int getMax (double[] x) {
        int index = 0;
        double max = x[0];
        for (int i = 1; i < x.length; i++) {
            if (x[i] > max) {
                max = x[i];
                index = i;
            }
        }
        return index;
    }

    static public int[] getIndexesOfNull (DataPoint dp) {
        Vec vec = dp.getNumericalValues();
        if (vec.countNaNs() == 0) {
            return new int[0];
        }
        int[] indexes = new int[vec.countNaNs()];
        int j = 0;
        for (int i = 0; i < vec.length(); i++) {
            if (Double.isNaN(vec.get(i))) {
                indexes[j++] = i;
            }
        }
        return indexes;
    }

    public static int[] getIntersection (int[] arr1, int[] arr2) {
        return Arrays.stream(arr1)
                .distinct()
                .filter(x -> Arrays.stream(arr2).anyMatch(y -> y == x))
                .toArray();
    }
}
