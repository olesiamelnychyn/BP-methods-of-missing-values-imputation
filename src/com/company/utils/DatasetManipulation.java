package com.company.utils;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.io.CSV;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.linear.LUDecomposition;

import static com.company.utils.MathCalculations.*;
import static java.lang.Math.abs;
import static jsat.linear.distancemetrics.PearsonDistance.correlation;


public class DatasetManipulation {

    /** Encode Missingness
     *
     * @param filename original filename
     * @return filename of the dataset with missingness encoded
     *
     * This method replaces different representations of missing values in one which is appropriate for jsat.io.CSV.read(),
     * namely - empty string
     */
    private static String encodeMissingness (String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        filename = filename.replace(".csv", "_NULL.csv");

        BufferedWriter bw = new BufferedWriter(new FileWriter(filename));

        String line = "";
        while ((line = br.readLine()) != null) {
            String toWrite = "";
            String[] values = line.split(",", -1);

            //values of new line which will be written to csv instead of current
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

    /** Read Dataset
     *
     * @param fileName original filename
     * @param containMissing if it contains missing values
     * @return dataset read from file
     *
     * Reads dataset from file. In case it contains missing values, it firstly encodes missingnes,
     * so that jsat.io.CSV.read() do not throw exception.
     */
    static public SimpleDataSet readDataset (String fileName, boolean containMissing) throws IOException {
        if (containMissing) {
            fileName = encodeMissingness(fileName);
        }

        SimpleDataSet simpleDataSet = CSV.read(Paths.get(fileName), ',', 0, ' ', new HashSet());
        //count number of missing values
        int nans = 0;
        for (DataPoint dp : simpleDataSet.getDataPoints()) {
            if (dp.getNumericalValues().countNaNs() != 0) {
                nans++;
            }
        }
        System.out.println("Number of missing values: " + nans);
        return simpleDataSet;
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

    /** Split dataset
     *
     * @param dataset original dataset
     * @param index index of current record
     * @param columnPredicted index of column to be predicted
     * @param columnPredictors index(es) of column(s) to be used for prediction
     * @return ArrayList of datasets: datasets[0] - training dataset,
     *                                datasets[1] - dataset to be predicted
     *
     * This method splits data records close to the predicted index into those which will be used for the prediction and
     * those which will be predicted (since they are close to each other, there might be no sense in predicting them separately)
     */
    static public ArrayList<SimpleDataSet> getToBeImputedAndTrainDeepCopiesAroundIndex (SimpleDataSet dataset, int index, int columnPredicted, int[] columnPredictors) {
        List<DataPoint> dataPointsTrain = new ArrayList<>(); //records used for training
        List<DataPoint> dataPointsToBeImputed = new ArrayList<>();//records in which values should be predicted
        int firstIndex = index - 4; //index of the first record, by default 4 records before current one
        int nTraining = 0; //number of records used for training

        dataPointsToBeImputed.add(dataset.getDataPoints().get(index)); //current record is always the one to be predicted
        if (firstIndex != -4) { //if current record is not the first in dataset
            if (firstIndex < 0) { //if current record is 2nd/3rd
                firstIndex = 0;
            }
            //records before current are always assumed to be already complete as we traverse the dataset from beginning
            for (DataPoint obj : dataset.getDataPoints().subList(firstIndex, index)) {
                dataPointsTrain.add(obj.clone());
                nTraining++;
            }
        }

        int n = dataset.getSampleSize(); //the last index in dataset
        //filling dataset used for prediction till we have 8 records in it or there are no more records in dataset after current
        for (int i = index + 1; i < n && nTraining < 8; i++) {
            DataPoint obj = dataset.getDataPoint(i);

            //if a record has one or more of the predictors' equal to null it cannot be used for prediction
            if (getIntersection(getIndexesOfNull(obj), columnPredictors).length == 0) {
                //if record contains value in column which is going to be predicted
                //then add it to the dataset used for prediction,
                if (!Double.isNaN(obj.getNumericalValues().get(columnPredicted))) {
                    dataPointsTrain.add(obj.clone());
                    nTraining++;
                } else {
                    //otherwise add it to the dataset which is going to be predicted
                    dataPointsToBeImputed.add(obj);
                }
            }
        }

        ArrayList<SimpleDataSet> datasets = new ArrayList<>();
        datasets.add(new SimpleDataSet(dataPointsTrain));
        datasets.add(new SimpleDataSet(dataPointsToBeImputed));
        return datasets;
    }

    /** Convert columns of dataset to double[][] array
     *
     * @param dataset original dataset
     * @param columns array of indexes of columns to be present in array
     * @return reternes converted part of the dataset
     *
     * It converts dataset to an array, which contains only values in the specified columns and
     * additional column as constant that equals to 1
     */
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

    static public boolean isStrictlyIncreasing (SimpleDataSet dataSet, int column) {
        double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
        for (int i = 1; i < dataSet.getSampleSize(); i++) {
            if (dataSet.getDataPoint(i).getNumericalValues().get(column) < current) {
                return false;
            }
            current = dataSet.getDataPoint(i).getNumericalValues().get(column);
        }
        return true;
    }

    static public boolean isStrictlyDecreasing (SimpleDataSet dataSet, int column) {
        double current = dataSet.getDataPoint(0).getNumericalValues().get(column);
        for (int i = 1; i < dataSet.getSampleSize(); i++) {

            if (dataSet.getDataPoint(i).getNumericalValues().get(column) > current) {
                return false;
            }
            current = dataSet.getDataPoint(i).getNumericalValues().get(column);
        }
        return true;
    }

    static public boolean isCloseToMean (SimpleDataSet dataSet, int columnPredicted) {
        double std = dataSet.getDataMatrix().getColumn(columnPredicted).standardDeviation();
        double mean = dataSet.getDataMatrix().getColumn(columnPredicted).mean();
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
        if (dist / 8 / median <= 0.2) {
            return true;
        }
        return false;
    }

    //calculate pearson correlation with one predictor
    static public boolean hasLinearRelationship (SimpleDataSet dataSet, int columnPredicted, int columnPredictor) {
        int[] col = new int[]{
                columnPredicted, columnPredictor
        };
        double corr = correlation(dataSet.getNumericColumn(columnPredicted), dataSet.getNumericColumn(columnPredictor), true);
        if (abs(corr) < 0.4) {
            return false;
        }
        return true;
    }

    //calculate pearson correlation with multiple predictors
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
        //create raw polynomials of values in columnPredictor
        double[][] powPred = new double[4][n];
        for (int pow = 1; pow < 4; pow++) {
            for (int index = 0; index < n; index++) {
                double x = dataSet.getDataPoint(index).getNumericalValues().get(columnPredictor);
                powPred[pow][index] = getPolyWithoutCoefficients(x, pow);
            }
        }
        // calculate correlations between raw polynomials and values in columnPredicted
        double[] corr = new double[4];
        for (int i = 0; i < 4; i++) {
            corr[i] = correlation(dataSet.getNumericColumn(columnPredicted), new DenseVector(powPred[i]), true);
        }
        int iMax = getMax(corr); //choose the best correlation
        if (abs(corr[iMax]) < 0.3) {
            return -1;
        }
        return iMax + 1;
    }

    /** Reverse dataset
     *
     * @param dataSet original dataset
     * @return new reversed dataset
     */
    static public SimpleDataSet reverseDataset (SimpleDataSet dataSet) {
        ArrayList<DataPoint> list = new ArrayList<>();
        for (int i = dataSet.getSampleSize() - 1; i >= 0; i--) {
            list.add(dataSet.getDataPoint(i));
        }
        return new SimpleDataSet(list);
    }

    /** Remove columns of non-predictors
     *
     * @param dataset original dataset
     * @param degree max degree which should be in new dataset used for multiple polynomial regression
     * @param columnPredicted index of column to be predicted
     * @param predictors index(es) of column(s) to be used for prediction
     * @return new dataset which contains columns with powers of values (from 1 to degree)
     *
     * This method add columns of predictors' powers to the dataset
     */
    public static SimpleDataSet addPowerColumns (SimpleDataSet dataset, int degree, int[] predictors, int columnPredicted) {
        List<DataPoint> list = new ArrayList<>();
        for (DataPoint dp : dataset.getDataPoints()) {
            // new numerical values of a record which look like: [x1 x2... x1^2 x2^2... ... x1^n x2^n...]
            Vec vec = new DenseVector(predictors.length + 1);
            vec.set(columnPredicted, dp.getNumericalValues().get(columnPredicted));
            int next = predictors.length / degree + 1;
            for (int i = 0; i < dataset.getDataMatrix().cols(); i++) {
                if (i != columnPredicted) {
                    vec.set(i, dp.getNumericalValues().get(i)); //set x
                    for (int j = 2; j <= degree; j++) {
                        vec.set(next, Math.pow(dp.getNumericalValues().get(i), j)); //set x^2, ..., x^n
                        next++;
                    }
                }
            }

            list.add(new DataPoint(vec));
        }
        return new SimpleDataSet(list);
    }

    /** Remove columns of non-predictors
     *
     * @param dataset original dataset
     * @param columnPredicted index of column to be predicted
     * @param predictors index(es) of column(s) to be used for prediction
     * @return filename of the dataset with missingness encoded
     *
     * This method removes column(s) which is(are) not used for prediction
     */
    public static SimpleDataSet excludeNonPredictors (SimpleDataSet dataset, int[] predictors, int columnPredicted) {
        List<DataPoint> list = new ArrayList<>();
        for (DataPoint dp : dataset.getDataPoints()) {
            //new data record which contains only predicted value and values of predictors
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

    static public void printDataset (SimpleDataSet dataSet) {
        for (DataPoint dp : dataSet.getDataPoints()) {
            System.out.println(dp.getNumericalValues());
        }
    }

    static public void printStatistics (SimpleDataSet dataset, int columnPredictor, int columnPredicted) {
        System.out.println("Statistics:\n\tStandard deviation of predictor: " + dataset.getDataMatrix().getColumn(columnPredictor).standardDeviation());
        System.out.println("\tStandard deviation of predicted: " + dataset.getDataMatrix().getColumn(columnPredicted).standardDeviation());
        System.out.println("\tPearson Correlation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(columnPredictor), dataset.getDataMatrix().getColumn(columnPredicted)) + "\n");
    }

}
