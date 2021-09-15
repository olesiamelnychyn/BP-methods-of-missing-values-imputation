package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.io.CSV;
import jsat.math.DescriptiveStatistics;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class DatasetManipulation {

    static public SimpleDataSet readDataset (String fileName) throws IOException {
        SimpleDataSet simpleDataSet = CSV.read(Paths.get(fileName), ',', 0, ' ', new HashSet());
        List<DataPoint> list = new ArrayList<>();
        for (DataPoint dp : simpleDataSet.getDataPoints()) {
            if (dp.getNumericalValues().countNaNs() == 0) {
                list.add(dp);
            }
        }
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

    static public void printStatistics (SimpleDataSet dataset, int columnPredictor, int columnPredicted) {
        System.out.println("Statistics:\n\tStandard deviation of predictor: " + dataset.getDataMatrix().getColumn(columnPredictor).standardDeviation());
        System.out.println("\tStandard deviation of predicted: " + dataset.getDataMatrix().getColumn(columnPredicted).standardDeviation());
        System.out.println("\tCorrelation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(columnPredictor), dataset.getDataMatrix().getColumn(columnPredicted)) + "\n");
    }
}
