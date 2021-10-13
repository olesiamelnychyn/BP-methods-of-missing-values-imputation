package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.io.CSV;
import jsat.math.DescriptiveStatistics;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
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

    static public void printStatistics (SimpleDataSet dataset, int columnPredictor, int columnPredicted) {
        System.out.println("Statistics:\n\tStandard deviation of predictor: " + dataset.getDataMatrix().getColumn(columnPredictor).standardDeviation());
        System.out.println("\tStandard deviation of predicted: " + dataset.getDataMatrix().getColumn(columnPredicted).standardDeviation());
        System.out.println("\tCorrelation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(columnPredictor), dataset.getDataMatrix().getColumn(columnPredicted)) + "\n");
    }
}
