package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.math.DescriptiveStatistics;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DatasetManipulation {

    static public SimpleDataSet readDataset(String fileName){

//        System.out.println("Hello");
//        double[] values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//        Instance instance = new DenseInstance(values, "positive");
//        System.out.println(instance+"\n");

        Dataset data = new DefaultDataset();
        try {
            data = FileHandler.loadDataset(new File(fileName), 0, ",");

        } catch (IOException e) {
            e.printStackTrace();
        }

        List<DataPoint> listDataPoints = new ArrayList<>();
        for (int i=0; i<data.size(); i++){
            DataPoint dp = new DataPoint(new DenseVector(new ArrayList<>(data.get(i).values())));
            listDataPoints.add(dp);
        }

        return new SimpleDataSet(listDataPoints);
    }

    static public SimpleDataSet createDeepCopy(SimpleDataSet dataset, int from, int to){
        List<DataPoint> ll = new ArrayList<>();
        ll.add(dataset.getDataPoint(from++).clone());
        SimpleDataSet datasetCopy = new SimpleDataSet(ll);
        for(DataPoint obj : dataset.getDataPoints().subList(from,to)) {
            datasetCopy.add(obj.clone());
        }
         return datasetCopy;
    }

    static public void printStatistics(SimpleDataSet dataset, int columnPredictor, int columnPredicted){
        System.out.println( "Statistics:\n\tStandard deviation of predictor: "+ dataset.getDataMatrix().getColumn(columnPredictor).standardDeviation());
        System.out.println( "\tStandard deviation of predicted: "+dataset.getDataMatrix().getColumn(columnPredicted).standardDeviation());
        System.out.println( "\tCorrelation Coefficient: "+ DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(columnPredictor), dataset.getDataMatrix().getColumn(columnPredicted))+"\n");
    }
}
