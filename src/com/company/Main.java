package com.company;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.math.DescriptiveStatistics;
import jsat.math.SimpleLinearRegression;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {

//        System.out.println("Hello");
//        double[] values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
//        Instance instance = new DenseInstance(values, "positive");
//        System.out.println(instance+"\n");

        Dataset data = new DefaultDataset();
        try {
            data = FileHandler.loadDataset(new File ("src/com/company/Dataset.csv"), 0, ",");

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(data.get(11).values());

        List<DataPoint> listDataPoints = new ArrayList<>();
        for (int i=0; i<data.size(); i++){
            DataPoint dp = new DataPoint(new DenseVector(new ArrayList<Double>(data.get(i).values())));
            listDataPoints.add(dp);
        }
        SimpleDataSet dataset = new SimpleDataSet(listDataPoints);
        System.out.println(dataset.getDataPoints().subList(11, 12).get(0).getNumericalValues());

//        List toBeDeletedList = data.subList(10,20);
//        Dataset toBeDeleted = new DefaultDataset(toBeDeletedList);
//        System.out.println(toBeDeleted);

        // change dataset so that it contains 0.0
        for (int i=10; i<=20; i++){
            Instance instanceX = data.get(i);
            instanceX.put(1,0.0); // there is no 0 in the dataset, so it will be the undefined
        }

        int predicted =4;
        int predictor =2;
        System.out.println( "\n\nStatistics:\nStandard deviation 1: "+ dataset.getDataMatrix().getColumn(predicted).standardDeviation());
        System.out.println( "Standard deviation 2: "+dataset.getDataMatrix().getColumn(predictor).standardDeviation());
        System.out.println( "Correlation Coefficient: "+DescriptiveStatistics.sampleCorCoeff(dataset.getDataMatrix().getColumn(predictor), dataset.getDataMatrix().getColumn(predicted)));
        double[] reg = SimpleLinearRegression.regres( dataset.getDataMatrix().getColumn(predictor), dataset.getDataMatrix().getColumn(predicted));
        System.out.println("a & b: "+reg[0]+" "+reg[1]);

        List<DataPoint> original = new ArrayList<>();
        for(DataPoint obj : dataset.getDataPoints().subList(10, 20)) {
            original.add(obj.clone());
        }

        System.out.println("\n\n(original,predicted):");
        DecimalFormat df2 = new DecimalFormat("#.##");
        for (DataPoint dp: dataset.getDataPoints().subList(10, 20)){
            double newValue= Double.parseDouble(df2.format(reg[0]+reg[1]*dp.getNumericalValues().get(predictor)).replace(',', '.'));
            System.out.println("("+ dp.getNumericalValues().get(predicted)+"," +newValue+")" );
            dp.getNumericalValues().set(predicted, newValue);
        }

        SimpleDataSet org = new SimpleDataSet(original);
        SimpleDataSet nV = new SimpleDataSet(dataset.getDataPoints().subList(10, 20));

        System.out.println( "\nDistance: "+DescriptiveStatistics.sampleCorCoeff(org.getDataMatrix().getColumn(predicted), nV.getDataMatrix().getColumn(predicted)));

    }

}
