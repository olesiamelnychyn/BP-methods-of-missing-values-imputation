package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;
import jsat.math.SimpleLinearRegression;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;

import java.text.DecimalFormat;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.PerformanceMeasures.*;

public class Regressions {
	int columnPredicted;
	SimpleDataSet dataset;
	SimpleDataSet original;
	DecimalFormat df2 = new DecimalFormat("#.##");

	public Regressions (int columnPredicted, SimpleDataSet dataSet, SimpleDataSet original) {
		this.columnPredicted = columnPredicted;
		dataset = dataSet;
		this.original = original;
	}

	public void LinearRegression (int columnPredictor) {
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, 1000);
//        DatasetManipulation.printStatistics(dataset, columnPredictor, columnPredicted);
		double[] reg = SimpleLinearRegression.regres(datasetCopy.getDataMatrix().getColumn(columnPredictor), datasetCopy.getDataMatrix().getColumn(columnPredicted));
		System.out.println(ANSI_PURPLE_BACKGROUND+"LinearRegression (columnPredictor=" + columnPredictor + ")"+ANSI_RESET+"\na & b: [" + reg[0] + "," + reg[1]+"]");
//        System.out.println("\n\n(original,columnPredicted):");
		for (DataPoint dp : datasetCopy.getDataPoints().subList(10, 20)) {
			double newValue = Double.parseDouble(df2.format(reg[0] + reg[1] * dp.getNumericalValues().get(columnPredictor)).replace(',', '.'));
//            System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted));

	}

	public void MultipleLinearRegression () {
		System.out.println(ANSI_PURPLE_BACKGROUND+"MultipleLinearRegression"+ANSI_RESET);
		RegressionDataSet regressionDataSet = new SimpleDataSet(dataset.shallowClone().getDataPoints().subList(0, 1000)).asRegressionDataSet(columnPredicted);
		SimpleDataSet datasetCopy2 = DatasetManipulation.createDeepCopy(dataset, 0, 1000);
		MultipleLinearRegression multipleLinearRegression = new MultipleLinearRegression();
		multipleLinearRegression.train(regressionDataSet);

		System.out.println("Weights: "+multipleLinearRegression.getRawWeight());

		int index = 10;
		for (DataPoint dp : regressionDataSet.getDataPoints().subList(10, 20)) {
			DataPoint simple = datasetCopy2.getDataPoint(index);
			index++;
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.regress(dp)).replace(',', '.'));
//            System.out.println("("+ simple.getNumericalValues().get(columnPredicted)+"," +newValue+")" );
			simple.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy2.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted));
	}

	static public void printPerformanceMeasures (Vec original, Vec predicted) {
		System.out.println("\nPerformance:");
		System.out.println("\tMean-Squared Error: " + MSError(original, predicted));
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + RMSError(original, predicted) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println("\tMean-Absolute Error: " + meanAbsoluteError(original, predicted));
		System.out.println("\tRelative-Squared Error: " + relativeSquaredError(original, predicted));
		System.out.println("\tRoot Relative-Squared Error: " + rootRelativeSquaredError(original, predicted));
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + relativeAbsoluteError(original, predicted) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println("\tCorrelation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(original, predicted) + "\n\n");
	}
}
