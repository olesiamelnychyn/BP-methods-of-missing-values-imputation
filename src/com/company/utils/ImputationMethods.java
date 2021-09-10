package com.company.utils;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.math.DescriptiveStatistics;
import jsat.math.SimpleLinearRegression;
import jsat.regression.MultipleLinearRegression;
import jsat.regression.RegressionDataSet;
import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.fitting.GaussianCurveFitter;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.PerformanceMeasures.*;

public class ImputationMethods {
	int columnPredicted;
	SimpleDataSet dataset;
	SimpleDataSet original;
	DecimalFormat df2 = new DecimalFormat("#.##");
	int datasetSize=1000;

	public ImputationMethods (int columnPredicted, SimpleDataSet dataSet, SimpleDataSet original) {
		this.columnPredicted = columnPredicted;
		dataset = dataSet;
		this.original = original;
	}

	public void setDatasetSize (int datasetSize) {
		this.datasetSize = datasetSize;
	}

	public void LinearRegression (int columnPredictor) throws IOException {
		String method = "LinearRegression (columnPredictor=" + columnPredictor + ")";
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, datasetSize);
//        DatasetManipulation.printStatistics(dataset, columnPredictor, columnPredicted);
		double[] reg = SimpleLinearRegression.regres(datasetCopy.getDataMatrix().getColumn(columnPredictor), datasetCopy.getDataMatrix().getColumn(columnPredicted));
		System.out.println(ANSI_PURPLE_BACKGROUND+method+ANSI_RESET+"\na & b: [" + reg[0] + "," + reg[1]+"]");
//        System.out.println("\n\n(original,columnPredicted):");
		for (DataPoint dp : datasetCopy.getDataPoints().subList(10, 20)) {
			double newValue = Double.parseDouble(df2.format(reg[0] + reg[1] * dp.getNumericalValues().get(columnPredictor)).replace(',', '.'));
//            System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted), method);

	}

	public void MultipleLinearRegression () throws IOException {
		String method = "MultipleLinearRegression";
		System.out.println(ANSI_PURPLE_BACKGROUND+method+ANSI_RESET);
		RegressionDataSet regressionDataSet = new SimpleDataSet(dataset.shallowClone().getDataPoints().subList(0, datasetSize)).asRegressionDataSet(columnPredicted);
		SimpleDataSet datasetCopy2 = DatasetManipulation.createDeepCopy(dataset, 0, datasetSize);
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
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted), method);
	}

	public void PolynomialCurveFitter(int columnPredictor ) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND+method+ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, datasetSize);
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(3);

		for (DataPoint dp : datasetCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length-1 ; i++) {
			System.out.print(coeff[i]+",");
		}
		System.out.println(coeff[coeff.length-1]+"]");

		for (DataPoint dp : datasetCopy.getDataPoints().subList(10, 20)) {
			double newValue = Double.parseDouble(df2.format(polyValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
//			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted), method);

	}

	public void GaussianCurveFitter(int columnPredictor ) throws IOException {
		String method= "GaussianCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND+method+ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, datasetSize);
		final GaussianCurveFitter fitter = GaussianCurveFitter.create();

		for (DataPoint dp : datasetCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length-1 ; i++) {
			System.out.print(coeff[i]+",");
		}
		System.out.println(coeff[coeff.length-1]+"]");

		for (DataPoint dp : datasetCopy.getDataPoints().subList(10, 20)) {
			double newValue = Double.parseDouble(df2.format(polyValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
//			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted), method);

	}

	public void LinearInterpolator(int columnPredictor ) throws IOException {
		String method="PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println("\n"+ANSI_PURPLE_BACKGROUND+method+ANSI_RESET);
		LinearInterpolator linearInterpolator = new LinearInterpolator();
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, datasetSize);
		PolynomialSplineFunction polynomialSplineFunction = linearInterpolator.interpolate(datasetCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), datasetCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());
		double [] coeff = polynomialSplineFunction.getKnots();

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length-1 ; i++) {
			System.out.print(coeff[i]+",");
		}
		System.out.println(coeff[coeff.length-1]+"]");


		for (DataPoint dp : datasetCopy.getDataPoints().subList(10, 20)) {
			double newValue = Double.parseDouble(df2.format(polynomialSplineFunction.value(dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		SimpleDataSet predicted = new SimpleDataSet(datasetCopy.getDataPoints().subList(10, 20));
		printPerformanceMeasures(original.getDataMatrix().getColumn(columnPredicted), predicted.getDataMatrix().getColumn(columnPredicted), method);
	}

	static public void printPerformanceMeasures (Vec original, Vec predicted, String method) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt", true));
		writer.append(method);
		writer.append("\n\nPerformance:");
		writer.append("\n\tMean-Squared Error: " + MSError(original, predicted));
		writer.append("\n\tRoot Mean-Squared Error: " + RMSError(original, predicted));
		writer.append("\n\tMean-Absolute Error: " + meanAbsoluteError(original, predicted));
		writer.append("\n\tRelative-Squared Error: " + relativeSquaredError(original, predicted));
		writer.append("\n\tRoot Relative-Squared Error: " + rootRelativeSquaredError(original, predicted));
		writer.append("\n\tRelative-Absolute Error: " + relativeAbsoluteError(original, predicted));
		writer.append("\n\tCorrelation Coefficient: " + DescriptiveStatistics.sampleCorCoeff(original, predicted) + "\n\n");
		writer.close();

		System.out.println("Performance:");
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + RMSError(original, predicted) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + relativeAbsoluteError(original, predicted) + ANSI_RESET + ANSI_BOLD_OFF+"\n\n");

	}

	public static double polyValue(double [] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double val = p[p.length-1];

		for (int i = p.length-2; i >= 0; i--) {
			val = val * x0 + p[i];
		}

		return val;
	}


}
