package com.company.utils;

import com.company.utils.regressions.*;
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
import java.util.List;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.PerformanceMeasures.*;

public class ImputationMethods {
	int columnPredicted;
	SimpleDataSet dataset;
	SimpleDataSet test;
	SimpleDataSet training;
	DecimalFormat df2 = new DecimalFormat("#.##");

	public ImputationMethods (int columnPredicted, SimpleDataSet dataSet) {
		this.columnPredicted = columnPredicted;
		dataset = dataSet;

		this.test = DatasetManipulation.createDeepCopy(dataset, 5, 6);
		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(dataset, 0, 30);
		List<DataPoint> training = datasetCopy.getDataPoints();
		for (int i = 5; i < 6; i++) {
			training.remove(datasetCopy.getDataPoint(i));
		}
		this.training = new SimpleDataSet(training);
	}

	public void LinearRegressionJSAT (int columnPredictor) throws IOException {
		String method = "LinearRegression (columnPredictor=" + columnPredictor + ")";
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());

		System.out.println("Size:" + trainingCopy.getSampleSize() + " " + testCopy.getSampleSize());
		DatasetManipulation.printStatistics(dataset, columnPredictor, columnPredicted);
		double[] reg = SimpleLinearRegression.regres(trainingCopy.getDataMatrix().getColumn(columnPredictor), trainingCopy.getDataMatrix().getColumn(columnPredicted));
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET + "\na & b: [" + reg[0] + "," + reg[1] + "]");
//        System.out.println("\n\n(test,columnPredicted):");
		for (DataPoint dp : testCopy.getDataPoints()) {
			double newValue = Double.parseDouble(df2.format(reg[0] + reg[1] * dp.getNumericalValues().get(columnPredictor)).replace(',', '.'));
			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);

	}

	public void MultipleLinearRegressionJSAT () throws IOException {
		String method = "MultipleLinearRegression";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		RegressionDataSet regressionDataSet = new SimpleDataSet(trainingCopy.shallowClone().getDataPoints().subList(0, trainingCopy.getSampleSize())).asRegressionDataSet(columnPredicted);

		RegressionDataSet regressionTestDataSet = new SimpleDataSet(testCopy.shallowClone().getDataPoints().subList(0, testCopy.getSampleSize())).asRegressionDataSet(columnPredicted);

		MultipleLinearRegression multipleLinearRegression = new MultipleLinearRegression();
		multipleLinearRegression.train(regressionDataSet);

		System.out.println("Weights: " + multipleLinearRegression.getRawWeight());

		int index = 0;
		for (DataPoint dp : regressionTestDataSet.getDataPoints()) {
			DataPoint simple = testCopy.getDataPoint(index);
			index++;
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.regress(dp)).replace(',', '.'));
			System.out.println("(" + simple.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			simple.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
	}

	public void PolynomialCurveFitterApache (int columnPredictor) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(3);

		for (DataPoint dp : trainingCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());
		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length - 1; i++) {
			System.out.print(coeff[i] + ",");
		}
		System.out.println(coeff[coeff.length - 1] + "]");

		for (DataPoint dp : testCopy.getDataPoints()) {
			double newValue = Double.parseDouble(df2.format(polyValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);

	}

	public void GaussianCurveFitterApache (int columnPredictor) throws IOException {
		String method = "GaussianCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		final GaussianCurveFitter fitter = GaussianCurveFitter.create();

		for (DataPoint dp : trainingCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length - 1; i++) {
			System.out.print(coeff[i] + ",");
		}
		System.out.println(coeff[coeff.length - 1] + "]");

		for (DataPoint dp : testCopy.getDataPoints()) {
			double newValue = Double.parseDouble(df2.format(gaussianValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);

	}

	public void LinearInterpolatorApache (int columnPredictor) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println("\n" + ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		LinearInterpolator linearInterpolator = new LinearInterpolator();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		PolynomialSplineFunction polynomialSplineFunction = linearInterpolator.interpolate(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());
		double[] coeff = polynomialSplineFunction.getKnots();

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length - 1; i++) {
			System.out.print(coeff[i] + ",");
		}
		System.out.println(coeff[coeff.length - 1] + "]");


		for (DataPoint dp : testCopy.getDataPoints()) {
			double newValue = Double.parseDouble(df2.format(polynomialSplineFunction.value(dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
	}

	public void PolynomialRegressionJama (int columnPredictor) throws IOException {
		String method = "PolynomialRegressionJama (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		PolynomialRegression polynomialRegression = new PolynomialRegression(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy(), 3);

		System.out.print("Coefficients: [");
		for (int i = 0; i <= polynomialRegression.degree() - 1; i++) {
			System.out.print(polynomialRegression.beta(i) + ",");
		}
		System.out.println(polynomialRegression.beta(polynomialRegression.degree()) + "]");

		for (DataPoint dp : testCopy.getDataPoints()) {
			double newValue = Double.parseDouble(df2.format(polynomialRegression.predict(dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
//			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);

	}

	public void MultipleLinearRegressionJama () throws IOException {
		String method = "MultipleLinearRegressionJama ";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		double[][] regressionTrainingDataSet = DatasetManipulation.toArray(trainingCopy, new int[]{0, 1, 2, 3});
		double[][] regressionTestDataSet = DatasetManipulation.toArray(testCopy, new int[]{0, 1, 2, 3});
		MultipleLinearRegressionJama multipleLinearRegression = new MultipleLinearRegressionJama(regressionTrainingDataSet, trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());

		System.out.print("Coefficients: [");
		for (int i = 0; i < 4; i++) {
			System.out.print(multipleLinearRegression.beta(i) + ",");
		}
		System.out.println(multipleLinearRegression.beta(4) + "]");

		for (int i = 0; i < testCopy.getSampleSize(); i++) {
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.predict(regressionTestDataSet[i])).replace(',', '.'));
			System.out.println("(" + testCopy.getDataPoint(i).getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			testCopy.getDataPoint(i).getNumericalValues().set(columnPredicted, newValue);
		}

		printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);

	}

	public void printPerformanceMeasures (Vec test, Vec predicted, double meanTraining, String method) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt", true));
		writer.append(method);
		writer.append("\n\nPerformance:");
		writer.append("\n\tMean-Squared Error: " + df2.format(MSError(test, predicted)));
		writer.append("\n\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)));
		writer.append("\n\tMean-Absolute Error: " + df2.format(meanAbsoluteError(test, predicted)));
		writer.append("\n\tRelative-Squared Error: " + df2.format(relativeSquaredError(test, predicted, meanTraining)));
		writer.append("\n\tRoot Relative-Squared Error: " + df2.format(rootRelativeSquaredError(test, predicted, meanTraining) * 100) + "%");
		writer.append("\n\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%");
		writer.append("\n\tCorrelation Coefficient: " + df2.format(DescriptiveStatistics.sampleCorCoeff(test, predicted)) + "\n\n");
		writer.close();

		System.out.println("Performance:");
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF + "\n\n");

	}

	public static double polyValue (double[] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double val = p[p.length - 1];

		for (int i = p.length - 2; i >= 0; i--) {
			val = val * x0 + p[i];
		}

		return val;
	}

	public static double gaussianValue (double[] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double val = p[0] * Math.exp(Math.pow((x0 - p[1]) / p[2], 2) / -2);

		return val;
	}


}
