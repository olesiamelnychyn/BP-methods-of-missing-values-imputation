package com.company.utils;

import com.company.utils.regressions.*;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.PerformanceMeasures.*;
import static com.company.utils.PerformanceMeasures.meanAbsolutePercentageError;

public class ImputationMethods {
	int columnPredicted;
	int[] columnPredictors;
	SimpleDataSet datasetComplete;
	SimpleDataSet datasetMissing;
	ArrayList<DataPoint> listPredicted = new ArrayList<>();
	ArrayList<DataPoint> listActual = new ArrayList<>();

	SimpleDataSet test;
	SimpleDataSet training;
	DecimalFormat df2 = new DecimalFormat("#.##");

	public ImputationMethods (int columnPredicted, int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing) {
		this.columnPredicted = columnPredicted;
		this.datasetComplete = datasetComplete;
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		this.test = DatasetManipulation.createDeepCopy(datasetComplete, 5, 7);
//		SimpleDataSet datasetCopy = DatasetManipulation.createDeepCopy(datasetComplete, 0, 30);
//		List<DataPoint> training = datasetCopy.getDataPoints();
//		for (int i = 5; i < 7; i++) {
//			training.remove(datasetCopy.getDataPoint(i));
//		}
//		this.training = new SimpleDataSet(training);
	}

	public void impute () throws IOException {
		for (DataPoint dp : datasetMissing.getDataPoints()) {

			if (dp.getNumericalValues().countNaNs() > 0) {
				int index = datasetMissing.getDataPoints().indexOf(dp);
				//the main body - decision tree

				if (isLowStandardDeviation(columnPredicted, index, 4, 4)) {
					MeanImputation(index, 4, 4);
				} else {
					for (int columnPredictor : columnPredictors) {
						if (!LinearRegressionJSAT(columnPredictor, index, 4, 4)) {
							PolynomialCurveFitterApache(columnPredictor, index, 4, 4);
						}
					}
				}
			}
		}

		evaluateFinal();
	}

	private boolean isLowStandardDeviation (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter) {
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		double std = trainingCopy.getDataMatrix().getColumn(columnPredicted).standardDeviation();
		double mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
		System.out.println(std / mean);
		System.out.println(mean);
		System.out.println(datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted));
		if (std / mean <= 0.3) {
			return true;
		} else {
			return false;
		}
	}

	public void MeanImputation (int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "MeanImputation";
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		double mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET + "\nMean: [" + mean + "]");
		toBePredicted.getNumericalValues().set(columnPredicted, mean);

		if (datasetComplete != null) {
			System.out.println(datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted) + " " + toBePredicted.getNumericalValues().get(columnPredicted) + " " + trainingCopy.getDataMatrix().getColumn(columnPredicted).mean());
			printPerformanceMeasures(datasetComplete.getDataPoint(indexMissing), toBePredicted, trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}

		listPredicted.add(toBePredicted);
		listActual.add(datasetComplete.getDataPoint(indexMissing));

	}

	public boolean LinearRegressionJSAT (int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "LinearRegression (columnPredictor=" + columnPredictor + ")";
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		double[] reg = SimpleLinearRegression.regres(trainingCopy.getDataMatrix().getColumn(columnPredictor), trainingCopy.getDataMatrix().getColumn(columnPredicted));
		if (String.valueOf(reg[0]) == "NaN" || String.valueOf(reg[1]) == "NaN") {
			return false;
		}
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET + "\na & b: [" + reg[0] + "," + reg[1] + "]");

		double newValue = Double.parseDouble(df2.format(reg[0] + reg[1] * toBePredicted.getNumericalValues().get(columnPredictor)).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		if (datasetComplete != null) {
			System.out.println(datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted) + " " + toBePredicted.getNumericalValues().get(columnPredicted) + " " + trainingCopy.getDataMatrix().getColumn(columnPredicted).mean());
			if (printPerformanceMeasures(datasetComplete.getDataPoint(indexMissing), toBePredicted, trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method) > 15) {
				return false;
			}
		}

		listPredicted.add(toBePredicted);
		listActual.add(datasetComplete.getDataPoint(indexMissing));
		return true;
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
//			System.out.println("(" + simple.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			simple.getNumericalValues().set(columnPredicted, newValue);
		}

		if (datasetComplete != null) {
			printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}
	}

	private boolean PolynomialCurveFitterApache (int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);
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

		double newValue = Double.parseDouble(df2.format(polyValue(coeff, toBePredicted.getNumericalValues().get(columnPredictor))).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		if (datasetComplete != null) {
			System.out.println(datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted) + " " + toBePredicted.getNumericalValues().get(columnPredicted) + " " + trainingCopy.getDataMatrix().getColumn(columnPredicted).mean());
			if (printPerformanceMeasures(datasetComplete.getDataPoint(indexMissing), toBePredicted, trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method) > 15) {
				return false;
			}
		}

		listPredicted.add(toBePredicted);
		listActual.add(datasetComplete.getDataPoint(indexMissing));
		return true;
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
//			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		if (datasetComplete != null) {
			printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}

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
//			System.out.println("(" + dp.getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			dp.getNumericalValues().set(columnPredicted, newValue);
		}

		if (datasetComplete != null) {
			printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}
	}

	public void PolynomialRegressionJama (int columnPredictor) throws IOException {
		String method = "PolynomialRegressionJama (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());
		PolynomialRegression polynomialRegression = new PolynomialRegression(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy(), 2);

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

		if (datasetComplete != null) {
			printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}

	}

	private void MultipleRegressionJama (int[] predictors, boolean polynomial, int degree) throws IOException {
		String method = "MultipleLinearRegressionJama ";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		if (predictors.length <= 1) {
			System.out.println(ANSI_RED + "Number of predictors is too small (0 or 1) for " + method + ANSI_RESET + "\n\n");
			return;
		}
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(training, 0, training.getSampleSize());
		SimpleDataSet testCopy = DatasetManipulation.createDeepCopy(test, 0, test.getSampleSize());

		if (polynomial) {
			predictors = Arrays.copyOf(predictors, predictors.length * degree);
			for (int i = predictors.length / degree; i < predictors.length; i++) {
				predictors[i] = i + 1;
			}
			trainingCopy = addPowerColumns(trainingCopy, degree, predictors);
			testCopy = addPowerColumns(testCopy, degree, predictors);
		}

		double[][] regressionTrainingDataSet = DatasetManipulation.toArray(trainingCopy, predictors);
		double[][] regressionTestDataSet = DatasetManipulation.toArray(testCopy, predictors);
		MultipleLinearRegressionJama multipleLinearRegression = new MultipleLinearRegressionJama(regressionTrainingDataSet, trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());

		System.out.print("Coefficients: [");
		for (int i = 0; i < predictors.length - 1; i++) {
			System.out.print(multipleLinearRegression.beta(i) + ",");
		}
		System.out.println(multipleLinearRegression.beta(predictors.length) + "]");

		for (int i = 0; i < testCopy.getSampleSize(); i++) {
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.predict(regressionTestDataSet[i])).replace(',', '.'));
//			System.out.println("(" + testCopy.getDataPoint(i).getNumericalValues().get(columnPredicted) + "," + newValue + ")");
			testCopy.getDataPoint(i).getNumericalValues().set(columnPredicted, newValue);
		}

		if (datasetComplete != null) {
			printPerformanceMeasures(test.getDataMatrix().getColumn(columnPredicted), testCopy.getDataMatrix().getColumn(columnPredicted), trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method);
		}

	}

	public void MultipleLinearRegressionJama (int[] predictors) throws IOException {
		MultipleRegressionJama(predictors, false, 0);
	}

	public void MultiplePolynomialRegressionJama (int[] predictors, int degree) throws IOException {
		MultipleRegressionJama(predictors, true, degree);
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
		writer.append("\n\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted) * 100) + "%");
		writer.close();

		System.out.println("Performance:");
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF + "\n\n");

	}

	public double printPerformanceMeasures (DataPoint act, DataPoint pred, double meanTraining, String method) throws IOException {

		double[] act1 = new double[1];
		act1[0] = act.getNumericalValues().get(columnPredicted);
		Vec test = new DenseVector(act1);
		double[] pred1 = new double[1];
		pred1[0] = pred.getNumericalValues().get(columnPredicted);
		Vec predicted = new DenseVector(pred1);

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
		writer.append("\n\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted) * 100) + "%");
		writer.close();

		System.out.println("Performance:");
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF + "\n\n");

		return meanAbsolutePercentageError(test, predicted);
	}

	public void evaluateFinal () throws IOException {
		System.out.println(ANSI_PURPLE_BACKGROUND + "All together" + ANSI_RESET);
		SimpleDataSet act = new SimpleDataSet(listActual);
		SimpleDataSet pred = new SimpleDataSet(listPredicted);
		printPerformanceMeasures(act.getDataMatrix().getColumn(columnPredicted), pred.getDataMatrix().getColumn(columnPredicted), act.getDataMatrix().getColumn(columnPredicted).mean(), "All");
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

	private SimpleDataSet addPowerColumns (SimpleDataSet dataset, int degree, int[] predictors) {
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
