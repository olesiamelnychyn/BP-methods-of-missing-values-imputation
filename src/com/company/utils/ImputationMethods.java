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
import java.util.Arrays;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.PerformanceMeasures.*;
import static com.company.utils.PerformanceMeasures.meanAbsolutePercentageError;

public class ImputationMethods {
	int[] columnPredictors;
	SimpleDataSet datasetComplete;
	SimpleDataSet datasetMissing;
	Vec valuesPredicted;
	Vec valuesActual;
	int valuesImputed = 0;
	static DecimalFormat df2 = new DecimalFormat("#.##");

	public ImputationMethods (int columnPredicted, int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing) {
		this.datasetComplete = datasetComplete;
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		valuesPredicted = new DenseVector(datasetComplete.getSampleSize());
		valuesActual = new DenseVector(datasetComplete.getSampleSize());
	}

	public void runImputation (int columnPredicted) throws IOException {
		for (DataPoint dp : datasetMissing.getDataPoints()) {

			if (columnPredicted != -1) {
				for (int i : columnPredictors) {
					if (i == columnPredicted) {
						System.out.println("Predictor cannot be predicted -- exit");
						return;
					}
				}
				if (Double.isNaN(dp.getNumericalValues().get(columnPredicted))) {
					impute(dp, columnPredicted);
				}
			} else {
				int[] indexes = DatasetManipulation.getIndexesOfNull(dp);
				if (getIntersection(indexes).length != 0) {
					System.out.println(ANSI_RED_BACKGROUND + "Predictor cannot be predicted -- skip" + ANSI_RESET + "\n");
					continue;
				} else if (indexes.length > 0) {
					for (int idx : indexes) {
						impute(dp, idx);
					}
				}
			}
		}
		evaluateFinal();
	}

	public void impute (DataPoint dp, int columnPredicted) throws IOException {

		int index = datasetMissing.getDataPoints().indexOf(dp);

		if (columnPredictors.length > 1) {
			if (DatasetManipulation.hasLinearRelationship(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted, columnPredictors)) {
				MultipleLinearRegressionJama(columnPredicted, index, 4, 4);
			} else {
//				if (!MultiplePolynomialRegressionJama(index, 6, 6, 4)) {
//					if (!MultiplePolynomialRegressionJama(index, 6, 6, 3)) {
				MultiplePolynomialRegressionJama(columnPredicted, index, 4, 4, 2);
//					}
//				}
			}
		} else {
			if (DatasetManipulation.isCloseToMean(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted)) {
				MeanImputation(columnPredicted, index, 4, 4);
			} else if (DatasetManipulation.isCloseToMedian(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted)) {
				MedianImputation(columnPredicted, index, 4, 4);
			} else if (DatasetManipulation.isStrictlyIncreasing(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted)) {
				LinearInterpolatorApache(columnPredicted, columnPredictors[0], index, 2, 2, true);
			} else if (DatasetManipulation.isStrictlyDecreasing(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted)) {
				LinearInterpolatorApache(columnPredicted, columnPredictors[0], index, 2, 2, false);
			} else if (DatasetManipulation.hasLinearRelationship(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted, columnPredictors[0])) {
				LinearRegressionJSAT(columnPredicted, columnPredictors[0], index, 4, 4);
			} else {
				int order = DatasetManipulation.getPolynomialOrder(DatasetManipulation.createDeepCopy(datasetMissing, index - 4, index, index + 1, index + 1 + 4), columnPredicted, columnPredictors[0]);
				if (order != -1) {
					PolynomialCurveFitterApache(columnPredicted, columnPredictors[0], index, 4, 4, order);
				} else {
					GaussianCurveFitterApache(columnPredicted, columnPredictors[0], index, 4, 4);
				}
			}
		}
	}

	public boolean MeanImputation (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {

		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		double mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
		String method = "Mean Imputation";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET + "\nMean: [" + mean + "]");
		toBePredicted.getNumericalValues().set(columnPredicted, mean);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean MedianImputation (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {

		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		double median = trainingCopy.getDataMatrix().getColumn(columnPredicted).median();
		String method = "Median Imputation";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET + "\nMedian: [" + median + "]");
		toBePredicted.getNumericalValues().set(columnPredicted, median);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean LinearRegressionJSAT (int columnPredicted, int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
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

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean MultipleLinearRegressionJSAT (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "MultipleLinearRegressionJSAT";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy_complete = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		SimpleDataSet trainingCopy = DatasetManipulation.excludeNonPredictors(trainingCopy_complete, columnPredictors, columnPredicted);
		SimpleDataSet toBePredictedDataSet = DatasetManipulation.excludeNonPredictors(DatasetManipulation.createDeepCopy(datasetMissing, indexMissing, indexMissing + 1), columnPredictors, columnPredicted);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		RegressionDataSet regressionDataSet = new SimpleDataSet(trainingCopy.shallowClone().getDataPoints().subList(0, trainingCopy.getSampleSize())).asRegressionDataSet(0);
		RegressionDataSet regressionTestDataSet = new SimpleDataSet(toBePredictedDataSet.shallowClone().getDataPoints().subList(0, toBePredictedDataSet.getSampleSize())).asRegressionDataSet(0);

		MultipleLinearRegression multipleLinearRegression = new MultipleLinearRegression();
		multipleLinearRegression.train(regressionDataSet);

		System.out.println("Weights: " + multipleLinearRegression.getRawWeight());

		double newValue = Double.parseDouble(df2.format(multipleLinearRegression.regress(regressionTestDataSet.getDataPoint(0))).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy_complete, method);
	}

	private boolean PolynomialCurveFitterApache (int columnPredicted, int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter, int order) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(order);

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

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean GaussianCurveFitterApache (int columnPredicted, int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "GaussianCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);
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

		double newValue = Double.parseDouble(df2.format(gaussianValue(coeff, toBePredicted.getNumericalValues().get(columnPredictor))).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean LinearInterpolatorApache (int columnPredicted, int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter, boolean increasing) throws IOException {
		String method = "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")";
		System.out.println("\n" + ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		LinearInterpolator linearInterpolator = new LinearInterpolator();
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		if (!increasing) {
			trainingCopy = DatasetManipulation.reverseDataset(trainingCopy);
		}

		PolynomialSplineFunction polynomialSplineFunction = linearInterpolator.interpolate(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());
		double[] coeff = polynomialSplineFunction.getKnots();

		System.out.print("Coefficients: [");
		for (int i = 0; i < coeff.length - 1; i++) {
			System.out.print(coeff[i] + ",");
		}
		System.out.println(coeff[coeff.length - 1] + "]");

		double newValue = Double.parseDouble(df2.format(polynomialSplineFunction.value(toBePredicted.getNumericalValues().get(columnPredictor))).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	public boolean PolynomialRegressionJama (int columnPredicted, int columnPredictor, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		String method = "PolynomialRegressionJama (columnPredictor=" + columnPredictor + ")";
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);
		PolynomialRegression polynomialRegression = new PolynomialRegression(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy(), 2);

		System.out.print("Coefficients: [");
		for (int i = 0; i <= polynomialRegression.degree() - 1; i++) {
			System.out.print(polynomialRegression.beta(i) + ",");
		}
		System.out.println(polynomialRegression.beta(polynomialRegression.degree()) + "]");


		double newValue = Double.parseDouble(df2.format(polynomialRegression.predict(toBePredicted.getNumericalValues().get(columnPredictor))).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy, method);
	}

	private boolean MultipleRegressionJama (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter, boolean polynomial, int degree) throws IOException {
		String method = "MultipleLinearRegressionJama ";
		if (polynomial) {
			method = "MultiplePolynomialRegressionJama";
		}
		System.out.println(ANSI_PURPLE_BACKGROUND + method + ANSI_RESET);
		SimpleDataSet trainingCopy_complete = DatasetManipulation.createDeepCopy(datasetMissing, indexMissing - recordsBefore, indexMissing, indexMissing + 1, indexMissing + 1 + recordsAfter);
		SimpleDataSet trainingCopy = DatasetManipulation.excludeNonPredictors(trainingCopy_complete, columnPredictors, columnPredicted);
		SimpleDataSet toBePredictedDataSet = DatasetManipulation.excludeNonPredictors(DatasetManipulation.createDeepCopy(datasetMissing, indexMissing, indexMissing + 1), columnPredictors, columnPredicted);
		DataPoint toBePredicted = datasetMissing.getDataPoint(indexMissing);

		int[] predictors = Arrays.copyOf(columnPredictors, columnPredictors.length);

		if (polynomial) {
			System.out.println("Polynomial degree: " + degree);
			for (int i = 0; i < predictors.length; i++) {
				predictors[i] = i + 1;
			}
			predictors = Arrays.copyOf(predictors, predictors.length * degree);
			for (int i = predictors.length / degree; i < predictors.length; i++) {
				predictors[i] = i + 1;
			}
			trainingCopy = DatasetManipulation.addPowerColumns(trainingCopy, degree, predictors, 0);
			toBePredictedDataSet = DatasetManipulation.addPowerColumns(toBePredictedDataSet, degree, predictors, 0);
		}

		double[][] regressionTrainingDataSet = DatasetManipulation.toArray(trainingCopy, predictors);
		double[][] regressionTestDataSet = DatasetManipulation.toArray(toBePredictedDataSet, predictors);
		MultipleLinearRegressionJama multipleLinearRegression = new MultipleLinearRegressionJama(regressionTrainingDataSet, trainingCopy.getDataMatrix().getColumn(0).arrayCopy());

		System.out.print("Coefficients: [");
		for (int i = 0; i < columnPredictors.length - 1; i++) {
			System.out.print(multipleLinearRegression.beta(i) + ",");
		}
		System.out.println(multipleLinearRegression.beta(columnPredictors.length) + "]");

		double newValue = Double.parseDouble(df2.format(multipleLinearRegression.predict(regressionTestDataSet[0])).replace(',', '.'));
		toBePredicted.getNumericalValues().set(columnPredicted, newValue);

		return evaluate_concat(columnPredicted, indexMissing, toBePredicted, trainingCopy_complete, method);
	}

	public boolean MultipleLinearRegressionJama (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter) throws IOException {
		return MultipleRegressionJama(columnPredicted, indexMissing, recordsBefore, recordsAfter, false, 0);
	}

	public boolean MultiplePolynomialRegressionJama (int columnPredicted, int indexMissing, int recordsBefore, int recordsAfter, int degree) throws IOException {
		return MultipleRegressionJama(columnPredicted, indexMissing, recordsBefore, recordsAfter, true, degree);
	}

	private void printPerformanceMeasures (Vec test, Vec predicted, double meanTraining, String method) throws IOException {
		writeOutput(test, predicted, meanTraining, method, true);

		System.out.println("Performance:");
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean-Squared Error: " + df2.format(MSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean-Absolute Error: " + df2.format(meanAbsoluteError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Squared Error: " + df2.format(relativeSquaredError(test, predicted, meanTraining)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Relative-Squared Error: " + df2.format(rootRelativeSquaredError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tCorrelation Coefficient: " + df2.format(DescriptiveStatistics.sampleCorCoeff(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF + "\n\n");

	}

	private double printPerformanceMeasures (int columnPredicted, DataPoint act, DataPoint pred, double meanTraining, String method) throws IOException {

		double[] act1 = new double[1];
		act1[0] = act.getNumericalValues().get(columnPredicted);
		Vec test = new DenseVector(act1);
		double[] pred1 = new double[1];
		pred1[0] = pred.getNumericalValues().get(columnPredicted);
		Vec predicted = new DenseVector(pred1);
		writeOutput(test, predicted, meanTraining, method, false);

//		System.out.println("Performance:");
//		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)) + ANSI_RESET + ANSI_BOLD_OFF);
//		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
		System.out.println(ANSI_BOLD_ON + ANSI_PURPLE + "\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF + "\n\n");

		return meanAbsolutePercentageError(test, predicted);
	}

	static private void writeOutput (Vec test, Vec predicted, double meanTraining, String method, boolean all) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt", true));
		writer.append("\n\n" + method);
		writer.append("\nPerformance:");
		writer.append("\n\tRoot Mean-Squared Error: " + df2.format(RMSError(test, predicted)));
		writer.append("\n\tRelative-Absolute Error: " + df2.format(relativeAbsoluteError(test, predicted, meanTraining) * 100) + "%");
		if (all) {
			writer.append("\n\tMean-Squared Error: " + df2.format(MSError(test, predicted)));
			writer.append("\n\tMean-Absolute Error: " + df2.format(meanAbsoluteError(test, predicted)));
			writer.append("\n\tRelative-Squared Error: " + df2.format(relativeSquaredError(test, predicted, meanTraining)));
			writer.append("\n\tRoot Relative-Squared Error: " + df2.format(rootRelativeSquaredError(test, predicted, meanTraining) * 100) + "%");
			writer.append("\n\tCorrelation Coefficient: " + df2.format(DescriptiveStatistics.sampleCorCoeff(test, predicted)));
		}
		writer.append("\n\tMean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + "\n");
		writer.close();
	}

	private boolean evaluate_concat (int columnPredicted, int indexMissing, DataPoint toBePredicted, SimpleDataSet trainingCopy, String method) throws IOException {
		if (datasetComplete != null) {
			valuesActual.set(valuesImputed, datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted));
			valuesPredicted.set(valuesImputed++, toBePredicted.getNumericalValues().get(columnPredicted));

			if (printPerformanceMeasures(columnPredicted, datasetComplete.getDataPoint(indexMissing), toBePredicted, trainingCopy.getDataMatrix().getColumn(columnPredicted).mean(), method) > 15) {
				return false;
			}
		}
		return true;
	}

	private void evaluateFinal () throws IOException {
		System.out.println(ANSI_PURPLE_BACKGROUND + "All together" + ANSI_RESET);
		Vec act = new DenseVector(valuesImputed);
		Vec pred = new DenseVector(valuesImputed);
		for (int i = 0; i < valuesImputed; i++) {
			double actual = valuesActual.get(i);
			double predicted = valuesPredicted.get(i);
			System.out.println(actual + " --- " + predicted);
			act.set(i, actual);
			pred.set(i, predicted);
		}
		printPerformanceMeasures(act, pred, act.mean(), "All");
	}

	private static double polyValue (double[] p, double x0) {

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

	private static double gaussianValue (double[] p, double x0) {

		if (p == null) {
			return Double.NaN;
		}
		if (p.length < 1) {
			return 0.0;
		}

		double val = p[0] * Math.exp(Math.pow((x0 - p[1]) / p[2], 2) / -2);
		return val;
	}

	private int[] getIntersection (int[] indexes) {
		return Arrays.stream(indexes)
				.distinct()
				.filter(x -> Arrays.stream(columnPredictors).anyMatch(y -> y == x))
				.toArray();
	}
}
