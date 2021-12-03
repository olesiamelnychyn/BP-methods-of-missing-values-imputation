package com.company.utils;

import com.company.utils.objects.ImputedValue;
import com.company.utils.objects.PerformanceMeasures;
import com.company.utils.objects.Statistics;
import com.company.utils.regressions.*;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
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
import java.util.*;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.calculations.MathCalculations.*;
import static com.company.utils.calculations.StatCalculations.*;
import static com.company.utils.objects.PerformanceMeasures.df2;
import static com.company.utils.objects.PerformanceMeasures.meanAbsolutePercentageError;

/**
 * This class perform the main logic of the program - predict missing values
 */
public class ImputationMethods {
	SimpleDataSet datasetComplete;
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, List<ImputedValue>> values = new HashMap<>(); //Map of <column, imputedValue>
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal;

	public ImputationMethods (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing, boolean printOnlyFinal) {
		this.datasetComplete = datasetComplete;
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		this.printOnlyFinal = printOnlyFinal;
	}

	private void calcStatistics (int columnPredicted) {

		if (columnPredicted != -1) {
			Statistics statistic = columnPredictors.length == 1
					? new Statistics(datasetMissing.getNumericColumn(columnPredicted), datasetMissing.getNumericColumn(columnPredictors[0]))
					: new Statistics(datasetMissing, columnPredicted, columnPredictors);
			statistics.put(columnPredicted, statistic);
		} else {
			int n = datasetMissing.getNumericColumns().length;
			int[] arrColumns = new int[n];
			for (int i = 0; i < n; i++) {
				arrColumns[i] = i;
			}
			int[] predicted = getDifference(arrColumns, columnPredictors);
			for (int j : predicted) {
				Statistics statistic = columnPredictors.length == 1
						? new Statistics(datasetMissing.getNumericColumn(j), datasetMissing.getNumericColumn(columnPredictors[0]))
						: new Statistics(datasetMissing, j, columnPredictors);
				statistics.put(j, statistic);
			}
		}
	}

	public void runImputation (int columnPredicted) throws IOException {
		// check if column predicted is not predicted
		if (columnPredicted != -1) {
			for (int i : columnPredictors) {
				if (i == columnPredicted) {
					System.out.println("Predictor cannot be predicted -- exit");
					return;
				}
			}
		}
		calcStatistics(columnPredicted);

		int skipped = 0;
		for (DataPoint dp : datasetMissing.getDataPoints()) { //traverse dataset one by one

			if (columnPredicted != -1) { //if column to be imputed is specified
				if (Double.isNaN(dp.getNumericalValues().get(columnPredicted))) {
					impute(dp, columnPredicted);
				}
			} else { //all columns should be imputed
				int[] indexes = getIndexesOfNull(dp);
				if (getIntersection(indexes, columnPredictors).length != 0) {
					skipped++;
					if (!printOnlyFinal) {
						System.out.println(ANSI_RED_BACKGROUND + "Predictor cannot be predicted -- skip" + ANSI_RESET + "\n\n");
					}
				} else if (indexes.length > 0) {
					for (int idx : indexes) {
						impute(dp, idx);
					}
				}
			}
		}

		if (skipped != 0) {
			System.out.println("Number of records skipped due to absence of predictor : " + skipped);
		}

		evaluateFinal();
	}

	public void impute (DataPoint dp, int columnPredicted) {

		//split dataset into training and the one to be imputed
		ArrayList<SimpleDataSet> datasets = DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(datasetMissing, datasetMissing.getDataPoints().indexOf(dp), columnPredicted, columnPredictors);

		if (columnPredictors.length > 1) { //if it is multiple regression
			if (hasLinearRelationship(datasets.get(0), columnPredicted, columnPredictors, statistics.get(columnPredicted).getThresholds()[2])) {
				MultipleLinearRegressionJama(columnPredicted, datasets);
			} else {
				MultiplePolynomialRegressionJama(columnPredicted, datasets, 2);
			}
		} else { //if it is simple regression (only one predictor)
			if (isCloseToMean(datasets.get(0), columnPredicted, statistics.get(columnPredicted).getThresholds()[0])) {
				MeanImputation(columnPredicted, datasets);
				return;
			} else if (isCloseToMedian(datasets.get(0), columnPredicted, statistics.get(columnPredicted).getThresholds()[1])) {
				MedianImputation(columnPredicted, datasets);
				return;
			} else if (isStrictlyIncreasing(datasets.get(0), columnPredicted) && isStrictlyIncreasing(datasets.get(0), columnPredictors[0])) {
				LinearInterpolatorApache(columnPredicted, columnPredictors[0], datasets, true);
				return;
			} else if (isStrictlyDecreasing(datasets.get(0), columnPredicted) && isStrictlyDecreasing(datasets.get(0), columnPredictors[0])) {
				LinearInterpolatorApache(columnPredicted, columnPredictors[0], datasets, false);
				return;
			} else if (hasLinearRelationship(datasets.get(0), columnPredicted, columnPredictors[0], statistics.get(columnPredicted).getThresholds()[2])) {
				LinearRegressionJSAT(columnPredicted, columnPredictors[0], datasets);
				return;
			} else {
				int order = getPolynomialOrder(datasets.get(0), columnPredicted, columnPredictors[0], statistics.get(columnPredicted).getThresholds()[3]);
				if (order != -1) {
					PolynomialCurveFitterApache(columnPredicted, columnPredictors[0], datasets, order);
					return;
				}
			}
			MeanImputation(columnPredicted, datasets); //default solution if everything fails to meet conditions
		}
	}

	public boolean MeanImputation (int columnPredicted, ArrayList<SimpleDataSet> datasets) {

		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);
		double mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "Mean Imputation" + ANSI_RESET + "\nMean: [" + mean + "]");
		}
		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			dp.getNumericalValues().set(columnPredicted, mean);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean MedianImputation (int columnPredicted, ArrayList<SimpleDataSet> datasets) {

		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);

		double median = trainingCopy.getDataMatrix().getColumn(columnPredicted).median();
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "Median Imputation" + ANSI_RESET + "\nMedian: [" + median + "]");
		}
		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			dp.getNumericalValues().set(columnPredicted, median);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean LinearRegressionJSAT (int columnPredicted, int columnPredictor, ArrayList<SimpleDataSet> datasets) {
		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);

		double[] reg = SimpleLinearRegression.regres(trainingCopy.getDataMatrix().getColumn(columnPredictor), trainingCopy.getDataMatrix().getColumn(columnPredicted));
		if ("NaN".equals(String.valueOf(reg[0])) || "NaN".equals(String.valueOf(reg[1]))) {
			return false;
		}
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "LinearRegression (columnPredictor=" + columnPredictor + ")" + ANSI_RESET + "\na & b: [" + reg[0] + "," + reg[1] + "]");
		}

		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(reg[0] + reg[1] * dp.getNumericalValues().get(columnPredictor)).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean MultipleLinearRegressionJSAT (int columnPredicted, ArrayList<SimpleDataSet> datasets_complete) {
		SimpleDataSet trainingCopy_complete = datasets_complete.get(0);
		SimpleDataSet toBePredicted = datasets_complete.get(1);
		SimpleDataSet trainingCopy = DatasetManipulation.excludeNonPredictors(trainingCopy_complete, columnPredictors, columnPredicted);
		SimpleDataSet toBePredictedDataSet = DatasetManipulation.excludeNonPredictors(toBePredicted, columnPredictors, columnPredicted);

		RegressionDataSet regressionDataSet = new SimpleDataSet(trainingCopy.shallowClone().getDataPoints().subList(0, trainingCopy.getSampleSize())).asRegressionDataSet(0);
		RegressionDataSet regressionTestDataSet = new SimpleDataSet(toBePredictedDataSet.shallowClone().getDataPoints().subList(0, toBePredictedDataSet.getSampleSize())).asRegressionDataSet(0);

		MultipleLinearRegression multipleLinearRegression = new MultipleLinearRegression();
		multipleLinearRegression.train(regressionDataSet);

		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "MultipleLinearRegressionJSAT" + ANSI_RESET);
			System.out.println("Weights: " + multipleLinearRegression.getRawWeight());
		}

		int i = 0;
		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.regress(regressionTestDataSet.getDataPoint(i++))).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	private boolean PolynomialCurveFitterApache (int columnPredicted, int columnPredictor, ArrayList<SimpleDataSet> datasets, int order) {
		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);

		final WeightedObservedPoints obs = new WeightedObservedPoints();
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(order);

		for (DataPoint dp : trainingCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
			System.out.print("Coefficients: [");
			for (int i = 0; i < coeff.length - 1; i++) {
				System.out.print(coeff[i] + ",");
			}
			System.out.println(coeff[coeff.length - 1] + "]");
		}

		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(polyValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean GaussianCurveFitterApache (int columnPredicted, int columnPredictor, ArrayList<SimpleDataSet> datasets) {
		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);

		final WeightedObservedPoints obs = new WeightedObservedPoints();
		final GaussianCurveFitter fitter = GaussianCurveFitter.create();

		for (DataPoint dp : trainingCopy.getDataPoints()) {
			obs.add(dp.getNumericalValues().get(columnPredictor), dp.getNumericalValues().get(columnPredicted));
		}

		final double[] coeff = fitter.fit(obs.toList());
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "GaussianCurveFitter (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
			System.out.print("Coefficients: [");
			for (int i = 0; i < coeff.length - 1; i++) {
				System.out.print(coeff[i] + ",");
			}
			System.out.println(coeff[coeff.length - 1] + "]");
		}

		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(gaussianValue(coeff, dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean LinearInterpolatorApache (int columnPredicted, int columnPredictor, ArrayList<SimpleDataSet> datasets, boolean increasing) {
		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);

		LinearInterpolator linearInterpolator = new LinearInterpolator();

		// if values are decreasing than reverse dataset
		if (!increasing) {
			trainingCopy = DatasetManipulation.reverseDataset(trainingCopy);
		}

		PolynomialSplineFunction polynomialSplineFunction = linearInterpolator.interpolate(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy());

		if (!printOnlyFinal) {
			System.out.println("\n" + ANSI_PURPLE_BACKGROUND + "PolynomialCurveFitter (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
			System.out.println("\n\nPiecewise functions:");
			Arrays.stream(polynomialSplineFunction.getPolynomials()).forEach(System.out::println);
//			double[] knots = polynomialSplineFunction.getKnots();
//			System.out.print("Knots: [");
//			for (int i = 0; i < knots.length - 1; i++) {
//				System.out.print(knots[i] + ",");
//			}
//			System.out.println(knots[knots.length - 1] + "]");
		}

		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(polynomialSplineFunction.value(dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	public boolean PolynomialRegressionJama (int columnPredicted, int columnPredictor, ArrayList<SimpleDataSet> datasets) {
		SimpleDataSet trainingCopy = datasets.get(0);
		SimpleDataSet toBePredicted = datasets.get(1);
		PolynomialRegression polynomialRegression = new PolynomialRegression(trainingCopy.getDataMatrix().getColumn(columnPredictor).arrayCopy(), trainingCopy.getDataMatrix().getColumn(columnPredicted).arrayCopy(), 2);

		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + "PolynomialRegressionJama (columnPredictor=" + columnPredictor + ")" + ANSI_RESET);
			System.out.print("Coefficients: [");
			for (int i = 0; i <= polynomialRegression.degree() - 1; i++) {
				System.out.print(polynomialRegression.beta(i) + ",");
			}
			System.out.println(polynomialRegression.beta(polynomialRegression.degree()) + "]");
		}

		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(polynomialRegression.predict(dp.getNumericalValues().get(columnPredictor))).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		if (!printOnlyFinal) {
			System.out.println("\n");
		}
		return true;
	}

	private boolean MultipleRegressionJama (int columnPredicted, ArrayList<SimpleDataSet> datasets_complete, boolean polynomial, int degree) {
		SimpleDataSet trainingCopy_complete = datasets_complete.get(0);
		SimpleDataSet toBePredicted = datasets_complete.get(1);
		if (!printOnlyFinal) {
			System.out.println(ANSI_PURPLE_BACKGROUND + (polynomial ? "MultiplePolynomialRegressionJama" : "MultipleLinearRegressionJama ") + ANSI_RESET);
		}
		SimpleDataSet trainingCopy = DatasetManipulation.excludeNonPredictors(trainingCopy_complete, columnPredictors, columnPredicted);
		SimpleDataSet toBePredictedDataSet = DatasetManipulation.excludeNonPredictors(toBePredicted, columnPredictors, columnPredicted);

		int[] predictors = Arrays.copyOf(columnPredictors, columnPredictors.length);

		// if it is polynomial regression change dataset by adding power columns
		if (polynomial) {
			if (!printOnlyFinal) {
				System.out.println("Polynomial degree: " + degree);
			}
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

		if (!printOnlyFinal) {
			System.out.print("Coefficients: [");
			for (int i = 0; i < columnPredictors.length - 1; i++) {
				System.out.print(multipleLinearRegression.beta(i) + ",");
			}
			System.out.println(multipleLinearRegression.beta(columnPredictors.length) + "]");
		}

		int i = 0;
		for (DataPoint dp : toBePredicted.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = Double.parseDouble(df2.format(multipleLinearRegression.predict(regressionTestDataSet[i++])).replace(',', '.'));
			dp.getNumericalValues().set(columnPredicted, newValue);
			evaluate_concat(columnPredicted, indexMissing, dp);
		}
		System.out.println("\n");
		return true;
	}

	public boolean MultipleLinearRegressionJama (int columnPredicted, ArrayList<SimpleDataSet> datasets) {
		return MultipleRegressionJama(columnPredicted, datasets, false, 0);
	}

	public boolean MultiplePolynomialRegressionJama (int columnPredicted, ArrayList<SimpleDataSet> datasets, int degree) {
		return MultipleRegressionJama(columnPredicted, datasets, true, degree);
	}

	private void evaluate_concat (int columnPredicted, int indexMissing, DataPoint toBePredicted) {
		//if there is no complete dataset, there won't be any evaluation
		if (datasetComplete != null) {
			ImputedValue value = new ImputedValue(indexMissing, datasetComplete.getDataPoint(indexMissing).getNumericalValues().get(columnPredicted), toBePredicted.getNumericalValues().get(columnPredicted));
			if (values.containsKey(columnPredicted)) {
				values.get(columnPredicted).add(value);
			} else {
				List<ImputedValue> list = new ArrayList<>();
				list.add(value);
				values.put(columnPredicted, list);
			}
			if (!printOnlyFinal) {
				Vec test = new DenseVector(new double[]{value.actual});
				Vec predicted = new DenseVector(new double[]{value.predicted});
				System.out.println(value.index + "\t" + value.actual + " --- " + value.predicted);
				System.out.println("\t" + indexMissing + "\t" + columnPredicted + "\t" + ANSI_BOLD_ON + ANSI_PURPLE + "Mean Absolute Percentage Error: " + df2.format(meanAbsolutePercentageError(test, predicted)) + "%" + ANSI_RESET + ANSI_BOLD_OFF);
			}
		}
	}

	/**Final evaluation
	 * @throws IOException
	 *
	 * Performs evaluation of all predicted values
	 */
	private void evaluateFinal () throws IOException {
		System.out.println("\n" + ANSI_PURPLE_BACKGROUND + "Results" + ANSI_RESET + "\n");
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write("Results:\n");
		writer.close();

		// separate evaluation by columns
		for (Map.Entry<Integer, List<ImputedValue>> entry : values.entrySet()) {
			List<ImputedValue> list = entry.getValue();
			int columnPredicted = entry.getKey();
			Vec act = new DenseVector(list.size());
			Vec pred = new DenseVector(list.size());
			int i = 0;
			for (ImputedValue value : list) {
				if (!printOnlyFinal) {
					System.out.println(value);
				}
				act.set(i, value.actual);
				pred.set(i++, value.predicted);
			}
			PerformanceMeasures performanceMeasures = new PerformanceMeasures(act, pred, datasetComplete.getDataMatrix().getColumn(columnPredicted).mean());
			performanceMeasures.printAndWriteResults(columnPredicted);
			System.out.println(statistics.get(columnPredicted));
		}
	}

}
