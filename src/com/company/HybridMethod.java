package com.company;

import com.company.imputationMethods.*;
import com.company.utils.DatasetManipulation;
import com.company.utils.Evaluation;
import com.company.utils.calculations.StatCalculations;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import com.sun.media.sound.InvalidDataException;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.calculations.MathCalculations.*;
import static com.company.utils.calculations.StatCalculations.*;

/**
 * This class perform the main logic of the program - predict missing values
 */
public class HybridMethod {
	ConfigManager configManager = ConfigManager.getInstance();
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal = Boolean.parseBoolean(configManager.get("input.printOnlyFinal"));
	boolean test = Boolean.parseBoolean(configManager.get("input.runTest"));
	Evaluation evaluation = null;
	MultipleImputationMethods multipleImputationMethods;
	SimpleImputationMethods simpleImputationMethods;

	public HybridMethod (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing) {
		if (datasetComplete != null) {
			evaluation = new Evaluation(datasetComplete, datasetMissing, printOnlyFinal);
		}
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
	}

	public void runImputation (int columnPredicted) throws IOException {
		long start = System.currentTimeMillis();

		// check if column predicted is not predicted
		if (columnPredicted != -1) {
			for (int i : columnPredictors) {
				if (i == columnPredicted) {
					System.out.println("Predictor cannot be predicted -- exit");
					return;
				}
			}
		}
		//create Statistics of column/-s
		statistics = StatCalculations.calcStatistics(columnPredicted, columnPredictors, datasetMissing);

		simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		multipleImputationMethods = new MultipleImputationMethods(datasetMissing, Boolean.parseBoolean(configManager.get("impute.weighted")), Boolean.parseBoolean(configManager.get("impute.useMultiDimensionalAsDefault")));

		for (DataPoint dp : datasetMissing.getDataPoints()) { //traverse dataset one by one
			int[] indexes = getIndexesOfNull(dp);
			boolean missingPredictor = getIntersection(indexes, columnPredictors).length > 0;
			int[] missingNonPredictors = getDifference(indexes, columnPredictors);

			// add "&& !missingPredictor" to the following condition if missing values in records with missing predictors should not be imputed
			// should be used for testing
			if (missingNonPredictors.length != 0) {
				if (columnPredicted != -1 && IntStream.of(missingNonPredictors).anyMatch(x -> x == columnPredicted)) { //if column to be imputed is specified
					impute(dp, columnPredicted, missingPredictor);
				} else if (columnPredicted == -1) { //all columns should be imputed
					for (int idx : missingNonPredictors) {
						impute(dp, idx, missingPredictor);
					}
				}
			}
		}

		long finish = System.currentTimeMillis();
		if (evaluation != null) {
			evaluation.evaluateFinal(statistics);
		}

		System.out.println("\nTime spent on predicting: " + ANSI_RED_BACKGROUND + (finish - start) + ANSI_RESET + "\n");
	}

	public void impute (DataPoint dp, int columnPredicted, boolean missingPredictor) throws InvalidDataException {

		if (!test) {
			predict(getHybridMethod(dp, columnPredicted, missingPredictor), true);
		} else {
			predict(getTestMethod(dp, columnPredicted, missingPredictor), false);
		}

	}

	protected void predict (ImputationMethod imputationMethod, boolean retryAllowed) {
		MainData data = imputationMethod.getData();
		imputationMethod.preprocessData();
		imputationMethod.fit();
		if (!printOnlyFinal) {
			imputationMethod.print();
		}

		int columnPredicted = imputationMethod.getColumnPredicted();
		SimpleDataSet dataPoints = imputationMethod.getToBePredicted();
		for (DataPoint dp : dataPoints.getDataPoints()) {
			double newValue = imputationMethod.predict(dp);

			if (retryAllowed &&
				(Double.isNaN(newValue) || (!data.isMultiple() && !StatCalculations.isWithinInterval(newValue, statistics.get(columnPredicted).getMinAndMax())))) {

				if (!(imputationMethod instanceof MultiplePolynomialRegressionJSATMethod) && data.isMultiple() && data.getColumnPredictors().length == 1) {
					predict(new MultiplePolynomialRegressionJSATMethod(data, 2), true);
				} else {
					predict(getMedianDevPercent(data) > getMeanDevPercent(data)
							? new MeanImputationMethod(data)
							: new MedianImputationMethod(data)
						, false);
				}
				continue;
			}

			if (evaluation != null) {
				evaluation.evaluate_concat(columnPredicted, dp, newValue);
			}
		}

		if (!printOnlyFinal) {
			System.out.println("\n");
		}
	}

	private ImputationMethod getHybridMethod (DataPoint dp, int columnPredicted, boolean missingPredictor) {
		MainData data = new MainData(!missingPredictor ? columnPredictors.clone() : new int[]{}, columnPredicted, dp, columnPredictors.length > 1);

		if (data.getColumnPredictors().length > 1) { //if it is multiple regression
			return multipleImputationMethods.imputeMultiple(data, statistics.get(columnPredicted));
		}
		return simpleImputationMethods.imputeSimple(data, statistics.get(columnPredicted));
	}

	private ImputationMethod getTestMethod (DataPoint dp, int columnPredicted, boolean missingPredictor) throws InvalidDataException {
		boolean prepareTraining = Boolean.parseBoolean(configManager.get("input.prepareTraining"));
		MainData data = new MainData(columnPredictors.clone(), columnPredicted, dp, columnPredictors.length > 1);

		if (prepareTraining) {
			if (data.isMultiple()) {
				DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 12, false); // for multiple
			} else {
				DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 14); // for simple
			}
			for (int i : columnPredictors) {
				if (missingPredictor || data.getTrain().getNumericColumn(i).countNaNs() > 0 || data.getImpute().getNumericColumn(i).countNaNs() > 0) {
					// the exception is thrown if at least one of the predictor in any data record is missing
					throw new InvalidDataException("Predictor is missing!");
				}
			}
		} else {
			Predicate<DataPoint> predictorsNotNull = (DataPoint x) -> IntStream.of(columnPredictors).noneMatch(col -> Double.isNaN(x.getNumericalValues().get(col)));
			data.setTrain(new SimpleDataSet(datasetMissing.getDataPoints().stream().filter(dps -> !Double.isNaN(dps.getNumericalValues().get(columnPredicted)) && predictorsNotNull.test(dps)).collect(Collectors.toList())));
			data.setImpute(new SimpleDataSet(datasetMissing.getDataPoints().stream().filter(dps -> Double.isNaN(dps.getNumericalValues().get(columnPredicted)) && predictorsNotNull.test(dps)).collect(Collectors.toList())));
		}

		if (!printOnlyFinal) {
			System.out.printf("Train: %d\nImpute: %d %n", data.getTrain().getSampleSize(), data.getImpute().getSampleSize());

		}

		// choose one of the following methods to test or some other
		return new MeanImputationMethod(data);
//		return new ClosestImputation(data);
//		return new MedianImputationMethod(data);
//		return new LinearRegressionJSATMethod(data);
//		return new MultiplePolynomialRegressionJSATMethod(data);
//		return new MultiplePolynomialRegressionJSATMethod(data, 2); // degree = 2
//		return new PolynomialCurveFitterApacheMethod(data, 2, 0); // degree = 2, index of predictor column = 0
//		return new PolynomialCurveFitterApacheMethod(data, 2, 1); // degree = 2, index of predictor column = 1
//		return new PolynomialCurveFitterApacheMethod(data, 2, 2); // degree = 2, index of predictor column = 2
//		return new PolynomialCurveFitterApacheMethod(data, 3, 0); // degree = 3, index of predictor column = 0
//		return new PolynomialCurveFitterApacheMethod(data, 3, 1); // degree = 3, index of predictor column = 1
//		return new PolynomialCurveFitterApacheMethod(data, 3, 2); // degree = 3, index of predictor column = 2
	}
}
