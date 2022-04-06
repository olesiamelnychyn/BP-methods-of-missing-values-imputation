package com.company;

import com.company.imputationMethods.ImputationMethod;
import com.company.imputationMethods.MeanImputationMethod;
import com.company.imputationMethods.MultipleImputationMethods;
import com.company.imputationMethods.SimpleImputationMethods;
import com.company.utils.Evaluation;
import com.company.utils.calculations.StatCalculations;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static com.company.utils.calculations.MathCalculations.*;
import static com.company.utils.objects.PerformanceMeasures.df2;

/**
 * This class perform the main logic of the program - predict missing values
 */
public class HybridMethod {
	ConfigManager configManager = ConfigManager.getInstance();
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal = Boolean.parseBoolean(configManager.get("input.printOnlyFinal"));
	Evaluation evaluation = null;
	MultipleImputationMethods multipleImputationMethods;
	SimpleImputationMethods simpleImputationMethods;

	public HybridMethod (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing) {
		if (datasetComplete != null) {
			evaluation = new Evaluation(datasetComplete, printOnlyFinal);
		}
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
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
		//create Statistics of column/-s
		statistics = StatCalculations.calcStatistics(columnPredicted, columnPredictors, datasetMissing);

		simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		multipleImputationMethods = new MultipleImputationMethods(datasetMissing, Boolean.parseBoolean(configManager.get("impute.weighted")), Boolean.parseBoolean(configManager.get("impute.useMultiDimensionalAsDefault")));

		for (DataPoint dp : datasetMissing.getDataPoints()) { //traverse dataset one by one
			int[] indexes = getIndexesOfNull(dp);
			boolean missingPredictor = getIntersection(indexes, columnPredictors).length > 0;
			int[] missingNonPredictors = getDifference(indexes, columnPredictors);

			if (columnPredicted != -1 && IntStream.of(missingNonPredictors).anyMatch(x -> x == columnPredicted)) { //if column to be imputed is specified
				impute(dp, columnPredicted, missingPredictor);
			} else { //all columns should be imputed
				for (int idx : missingNonPredictors) {
					impute(dp, idx, missingPredictor);
				}
			}
		}

		if (evaluation != null) {
			evaluation.evaluateFinal(statistics);
		}
	}

	public void impute (DataPoint dp, int columnPredicted, boolean missingPredictor) {
		MainData data = new MainData(!missingPredictor ? columnPredictors.clone() : new int[]{}, columnPredicted, dp);
		ImputationMethod method = null;

		if (data.getColumnPredictors().length > 1) { //if it is multiple regression
			method = multipleImputationMethods.imputeMultiple(data, statistics.get(columnPredicted));
		}

		if (method == null) { //if it is simple regression (none or only one predictor)
			method = simpleImputationMethods.imputeSimple(data, statistics.get(columnPredicted));
		}

		predict(method);
	}

	protected void predict (ImputationMethod imputationMethod) {
		imputationMethod.preprocessData();
		imputationMethod.fit();
		if (!printOnlyFinal) {
			imputationMethod.print();
		}

		int columnPredicted = imputationMethod.getColumnPredicted();
		SimpleDataSet dataPoints = imputationMethod.getToBePredicted();
		for (DataPoint dp : dataPoints.getDataPoints()) {
			int indexMissing = datasetMissing.getDataPoints().indexOf(dp);
			double newValue = imputationMethod.predict(dp);

			if (!(imputationMethod instanceof MeanImputationMethod) && (
				Double.isNaN(newValue)
					|| StatCalculations.isWithinMaxAndMin(newValue, statistics.get(columnPredicted).getMinAndMax())
					|| StatCalculations.isStepWithinMaxAndMinStep(newValue, imputationMethod.getData()))) {
				predict(new MeanImputationMethod(imputationMethod.getData()));
				continue;
			}

			dp.getNumericalValues().set(columnPredicted, getFormattedValue(newValue));
			if (evaluation != null) {
				evaluation.evaluate_concat(columnPredicted, indexMissing, dp);
			}
		}

		if (!printOnlyFinal) {
			System.out.println("\n");
		}
	}

	private double getFormattedValue (double value) {
		if (Double.isNaN(value)) {
			return value;
		}
		return Double.parseDouble(df2.format(value).replace(',', '.'));
	}
}
