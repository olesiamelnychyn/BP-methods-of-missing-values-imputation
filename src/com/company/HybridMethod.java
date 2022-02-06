package com.company;

import com.company.imputationMethods.*;
import com.company.utils.DatasetManipulation;
import com.company.utils.Evaluation;
import com.company.utils.calculations.StatCalculations;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import java.io.IOException;
import java.util.*;

import static com.company.utils.ColorFormatPrint.*;
import static com.company.utils.calculations.MathCalculations.*;
import static com.company.utils.calculations.StatCalculations.*;
import static com.company.utils.objects.PerformanceMeasures.df2;

/**
 * This class perform the main logic of the program - predict missing values
 */
public class HybridMethod {
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal;
	Evaluation evaluation = null;

	public HybridMethod (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing, boolean printOnlyFinal) {
		if (datasetComplete != null) {
			evaluation = new Evaluation(datasetComplete, printOnlyFinal);
		}
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		this.printOnlyFinal = printOnlyFinal;
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

		if (evaluation != null) {
			evaluation.evaluateFinal(statistics);
		}
	}

	public void impute (DataPoint dp, int columnPredicted) {

		//training dataset and the one to be imputed
		ArrayList<SimpleDataSet> datasets;
		Statistics stat = statistics.get(columnPredicted);
		ImputationMethod method = null;
		if (columnPredictors.length > 1) { //if it is multiple regression
			datasets = DatasetManipulation.getToBeImputedAndTrainDeepCopiesByClosestDistance(datasetMissing, datasetMissing.getDataPoints().indexOf(dp), columnPredicted, columnPredictors, 10);
			if (hasLinearRelationship(datasets.get(0), stat)) {
				method = new MultipleRegressionJamaMethod(columnPredicted, datasets, columnPredictors);
			} else {
				method = new MultipleRegressionJamaMethod(columnPredicted, datasets, columnPredictors, 2);
			}
		} else { //if it is simple regression (only one predictor)
			datasets = DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(datasetMissing, datasetMissing.getDataPoints().indexOf(dp), columnPredicted, columnPredictors, 8);
			if (isCloseToMean(datasets.get(0), stat)) {
				method = new MeanImputationMethod(columnPredicted, datasets);
			} else if (isCloseToMedian(datasets.get(0), stat)) {
				method = new MedianImputationMethod(columnPredicted, datasets);
			} else if (isStrictlyIncreasing(datasets.get(0), columnPredicted) && isStrictlyIncreasing(datasets.get(0), columnPredictors[0])) {
				method = new LinearInterpolatorApacheMethod(columnPredicted, datasets, columnPredictors[0], true);
			} else if (isStrictlyDecreasing(datasets.get(0), columnPredicted) && isStrictlyDecreasing(datasets.get(0), columnPredictors[0])) {
				method = new LinearInterpolatorApacheMethod(columnPredicted, datasets, columnPredictors[0], false);
			} else if (hasLinearRelationship(datasets.get(0), stat)) {
				method = new LinearRegressionJSATMethod(columnPredicted, datasets, columnPredictors[0]);
			} else {
				int order = getPolynomialOrder(datasets.get(0), stat);
				if (order != -1) {
					method = new PolynomialCurveFitterApacheMethod(columnPredicted, datasets, columnPredictors[0], order);
				}
			}
		}

		if (method != null) {
			predict(method);
		} else {
			predict(new MeanImputationMethod(columnPredicted, datasets)); //default solution if everything fails to meet conditions
		}
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

			if (Double.isNaN(newValue) && !(imputationMethod instanceof MeanImputationMethod)) { // returned value is NaN, try default method - mean imputation
				ArrayList<SimpleDataSet> dataSets = new ArrayList<>();
				dataSets.add(imputationMethod.getTrainingCopy());
				dataSets.add(dataPoints);
				predict(new MeanImputationMethod(columnPredicted, dataSets));
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
