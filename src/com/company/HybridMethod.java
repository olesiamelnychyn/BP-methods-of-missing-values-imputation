package com.company;

import com.company.imputationMethods.*;
import com.company.utils.DatasetManipulation;
import com.company.utils.objects.ImputedValue;
import com.company.utils.objects.PerformanceMeasures;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

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
public class HybridMethod {
	SimpleDataSet datasetComplete;
	SimpleDataSet datasetMissing;
	int[] columnPredictors;
	Map<Integer, List<ImputedValue>> values = new HashMap<>(); //Map of <column, imputedValue>
	Map<Integer, Statistics> statistics = new HashMap<>(); //Map of <column, statistics>
	boolean printOnlyFinal;

	public HybridMethod (int[] columnPredictors, SimpleDataSet datasetComplete, SimpleDataSet datasetMissing, boolean printOnlyFinal) {
		this.datasetComplete = datasetComplete;
		this.datasetMissing = datasetMissing;
		this.columnPredictors = columnPredictors;
		this.printOnlyFinal = printOnlyFinal;
	}

	private void calcStatistics (int columnPredicted) {

		if (columnPredicted != -1) {
			Statistics statistic = columnPredictors.length == 1
					? new Statistics(datasetMissing, columnPredicted, columnPredictors[0])
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
						? new Statistics(datasetMissing, j, columnPredictors[0])
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
		//create Statistics of column/-s
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
		Statistics stat = statistics.get(columnPredicted);
		ImputationMethod method = null;
		if (columnPredictors.length > 1) { //if it is multiple regression
			if (hasLinearRelationship(datasets.get(0), stat)) {
				method = new MultipleRegressionJamaMethod(columnPredicted, datasets, columnPredictors);
			} else {
				method = new MultipleRegressionJamaMethod(columnPredicted, datasets, columnPredictors, 2);
			}
		} else { //if it is simple regression (only one predictor)
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
			evaluate_concat(columnPredicted, indexMissing, dp);
		}

		if (!printOnlyFinal) {
			System.out.println("\n");
		}
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

	private double getFormattedValue (double value) {
		if (Double.isNaN(value)) {
			return value;
		}
		return Double.parseDouble(df2.format(value).replace(',', '.'));
	}
}
